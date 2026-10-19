#!/usr/bin/env python3
"""
wifiknight45 // Multi-Mode Sound Propagation Analyzer  (v4)

Real-time scrolling spectrogram (waterfall) + live spectrum with peak hold.

Keys:
  Space  pause / resume          C  cycle palette
  M      cycle processing mode   N  learn noise profile (2 s of "silence")
  R      start / stop WAV record S  save snapshot PNG
  B      toggle bird-band overlay V  reset view
"""
import sys
import time
from collections import deque

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import windows
import pyqtgraph as pg
# Use pyqtgraph's own Qt binding so widgets always match (PyQt5/PyQt6/PySide2/PySide6).
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# QShortcut lives in QtWidgets on Qt5 but in QtGui on Qt6
QShortcut = getattr(QtGui, 'QShortcut', None) or QtWidgets.QShortcut
Qt = QtCore.Qt

# --- Configuration ---------------------------------------------------------
SAMPLE_RATES = [16000, 22050, 44100, 48000]
FFT_SIZES = [512, 1024, 2048, 4096]
DEFAULT_SR = 22050
DEFAULT_FFT = 1024
HISTORY_LEN = 300          # waterfall rows
OVERLAP = 0.5              # 50 % overlap -> hop = FFT_SIZE * (1 - OVERLAP)
BIRD_BAND = (1000, 8000)   # Hz, typical songbird range
NOISE_LEARN_SECONDS = 2.0
EMPTY = -200.0             # "no data" fill value (clamps to darkest colour)

MODES = ["Standard dBFS", "Bioacoustic noise gate", "Learned noise profile"]

# --- Palettes --------------------------------------------------------------
BUILTIN_PALETTES = ['magma', 'viridis', 'inferno']
CUSTOM_PALETTES = {
    'turbo': [(48, 18, 59), (70, 107, 227), (40, 188, 235), (50, 241, 137),
              (163, 253, 60), (251, 190, 45), (238, 90, 18), (122, 4, 3)],
    'ocean': [(0, 0, 0), (0, 40, 90), (0, 120, 180), (80, 220, 230), (255, 255, 255)],
    'coolwarm': [(59, 76, 192), (141, 176, 254), (221, 221, 221), (244, 154, 123), (180, 4, 38)],
    'phosphor': [(0, 0, 0), (0, 60, 20), (0, 200, 70), (180, 255, 200)],
    'fire': [(0, 0, 0), (120, 0, 0), (255, 90, 0), (255, 220, 60), (255, 255, 255)],
    'binary': [(255, 255, 255), (0, 0, 0)],
}
PALETTES = BUILTIN_PALETTES + list(CUSTOM_PALETTES)


def build_lut(name):
    """Return a (256, 3) uint8 lookup table. Never depends on matplotlib."""
    try:
        if name in CUSTOM_PALETTES:
            cols = np.array(CUSTOM_PALETTES[name], dtype=np.ubyte)
            cmap = pg.ColorMap(np.linspace(0.0, 1.0, len(cols)), cols)
        else:
            cmap = pg.colormap.get(name)
        return cmap.getLookupTable(0.0, 1.0, 256, alpha=False)
    except Exception:
        return np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)


class WaterfallApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.sr = DEFAULT_SR
        self.fft_size = DEFAULT_FFT
        self.mode_index = 0
        self.paused = False
        self.dirty = True

        self.stream = None
        self.audio_status = None
        self.chunks = deque(maxlen=512)        # filled by audio thread, drained by GUI thread
        self.pending = np.zeros(0, dtype=np.float32)

        self.recording = False
        self.rec_chunks = []
        self.rec_start = 0.0

        self.learning = None                   # list of dB frames while learning, else None
        self.noise_profile = None
        self.last_db = None
        self.peak_freq = 0.0
        self.peak_db = EMPTY

        self.build_ui()
        self.build_shortcuts()
        self.setup_engine()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(33)                    # ~30 FPS is plenty for a 20-40 rows/s feed

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        self.setWindowTitle('wifiknight45 // Multi-Mode Sound Propagation Analyzer')
        self.resize(1200, 800)
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #111; color: #ddd; }"
            "QLabel { color: #ccc; }"
            "QToolBar { background: #111; border: none; spacing: 4px; }"
            "QPushButton { background: #222; border: 1px solid #444; padding: 4px 10px; }"
            "QPushButton:checked { background: #006655; border-color: #00FFCC; }"
            "QComboBox { background: #222; border: 1px solid #444; padding: 2px 6px; }"
        )

        # The plot widget is the window's central widget; all controls live in toolbars.
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground('#0a0a0a')
        self.setCentralWidget(self.graphics_layout)

        tb_info = self.addToolBar("Readout")
        self.addToolBarBreak()
        tb1 = self.addToolBar("Input")
        self.addToolBarBreak()
        tb2 = self.addToolBar("Display")
        for tb in (tb_info, tb1, tb2):
            tb.setMovable(False)
            tb.setFloatable(False)

        # Readout header
        self.info_label = QtWidgets.QLabel("Starting…")
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        font.setBold(True)
        self.info_label.setFont(font)
        self.info_label.setStyleSheet("color: #00FFCC; background: #000; padding: 8px;")
        self.info_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        tb_info.addWidget(self.info_label)

        # Control rows
        self.device_combo = QtWidgets.QComboBox()
        self.sr_combo = QtWidgets.QComboBox()
        self.fft_combo = QtWidgets.QComboBox()
        self.palette_combo = QtWidgets.QComboBox()
        self.mode_combo = QtWidgets.QComboBox()
        self.populate_devices()
        for sr in SAMPLE_RATES:
            self.sr_combo.addItem(f"{sr} Hz", sr)
        for n in FFT_SIZES:
            self.fft_combo.addItem(str(n), n)
        self.palette_combo.addItems(PALETTES)
        self.mode_combo.addItems(MODES)
        self.sr_combo.setCurrentIndex(self.sr_combo.findData(DEFAULT_SR))
        self.fft_combo.setCurrentIndex(self.fft_combo.findData(DEFAULT_FFT))
        for label, w in (("Input", self.device_combo), ("Rate", self.sr_combo),
                         ("FFT", self.fft_combo), ("Palette", self.palette_combo),
                         ("Mode", self.mode_combo)):
            tb1.addWidget(QtWidgets.QLabel(f" {label} "))
            tb1.addWidget(w)

        self.floor_slider = self.make_slider(-140, -30, -80)
        self.ceil_slider = self.make_slider(-80, 0, -30)
        self.floor_val = QtWidgets.QLabel()
        self.ceil_val = QtWidgets.QLabel()
        for s_ in (self.floor_slider, self.ceil_slider):
            s_.setMinimumWidth(160)
        for label, w in (("Floor", self.floor_slider), ("", self.floor_val),
                         ("Ceiling", self.ceil_slider), ("", self.ceil_val)):
            if label:
                tb2.addWidget(QtWidgets.QLabel(f" {label} "))
            tb2.addWidget(w)
        tb2.addSeparator()

        self.pause_btn = self.make_button("⏸ Pause", checkable=True)
        self.rec_btn = self.make_button("● Record", checkable=True)
        self.learn_btn = self.make_button("Learn noise")
        self.snap_btn = self.make_button("Snapshot")
        self.band_btn = self.make_button("Bird band", checkable=True)
        for b in (self.pause_btn, self.rec_btn, self.learn_btn, self.snap_btn, self.band_btn):
            tb2.addWidget(b)

        # Plots
        self.plot_item = self.graphics_layout.addPlot(row=0, col=0)
        self.plot_item.setLabel('left', 'Time ago', units='s')
        self.plot_item.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_item.invertY(True)                       # newest row at the top, scrolls down
        self.plot_item.setMouseEnabled(x=True, y=False)
        self.plot_item.getAxis('left').setWidth(70)
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        self.band_region = pg.LinearRegionItem(values=BIRD_BAND, movable=True,
                                               brush=pg.mkBrush(0, 255, 204, 25))
        self.band_region.setZValue(10)
        self.band_region.setVisible(False)
        self.plot_item.addItem(self.band_region, ignoreBounds=True)

        self.vline_wf = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 255, 255, 90), width=1))
        self.plot_item.addItem(self.vline_wf, ignoreBounds=True)

        self.spec_plot = self.graphics_layout.addPlot(row=1, col=0)
        self.spec_plot.setLabel('left', 'Level', units='dBFS')
        self.spec_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.spec_plot.setYRange(-120, 0, padding=0)
        self.spec_plot.setMouseEnabled(x=True, y=False)
        self.spec_plot.setXLink(self.plot_item)
        self.spec_plot.showGrid(x=True, y=True, alpha=0.25)
        self.spec_plot.getAxis('left').setWidth(70)
        self.peak_curve = self.spec_plot.plot(pen=pg.mkPen('#ff5555', width=1))
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen('#00FFCC', width=1.5),
                                              fillLevel=-120, brush=(0, 255, 204, 40))
        self.peak_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('#ffd500', style=Qt.PenStyle.DashLine))
        self.spec_plot.addItem(self.peak_line, ignoreBounds=True)
        self.vline_sp = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 255, 255, 90), width=1))
        self.spec_plot.addItem(self.vline_sp, ignoreBounds=True)
        self.graphics_layout.ci.layout.setRowStretchFactor(0, 3)
        self.graphics_layout.ci.layout.setRowStretchFactor(1, 1)

        # Status bar
        self.rec_label = QtWidgets.QLabel("")
        self.rec_label.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.statusBar().addPermanentWidget(self.rec_label)
        self.statusBar().showMessage(
            "Space pause | C palette | M mode | N learn noise | R record | S snapshot | B bird band | V reset view")

        self.mouse_proxy = pg.SignalProxy(self.plot_item.scene().sigMouseMoved,
                                          rateLimit=30, slot=self.on_mouse)

        # Initial state, then wire up signals
        self.apply_palette(0)
        self.update_slider_labels()
        for w in (self.device_combo, self.sr_combo, self.fft_combo, self.palette_combo,
                  self.mode_combo, self.pause_btn, self.rec_btn, self.learn_btn,
                  self.snap_btn, self.band_btn):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # keep Space/letters for shortcuts

        self.device_combo.currentIndexChanged.connect(self.setup_engine)
        self.sr_combo.currentIndexChanged.connect(self.setup_engine)
        self.fft_combo.currentIndexChanged.connect(self.setup_engine)
        self.palette_combo.currentIndexChanged.connect(self.apply_palette)
        self.mode_combo.currentIndexChanged.connect(self.set_mode)
        self.floor_slider.valueChanged.connect(self.on_levels_changed)
        self.ceil_slider.valueChanged.connect(self.on_levels_changed)
        self.pause_btn.toggled.connect(self.set_paused)
        self.rec_btn.toggled.connect(self.toggle_record)
        self.learn_btn.clicked.connect(self.start_learning)
        self.snap_btn.clicked.connect(self.snapshot)
        self.band_btn.toggled.connect(self.band_region.setVisible)

    def make_slider(self, lo, hi, val):
        s = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def make_button(self, text, checkable=False):
        b = QtWidgets.QPushButton(text)
        b.setCheckable(checkable)
        return b

    def build_shortcuts(self):
        def sc(key, fn):
            QShortcut(QtGui.QKeySequence(key), self).activated.connect(fn)
        sc('Space', self.pause_btn.toggle)
        sc('C', lambda: self.cycle(self.palette_combo))
        sc('M', lambda: self.cycle(self.mode_combo))
        sc('N', self.learn_btn.click)
        sc('R', self.rec_btn.toggle)
        sc('S', self.snap_btn.click)
        sc('B', self.band_btn.toggle)
        sc('V', self.reset_view)

    @staticmethod
    def cycle(combo):
        combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())

    def populate_devices(self):
        try:
            default_in = sd.default.device[0]
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0:
                    self.device_combo.addItem(f"{i}: {d['name']}", i)
            idx = self.device_combo.findData(default_in)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
        except Exception as exc:
            print(f"Could not list audio devices: {exc}", file=sys.stderr)
        if self.device_combo.count() == 0:
            self.device_combo.addItem("Default input", None)

    # -------------------------------------------------------------- engine
    def setup_engine(self, *_):
        """(Re)build all DSP buffers and restart the audio stream."""
        self.stop_stream()
        if self.recording:
            self.rec_btn.setChecked(False)         # finalize WAV at the old sample rate

        self.sr = self.resolve_sample_rate()
        self.fft_size = n = self.fft_combo.currentData()
        self.hop = max(1, int(n * (1 - OVERLAP)))
        self.window = windows.hann(n, sym=False).astype(np.float32)
        self.scale = 2.0 / float(self.window.sum())   # a full-scale sine reads 0 dBFS
        self.freqs = np.fft.rfftfreq(n, 1.0 / self.sr)
        self.num_bins = len(self.freqs)               # n//2 + 1  (rfft output length)
        self.bin_hz = self.sr / n
        self.nyquist = self.sr / 2.0
        self.row_dt = self.hop / self.sr
        self.span_s = HISTORY_LEN * self.row_dt

        self.wf = np.full((HISTORY_LEN, self.num_bins), EMPTY, dtype=np.float32)
        self.peak_hold = np.full(self.num_bins, EMPTY, dtype=np.float32)
        self.noise_profile = None
        self.learning = None
        self.last_db = None
        self.pending = np.zeros(0, dtype=np.float32)
        self.chunks.clear()

        # Map image pixels -> (Hz, seconds)
        self.image_item.setImage(self.wf.T, autoLevels=False, levels=self.current_levels())
        self.image_item.setRect(QtCore.QRectF(0, 0, self.nyquist, self.span_s))
        self.plot_item.setLimits(xMin=0, xMax=self.nyquist, yMin=0, yMax=self.span_s)
        self.reset_view()
        self.band_region.setBounds((0, self.nyquist))

        self.start_stream()
        self.dirty = True

    def resolve_sample_rate(self):
        """Use the requested rate if the device accepts it, else fall back to the device default."""
        want = self.sr_combo.currentData()
        dev = self.device_combo.currentData()
        try:
            sd.check_input_settings(device=dev, samplerate=want, channels=1)
            return want
        except Exception:
            pass
        try:
            default = int(sd.query_devices(dev, 'input')['default_samplerate'])
            sd.check_input_settings(device=dev, samplerate=default, channels=1)
        except Exception:
            return want                      # let start_stream report the real error
        nearest = min(SAMPLE_RATES, key=lambda r: abs(r - default))
        if nearest != default:
            return want
        self.sr_combo.blockSignals(True)
        self.sr_combo.setCurrentIndex(self.sr_combo.findData(nearest))
        self.sr_combo.blockSignals(False)
        self.statusBar().showMessage(f"{want} Hz not supported by this input — using {nearest} Hz", 6000)
        return nearest

    def start_stream(self):
        try:
            self.stream = sd.InputStream(
                device=self.device_combo.currentData(),
                samplerate=self.sr,
                channels=1,
                dtype='float32',
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            self.stream = None
            self.statusBar().showMessage(f"Audio error: {exc}", 8000)

    def stop_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        # Runs on the audio thread: keep it tiny and never touch shared numpy buffers.
        if status:
            self.audio_status = str(status)
        block = indata[:, 0].copy()
        self.chunks.append(block)                  # deque.append is thread-safe
        if self.recording:
            self.rec_chunks.append(block)

    # ---------------------------------------------------------------- DSP
    def process_audio(self):
        pieces = []
        while self.chunks:
            pieces.append(self.chunks.popleft())
        if self.paused:
            self.pending = np.zeros(0, dtype=np.float32)   # drop audio while frozen
            return
        if pieces:
            self.pending = np.concatenate([self.pending] + pieces)

        n = self.fft_size
        rows = []
        while len(self.pending) >= n:
            frame = self.pending[:n]
            self.pending = self.pending[self.hop:]
            mag = np.abs(np.fft.rfft(frame * self.window)) * self.scale
            db = (20.0 * np.log10(mag + 1e-10)).astype(np.float32)
            self.last_db = db
            np.maximum(self.peak_hold - 0.4, db, out=self.peak_hold)   # peak hold w/ decay
            self.feed_learning(db)
            rows.append(self.transform(db))

        if rows:
            k = min(len(rows), HISTORY_LEN)
            newest_first = np.asarray(rows[::-1][:k], dtype=np.float32)
            if k < HISTORY_LEN:
                self.wf[k:] = self.wf[:-k].copy()          # scroll older rows down
            self.wf[:k] = newest_first
            self.dirty = True

    def transform(self, db):
        if self.mode_index == 1:
            return db - (np.median(db) + 6.0)               # adaptive gate on spectrum median
        if self.mode_index == 2 and self.noise_profile is not None:
            return db - (self.noise_profile + 3.0)          # subtract learned per-bin noise floor
        if self.mode_index == 2:
            return db - (np.median(db) + 6.0)               # no profile yet -> behave like mode 1
        return db

    def current_levels(self):
        floor, ceil = self.floor_slider.value(), self.ceil_slider.value()
        if ceil - floor < 10:
            ceil = floor + 10
        return (floor, ceil) if self.mode_index == 0 else (0, ceil - floor)

    # --------------------------------------------------------- noise learn
    def start_learning(self):
        self.learning = []
        self.statusBar().showMessage(f"Learning noise profile for {NOISE_LEARN_SECONDS:.0f} s — keep the scene quiet…")

    def feed_learning(self, db):
        if self.learning is None:
            return
        self.learning.append(db)
        if len(self.learning) >= max(4, int(NOISE_LEARN_SECONDS / self.row_dt)):
            self.noise_profile = np.median(np.stack(self.learning), axis=0).astype(np.float32)
            self.learning = None
            self.mode_combo.setCurrentIndex(2)
            self.statusBar().showMessage("Noise profile captured — Learned noise profile mode active", 5000)

    # ----------------------------------------------------------- rendering
    def update_plot(self):
        if self.audio_status:
            self.statusBar().showMessage(f"Audio status: {self.audio_status}", 3000)
            self.audio_status = None

        self.process_audio()

        if self.recording:
            s = int(time.time() - self.rec_start)
            self.rec_label.setText(f"● REC {s // 60:02d}:{s % 60:02d}")

        if not self.dirty:
            return
        self.dirty = False

        self.image_item.setImage(self.wf.T, autoLevels=False, levels=self.current_levels())

        if self.last_db is not None:
            self.spec_curve.setData(self.freqs, self.last_db)
            self.peak_curve.setData(self.freqs, self.peak_hold)
            self.update_peak(self.last_db)

        learn = "  [LEARNING…]" if self.learning is not None else ""
        state = "  [PAUSED]" if self.paused else ""
        self.info_label.setText(
            f"Peak {self.peak_freq:8,.0f} Hz  {self.peak_db:6.1f} dBFS  |  "
            f"{MODES[self.mode_index]}  |  {PALETTES[self.palette_combo.currentIndex()]}  |  "
            f"{self.bin_hz:.1f} Hz/bin, {self.row_dt * 1000:.0f} ms/row{state}{learn}"
        )

    def update_peak(self, db):
        lo = int(np.searchsorted(self.freqs, 50.0))        # ignore DC / rumble
        if lo >= len(db) - 1:
            return
        k = lo + int(np.argmax(db[lo:]))
        delta = 0.0
        if 0 < k < len(db) - 1:                            # parabolic interpolation for sub-bin accuracy
            a, b, c = db[k - 1], db[k], db[k + 1]
            denom = a - 2 * b + c
            if denom != 0:
                delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        self.peak_freq = (k + delta) * self.bin_hz
        self.peak_db = float(db[k])
        self.peak_line.setPos(self.peak_freq)

    # -------------------------------------------------------------- events
    def on_mouse(self, evt):
        pos = evt[0]
        if not self.plot_item.sceneBoundingRect().contains(pos):
            return
        pt = self.plot_item.vb.mapSceneToView(pos)
        f, t = pt.x(), pt.y()
        self.vline_wf.setPos(f)
        self.vline_sp.setPos(f)
        row, b = int(t / self.row_dt), int(f / self.bin_hz)
        if 0 <= row < HISTORY_LEN and 0 <= b < self.num_bins:
            val = self.wf[row, b]
            txt = "—" if val <= -150 else f"{val:+.1f} dB"
            self.statusBar().showMessage(f"Cursor: {f:,.0f} Hz | {t:.2f} s ago | {txt}")

    def on_levels_changed(self, *_):
        self.update_slider_labels()
        self.dirty = True

    def update_slider_labels(self):
        self.floor_val.setText(f"{self.floor_slider.value()} dB")
        self.ceil_val.setText(f"{self.ceil_slider.value()} dB")

    def apply_palette(self, index):
        self.image_item.setLookupTable(build_lut(PALETTES[index]))
        self.dirty = True

    def set_mode(self, index):
        self.mode_index = index
        self.wf.fill(EMPTY)                                # old rows use different units
        self.dirty = True
        if index == 2 and self.noise_profile is None:
            self.statusBar().showMessage("No noise profile yet — press N in a quiet moment.", 5000)

    def set_paused(self, on):
        self.paused = on
        self.pause_btn.setText("▶ Resume" if on else "⏸ Pause")
        self.dirty = True

    def reset_view(self):
        self.plot_item.setRange(xRange=(0, self.nyquist), yRange=(0, self.span_s), padding=0)
        self.plot_item.disableAutoRange()

    def toggle_record(self, on):
        if on:
            self.rec_chunks = []
            self.rec_start = time.time()
            self.recording = True
            self.rec_btn.setText("■ Stop")
            return
        self.recording = False
        self.rec_btn.setText("● Record")
        self.rec_label.setText("")
        chunks, self.rec_chunks = self.rec_chunks, []
        if not chunks:
            return
        pcm = (np.clip(np.concatenate(chunks), -1.0, 1.0) * 32767).astype(np.int16)
        name = time.strftime('recording_%Y%m%d_%H%M%S.wav')
        wavfile.write(name, self.sr, pcm)
        self.statusBar().showMessage(f"Saved {name} ({len(pcm) / self.sr:.1f} s)", 6000)

    def snapshot(self):
        name = time.strftime('waterfall_%Y%m%d_%H%M%S.png')
        if self.graphics_layout.grab().save(name):
            self.statusBar().showMessage(f"Saved {name}", 5000)
        else:
            self.statusBar().showMessage("Snapshot failed", 5000)

    def closeEvent(self, event):
        self.timer.stop()
        if self.recording:
            self.rec_btn.setChecked(False)
        self.stop_stream()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = WaterfallApp()
    window.show()
    sys.exit(app.exec())
