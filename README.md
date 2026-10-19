# waterfaal

Multi Mode Sound Propagation Analyzer


A real time audio spectrogram for your microphone. It draws a scrolling waterfall display with a live spectrum underneath, and is tuned for spotting faint bird calls and other transient sounds.

Requirements
============

Python 3.8 or newer, plus these packages:

    pip install PyQt5 pyqtgraph scipy numpy sounddevice

On Linux you may also need the PortAudio library:

    sudo apt install libportaudio2

Running
=======

    python waterfall.py

If you have more than one Qt binding installed, you can pin one:

    PYQTGRAPH_QT_LIB=PyQt5 python waterfall_analyzer.py

Features
========

* Waterfall display with frequency in Hz and time in seconds
* Live spectrum with decaying peak hold and a peak frequency readout
* Selectable input device, sample rate and FFT size
* Three processing modes: Standard dBFS, Bioacoustic noise gate, and Learned noise profile
* Ten color palettes
* Floor and ceiling sliders to adjust contrast live
* Hover cursor showing frequency, age and level
* Bird band overlay, WAV recording and PNG snapshots

Keyboard Shortcuts
==================

    Space   Pause or resume
    C       Cycle color palette
    M       Cycle processing mode
    N       Learn a noise profile (keep the room quiet for 2 seconds)
    R       Start or stop WAV recording
    S       Save a PNG snapshot
    B       Toggle the bird band overlay
    V       Reset the view

Tips
====

For faint sounds, press N during a quiet moment. The app switches to the Learned noise profile mode, which subtracts the background noise at each frequency.

Recordings and snapshots are saved in the folder you launched the script from, named with a timestamp.

If your input device rejects the chosen sample rate, the app falls back to the device default and shows a message in the status bar.
