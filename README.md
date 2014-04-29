3D-maze
=======

First person maze written in Python with the module Tkinter.  


Required Software
-----------------
3D-maze relies on Python 2.7.  Python should already be installed on Linux and
most Macs, but you can download it for any system at
[python.org](https://www.python.org/download/releases/2.7).

This application relies on the python module Tkinter.  If you have python
installed, Tkinter should already be on your computer.

This can be run with the standard python2.7 interpreter, but it
runs faster when using pypy, which is not installed by default.  Instead, pypy
can be found at [pypy.org](http://pypy.org/download.html).

It is strongly suggested to have the font Helvetica installed on your computer.
If it's not present, Tkinter will choose a similar font that is installed, but
there's a chance an error may be thrown.



Installation
------------
If you have Linux installed, from the command line execute `./maze.py`.  For
Max and Windows, read the documentation on pypy or whichever interpreter you
choose.

By default, 3D-maze will run in 1366x768 and generate a maze with 15x15 walls.
To change this, edit the line `game = MazeGame(15,1366,768)`.  Replace `15`
with the desired width and height of the maze; replace 1366 with the desired
width of the window, and replace 768 with the desired height of the window.

In order to change the amount of "cycles" the maze contains, edit the global
`CYCLE_AMOUNT` at the top of the file AT YOUR OWN RISK.  It must be a
non-negative integer.  A higher number indicates a lower probability of a
cycle in the maze.



Features
--------
3D-maze has three modes: 
* a 2D top-down mode where only portions of walls visible in 360° are shown
* a 3D first-person mode
* a 3D red-cyan anaglyph mode 
In the first two modes, the segments are colored from a blueish to bright
green.  The goal is to follow the direction of increasing green and reach the
white cell in the far corner of the maze.
