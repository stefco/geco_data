#!/usr/bin/env python
# (c) Stefan Countryman 2018

"""
Take a directory of filenames with name format
`gstlal-offline-1164686041_+7.264e-09` and pick out the event times from those
filenames. Then, return a list of start/stop times of the sort accepted by
`geco_fetch_frame_files.py --times`.
"""

from __future__ import print_function

import sys
import os
from glob import glob

USAGE = """{} DIRECTORY-HOLDING-FILENAMES OUTFILE1 [OUTFILE2 ... OUTFILEN]

Print a space-delimited list of START/STOP times for windows surrounding the
event and save them to the provided output filenames, splitting the time
windows (roughly) evenly between output filenames. If no outfiles are provided,
write to STDOUT.

NOTE that OUTFILES WILL BE OVERWRITTEN.
""".format(os.path.basename(sys.argv[0]))
DEFAULT_DELTA_T = 32

if not sys.argv[1:] or "-h" in sys.argv or "--help" in sys.argv:
    print(USAGE)
    exit()


def eventtimes(eventsdir):
    """Parse a list of times from matching file globs in directory
    `eventsdir` and return a generator providing those names."""
    return (int(os.path.basename(f).split("_")[0].split("-")[-1]) for f in
            glob(os.path.join(sys.argv[1], "gstlal-offline-*")))


def searchwindows(etimes, deltat=DEFAULT_DELTA_T):
    """Get an iterator of start/stop tuples for search windows around a given
    iterable of event times looking +/- `deltat` seconds around each
    event."""
    return ((t-deltat, t+deltat) for t in etimes)


def split(windows, num):
    """Split an iterable of windows into `num` sublists of (roughly) equal
    length. Return a list of these sublists."""
    if num == 1:
        return [windows]
    windows = windows if isinstance(windows, list) else list(windows)
    length = len(windows) // num
    return [windows[i*length:(i+1)*length] for i in range(num)]


def main():
    outfiles = [open(f, 'w') for f in sys.argv[2:]] or sys.stdout
    windowlists = split(searchwindows(eventtimes(sys.argv[1])), len(outfiles))
    for i, out in enumerate(outfiles):
        out.writelines("{} {}\n".format(s, e) for s, e in windowlists[i])


if __name__ == "__main__":
    main()
