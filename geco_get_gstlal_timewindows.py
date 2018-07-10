#!/usr/bin/env python
# (c) Stefan Countryman (2018)

r"""
USAGE: geco_get_gstlal_timewindows.py TRIGGER_DIR FILENAME_PATTERN NUMBER_OF_LISTS DELTAT

Get a list of times from a GSTLAL trigger directory whose trigger directories
have names like:

gstlal-offline-1171612765_+4.118e-05

Where the first number is the GPS event time and the second number is the False
Alarm Rate.

For example:

geco_get_gstlal_timewindows.py \
    "/rigel/geco/users/shared/gstlal-skymaps/eventsdir" \
    "/rigel/home/stc2117/dev/geco_data/slurm-jobs/static/raw-frame-times-v3-{}.txt" \
    3 \
    5
"""

import os
import sys

if "-h" in sys.argv or "--help" in sys.argv or len(sys.argv) != 5:
    print(__doc__.strip())
    exit()

trigger_dir = sys.argv[1]
filename_pattern = sys.argv[2]
number_of_lists = int(sys.argv[3])
deltat = int(sys.argv[4])

times = [int(d.split("-")[2].split("_")[0])
         for d in os.listdir("/rigel/geco/users/shared/gstlal-skymaps/eventsdir")
         if d.startswith("gstlal-offline-")]
print("Getting triggers from: {}".format(trigger_dir))
print("Number of triggers: {}".format(len(times)))
print("Using filename pattern: {}".format(filename_pattern))
print("Splitting into {} lists of files.".format(number_of_lists))
print("Going +/- {} seconds around each event.".format(deltat))
slices = [i*len(times)//number_of_lists for i in range(number_of_lists)] + [len(times)]
for i in range(number_of_lists):
    filename = filename_pattern.format(i+1)
    print("Writing to file: {}".format(filename))
    with open(filename, 'w') as outf:
        for time in times[slices[i]:slices[i+1]]:
            outf.write("{} {}\n".format(time-deltat, time+deltat))
