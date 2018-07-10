#!/usr/bin/env python
# (c) Stefan Countryman (2018)

"""
Get a list of times from a GSTLAL trigger directory whose trigger directories
have names like:

gstlal-offline-1171612765_+4.118e-05

Where the first number is the GPS event time and the second number is the False
Alarm Rate.
"""

import os
import sys

if "-h" in sys.argv or "--help" in sys.argv or len(sys.argv) == 1:
    print("Usage: {} FILENAME_PATTERN NUMBER_OF_LISTS "
          "DELTAT".format(sys.argv[0]))
    print()
    print(__doc__)
    exit()

filename_pattern = sys.argv[1]
number_of_lists = int(sys.argv[2])
deltat = int(sys.argv[3])

times = [int(d.split("-")[2].split("_")[0])
         for d in os.listdir("/rigel/geco/users/shared/gstlal-skymaps/eventsdir")
         if d.startswith("gstlal-offline-")]
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
