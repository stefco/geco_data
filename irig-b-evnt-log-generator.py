#!/usr/bin/env python
# (c) Stefan Countryman 2017
# A script for generating a nominal entry for the EVNT log. Note that the text
# below is only true in the event that everything goes right!

import sys

if len(sys.argv) == 1:
    with open('graceid.txt', 'r') as f:
        graceid = f.read().strip()
elif '-h' in sys.argv:
    print('USAGE: {} GRACEID'.format(sys.argv[0]))
    exit(1)
else:
    graceid = sys.argv[1]

title = 'IRIG-B and DuoTone Timing Checks around {} Passed'.format(graceid)
body = ('Independent CNS-II GPS witness clocks at each end station and '
        'each site show IRIG-B timestamps in agreement with the '
        'timestamps written to frame for the 30 seconds surrounding the '
        '{} event. Note that the decoded Hanford times are 18 seconds '
        'later than the UTC time due to the fact that the Hanford site '
        'outputs its IRIG-B signal in GPS time. All output was as '
        'expected, consistent with a correct absolute timestamp at the '
        'time of {}.\n\n'
        'I also ran a python plotting script translated from Keita\'s '
        'duotone histogram script (found at $CalSVN/aligocalibration'
        '/trunk/Common/MatlabTools/timing/commissioningFrameDuotoneStat). '
        'The delays are within the 1us bounds, as '
        'needed.\n\n'
        'Additionally, DuoTone overlay plots (as seen and described in '
        'LIGO-T1600161) are attached. These plots show +/- 15 minutes of '
        'DuoTone signal surrounding the event with each second worth of data '
        'plotted on the same set of axes. The plots are visually '
        'indistinguishable from a single second worth of data, indicating that '
        'there were no large deviations from periodicity in the DuoTone '
        'signal.\n\n'
        'In conclusion, timing looks good.\n').format(graceid, graceid)

with open('event-log.txt', 'w') as f:
    f.write(title)
    f.write('\n\n')
    f.write(body)
