#!/usr/bin/env python
# (c) Stefan Countryman 2017

import matplotlib.pyplot as plt
import numpy as np
import geco_gwpy_dump as g
import gwpy.segments
import gwpy.time
import sys

if len(sys.argv) == 1:
    job = g.Job.load()
else:
    job = g.Job.load(sys.argv[1])

segs = gwpy.segments.DataQualityFlag.query_segdb('L1:DMT-ANALYSIS_READY:1',
                                                 job.start, job.end)

INDEX_MISSING_FMT = ('{} index not found for segment {} of {}, time {}\n'
                     'Setting {} index to {}.')
for i, q in enumerate(job.full_queries):
    means = []
    mins  = []
    maxs  = []
    stds  = []
    times = []
    t = q.read()
    for ii, s in enumerate(segs.active):
        # this next bit seems to be necessary due to a bug
        start = gwpy.time.to_gps(s.start).gpsSeconds
        end = gwpy.time.to_gps(s.end).gpsSeconds
        # the start index for this segment might be outside the full timeseries
        try:
            i_start = np.argwhere(t.times.value == (start // 60 * 60))[0][0]
        except IndexError:
            i_start = 0
            print(INDEX_MISSING_FMT.format('Start', ii, len(segs.active),
                                           start, 'start', i_start))
        # the end index for this segment might be outside the full timeseries
        try:
            i_end   = np.argwhere(t.times.value == (end // 60 * 60 + 60))[0][0]
        except IndexError:
            i_end   = -2
            print(INDEX_MISSING_FMT.format('End', ii, len(segs.active),
                                           end, 'end', i_end))
        tt = t[i_start:i_end+1]
        means.append( tt.mean().value )
        mins.append(  tt.min().value  )
        maxs.append(  tt.max().value  )
        stds.append(  tt.std().value  )
        times.append( tt.times.mean().value )
    f = plt.plot(times, means, "o'black'",
                 times, mins, "v'red'",
                 times, maxs, "^'blue'",
                 times, maxs-stds, "1'pink'",
                 times, maxs+stds, "2'teal'")
    f.set_title('{} from {} to {}'.format(t.channel.name,
                                          gwpy.time.from_gps(j.start),
                                          gwpy.time.from_gps(j.end)))
    f.savefig('{}__{}__{}.png'.format(q.start, q.end, q.sanitized_channel))
