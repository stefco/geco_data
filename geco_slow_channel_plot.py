#!/usr/bin/env python
# (c) Stefan Countryman 2017

import matplotlib.pyplot as plt
import numpy as np
import geco_gwpy_dump
import gwpy.segments
import gwpy.time
import collections
import json
import sys

DEFAULT_TREND = ''
DEFAULT_PLOT_FILETYPE = 'png'

if __name__ == '__main__':
    if len(sys.argv) == 1:
        job = geco_gwpy_dump.Job.load()
    else:
        job = geco_gwpy_dump.Job.load(sys.argv[1])

class PlottingJob(object):
    """A description of the plots that need to be made along with methods
    for making them. Has a similar interface to ``geco_gwpy_dump.Job``, and
    in fact contains a ``Job`` as one of its properties. Uses a superset
    of the data contained in a ``Job`` job specification JSON file. The
    ``geco_gwpy_dump.Job`` specifies what data is necessary to generate
    the plots, while instances of this class provides specific information on
    how to make those plots."""
    def __init__(self, job, run, channel_descriptions, dq_flag_channel_pairs):
        self.job = job
        self.run = run
        self.channel_descriptions = channel_descriptions
        self.dq_flag_channel_pairs = dq_flag_channel_pairs
    def to_dict(self):
        """Return a dict representation of this object."""
        job_dict = self.job.to_dict()
        # we modify the basic job_dict by throwing in an extra dictionary
        # with plotting-specific information
        plot_dict = {'run': self.run,
                     'channel_descriptions': self.channel_descriptions,
                     'dq_flag_channel_pairs': self.dq_flag_channel_pairs}
        job_dict['slow_channel_plots'] = plot_dict
        return job_dict
    def save(self, jobspecfile):
        """Save a JSON representation of this object."""
        with open(jobspecfile, 'w') as f:
            json.dump(self.to_dict(), f)
    @classmethod
    def from_dict(cls, d):
        """Instantiate a PlottingJob from a suitable dictionary
        representation."""
        plot_dict = d['slow_channel_plots']
        return cls(job = geco_gwpy_dump.Job.from_dict(d),
                   run = plot_dict['run'],
                   channel_descriptions = plot_dict['channel_descriptions'],
                   dq_flag_channel_pairs = plot_dict['dq_flag_channel_pairs'])
    @classmethod
    def load(cls, jobspecfile='jobspec.json'):
        """load this PlottingJob from a job specification file, assumed to be
        formatted in JSON."""
        with open(jobspecfile, "r") as f:
            return cls.from_dict(json.load(f))
    @property
    def start(self):
        return self.job.start
    @property
    def end(self):
        return self.job.end
    @property
    def channels(self):
        return self.job.channels
    @property
    def exts(self):
        return self.job.exts
    @property
    def dq_flags(self):
        return self.job.dq_flags
    @property
    def trends(self):
        return self.job.trends
    @property
    def plotters(self): #TODO
    def make_individual_plots(self): #TODO
    def combined_plotters(self): #TODO
    def make_combined_plots(self): #TODO

class Plotter(object):
    """Defines the parameters of a specific slow channel plot and provides
    methods for working with that plot's data and generating the plot
    itself. ``run`` and ``channel_description`` arguments are used to help
    clarify the meaning of the plot axes; they can be omitted if desired. This
    plot is only meant to show a single statistic for a single channel.
    
    ``start``:
        GPS Start time of the plotted period.

    ``end``:
        GPS End time of the plotted period.

    ``channel``:
        The channel name without trend extensions, e.g.
        ``"L1:SYS-TIMING_C_MA_A_PORT_2_SLAVE_CFC_TIMEDIFF_2"``.

    ``dq_flag``:
        The DataQualityFlag name specifying which subintervals of the full
        time interval should actually be plotted, e.g.
        ``"L1:DMT-ANALYSIS_READY:1"``

    ``trend``:
        Trend extension to use for this plot. DEFAULT: ``""``

    ``ext``:
        File type for saved timeseries data. Defaults to the same as
        ``geco_gwpy_dump.DEFAULT_EXTENSION[0]``.

    ``run``:
        Name of the run to be plotted, e.g. "O1". OPTIONAL.

    ``channel_description``:
        A human-readable description of this channel. OPTIONAL."""
    def __init__(self, start, end, channel, dq_flag, trend=DEFAULT_TREND,
                 ext=geco_gwpy_dump.DEFAULT_EXTENSION,
                 run=None, channel_description=None):
        if channel_description is None:
            channel_description = channel
        self.start = start
        self.end = end
        self.channel = channel
        self.dq_flag = dq_flag
        self.trend = trend
        self.ext = ext
        self.run = run
        self.channel_description = channel_description
    @property
    def job(self):
        """Get the ``geco_gwpy_dump.Job`` corresponding to this ``Plotter``."""
        return geco_gwpy_dump.Job(start = self.start, end = self.end,
                                  channels = [self.channel], exts = [self.ext],
                                  dq_flags = [self.dq_flag],
                                  trends = [self.trend])
    @property
    def query(self):
        """Get the ``geco_gwpy_dump.Query`` corresponding to this
        ``Plotter``."""
        return self.job.full_queries[0]
    @property
    def dq_segments(self):
        """Get the ``gwpy.segments.DataQualityFlag`` time segments
        corresponding to this ``Plotter``."""
        return self.job.get_dq_segments()[self.dq_flag]
    def read(self):
        """Read a list of timeseries for this channel corresponding to the
        time intervals when this Plotter's ``DataQualityFlag`` was active."""
        return self.query.read_and_split_into_segments(self.dq_segments)
    # define a named tuple for returning the statistics
    Stats = collections.namedtuple('Stats', ['ts', 'means', 'mins', 'maxs',
                                             'stds', 'times', 'ns'])
    def stats(self):
        """Return the following statistics on the timeseries with a value
        calculated for each time segments when this Plotter's
        ``DataQualityFlag`` was active:
        
        - means (means)
        - mins (mins)
        - maxs (maxs)
        - standard deviations (stds)
        - central times (times)
        - number of sample values per segment (ns)"""
        ts = self.read()
        means = np.array([t.mean().value for t in ts])
        mins  = np.array([t.min().value for t in ts])
        maxs  = np.array([t.max().value for t in ts])
        stds  = np.array([t.std().value for t in ts])
        times = np.array([t.times.mean().value for t in ts])
        ns    = np.array([len(t) for t in ts])
        return self.Stats(ts=ts, means=means, mins=mins, maxs=maxs, stds=stds,
                          times=times, ns=ns)
    @property
    def fname(self):
        """Get the filename for this plot."""
        return '{}.{}'.format(self.query.fname, DEFAULT_PLOT_FILETYPE)
    def get_plot(self, fig=None):
        """Generate a ``matplotlib.figure.Figure`` for the channel and
        dq_flag specified in this ``Plotter``. Optionally pass an existing
        figure as an argument to plot to that figure's axes."""
        if fig is None:
            fig = plt.figure()
        # plot everything
        f.gca().plot(times, means, marker="o", color="black")
        f.gca().plot(times, mins, marker="v", color="red")
        f.gca().plot(times, maxs, marker="^", color="blue")
        f.gca().plot(times, maxs-stds, marker="1", color="pink")
        f.gca().plot(times, maxs+stds, marker="2", color="teal")
        # come up with a title
        start = gwpy.time.from_gps(self.start)
        end = gwpy.time.from_gps(self.end)
        if self.run is None:
            fmt = '{} from {} to {}\nduring {} Segments'
            title = fmt.format(self.channel_description, start, end, self.dq_flag)
        else:
            fmt = '{} from {} to {}\nduring {} Segments for {}'
            title = fmt.format(self.channel_description, start, end, self.dq_flag,
                            self.run)
        f.gca().set_title(title)
        return f
    def save_plot(self):
        """Save this plot as an image file."""
        self.get_plot().savefig(self.fname)

class CombinedPlotter(object): #TODO

#INDEX_MISSING_FMT = ('{} index not found for segment {} of {}, time {}\n'
#                     'Setting {} index to {}.')
#for i, q in enumerate(job.full_queries):
#   means = np.ndarray(len(segs.active))
#   mins  = np.ndarray(len(segs.active))
#   maxs  = np.ndarray(len(segs.active))
#   stds  = np.ndarray(len(segs.active))
#   times = np.ndarray(len(segs.active))
#   t = q.read()
#   for ii, s in enumerate(segs.active):
#       # this next bit seems to be necessary due to a bug; IIRC, one time
#       # value might appear as text data rather than numerical data, forcing
#       # this stupid kludgy conversion.
#       start = gwpy.time.to_gps(s.start).gpsSeconds
#       end = gwpy.time.to_gps(s.end).gpsSeconds
#       # the start index for this segment might be outside the full timeseries
#       try:
#           i_start = np.argwhere(t.times.value == (start // 60 * 60))[0][0]
#       except IndexError:
#           i_start = 0
#           print(INDEX_MISSING_FMT.format('Start', ii, len(segs.active),
#                                          start, 'start', i_start))
#       # the end index for this segment might be outside the full timeseries
#       try:
#           i_end   = np.argwhere(t.times.value == (end // 60 * 60 + 60))[0][0]
#       except IndexError:
#           # just pick the index of the last value in t.times
#           i_end   = len(t.times) - 1
#           print(INDEX_MISSING_FMT.format('End', ii, len(segs.active),
#                                          end, 'end', i_end))
#       tt = t[i_start:i_end+1]
#       means[ii] = tt.mean().value
#       mins[ii]  = tt.min().value
#       maxs[ii]  = tt.max().value
#       stds[ii]  = tt.std().value
#       times[ii] = tt.times.mean().value
#   f = plt.figure(i)
#   f.gca().plot(times, means, marker="o", color="black")
#   f.gca().plot(times, mins, marker="v", color="red")
#   f.gca().plot(times, maxs, marker="^", color="blue")
#   f.gca().plot(times, maxs-stds, marker="1", color="pink")
#   f.gca().plot(times, maxs+stds, marker="2", color="teal")
#   f.gca().set_title('{} from {} to {}'.format(t.channel.name,
#                                         gwpy.time.from_gps(j.start),
#                                         gwpy.time.from_gps(j.end)))
#   f.savefig('{}__{}__{}.png'.format(q.start, q.end, q.sanitized_channel))
