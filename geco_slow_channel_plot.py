#!/usr/bin/env python
# (c) Stefan Countryman 2017

import matplotlib
if __name__ == '__main__':
    # necessary for headless plotting
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import geco_gwpy_dump
import gwpy.segments
import gwpy.time
import collections
import json
import sys
import abc

DEFAULT_TREND = ''
DEFAULT_PLOT_FILETYPE = 'png'
COMBINED_TRENDS = [
    ".mean,m-trend",
    ".min,m-trend",
    ".max,m-trend",
    ".rms,m-trend",
    ".n,m-trend"
]

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
    def individual_plotters(self):
        """Return a list of ``IndividualPlotter``s for this job."""
        plotters = []
        for trend in self.trends:
            for dq_flag_channel_pair in self.dq_flag_channel_pairs:
                dq_flag = dq_flag_channel_pair[0]
                channel = dq_flag_channel_pair[1]
                desc = self.channel_descriptions[channel]
                # only need the first extension since any should have saved data
                p = IndividualPlotter(start = self.start, end = self.end,
                                      channel = channel, dq_flag = dq_flag,
                                      trend = trend, ext = self.exts[0],
                                      run = self.run,
                                      channel_description = desc)
                plotters.append(p)
        return plotters
    def make_individual_plots(self):
        """Save all individual plots, i.e. every channel and every dq_flag gets
        its own plot."""
        for plotter in self.individual_plotters:
            plotter.save_plot()
    @property
    def combined_plotters(self):
        """Return a list of ``CombinedPlotter``s for this job."""
        plotters = []
        for dq_flag_channel_pair in self.dq_flag_channel_pairs:
            dq_flag = dq_flag_channel_pair[0]
            channel = dq_flag_channel_pair[1]
            desc = self.channel_descriptions[channel]
            # only need the first extension since any should have saved data
            p = CombinedPlotter(start = self.start, end = self.end,
                                channel = channel, dq_flag = dq_flag,
                                ext = self.exts[0], run = self.run,
                                channel_description = desc)
            plotters.append(p)
        return plotters
    def make_combined_plots(self):
        """Save all individual plots, i.e. every channel and every dq_flag gets
        its own plot."""
        for plotter in self.combined_plotters:
            plotter.save_plot()

class Plotter(object):
    """An abstract class for defining plotting jobs."""
    __metaclass__ = abc.ABCMeta
    @property
    def job(self):
        """Get the ``geco_gwpy_dump.Job`` corresponding to this ``Plotter``."""
        return geco_gwpy_dump.Job(start = self.start, end = self.end,
                                  channels = [self.channel], exts = [self.ext],
                                  dq_flags = [self.dq_flag],
                                  trends = self.trends)
    @property
    def queries(self):
        """Get the ``geco_gwpy_dump.Query`` objects corresponding to this
        ``Plotter``."""
        return self.job.full_queries
    # define a named tuple for returning the statistics
    Stats = collections.namedtuple('Stats', ['means', 'mins', 'maxs',
                                             'stds', 'times', 'ns'])
    @property
    def stats(self):
        """Return a dictionary with channel/trend combinations as the keys
        and values corresponding to the following statistics on that
        channel/trend's timeseries, where the values are calculated for each
        time segments when this Plotter's ``DataQualityFlag`` was active:
        
        - means (means)
        - mins (mins)
        - maxs (maxs)
        - standard deviations (stds)
        - central times (times)
        - number of sample values per segment (ns)"""
        ts = self.read()
        stats = {}
        for q in self.queries:
            ch = q.channel
            means = np.array([t.mean().value for t in ts[ch]])
            mins  = np.array([t.min().value for t in ts[ch]])
            maxs  = np.array([t.max().value for t in ts[ch]])
            stds  = np.array([t.std().value for t in ts[ch]])
            times = np.array([t.times.mean().value for t in ts[ch]])
            ns    = np.array([len(t) for t in ts[ch]])
            s = self.Stats(means=means, mins=mins, maxs=maxs, stds=stds,
                           times=times, ns=ns) 
            stats[ch] = s
        return stats
    @property
    def dq_segments(self):
        """Get the ``gwpy.segments.DataQualityFlag`` time segments
        corresponding to this ``Plotter``."""
        return self.job.get_dq_segments()[self.dq_flag]
    def save_plot(self):
        """Save this plot as an image file."""
        self.get_plot().savefig(self.fname)
    def read(self):
        """Read a dict of lists of timeseries for this channel corresponding to
        the channel/trend combinations loaded time intervals when this
        Plotter's ``DataQualityFlag`` was active. The dictionary key is just
        the full channel/trend combination."""
        ts = {}
        for q in self.queries:
            ts[q.channel] = q.read_and_split_into_segments(self.dq_segments)
        return ts
    @abc.abstractproperty
    def fname(self):
        """Return the filename for this plot"""
    @abc.abstractmethod
    def get_plot(self, fig=None):
        """Generate a ``matplotlib.figure.Figure`` for the channel and
        dq_flag specified in this ``Plotter``. Optionally pass an existing
        figure as an argument to plot to that figure's axes."""

class IndividualPlotter(Plotter):
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
        self.trends = [trend]
        self.ext = ext
        self.run = run
        self.channel_description = channel_description
    @property
    def fname(self):
        """Get the filename for this plot."""
        return '{}.{}'.format(self.queries[0].fname, DEFAULT_PLOT_FILETYPE)
    def get_plot(self, fig=None):
        """Generate a ``matplotlib.figure.Figure`` for the channel and
        dq_flag specified in this ``Plotter``. Optionally pass an existing
        figure as an argument to plot to that figure's axes."""
        if fig is None:
            fig = plt.figure()
        # get the statistics for this one channel (the only one we will plot)
        s = self.stats[self.queries[0].channel]
        # plot everything
        mean = fig.gca().errorbar(s.times, s.means, marker="o", color="black",
                                  yerr=s.stds)
        mins = fig.gca().plot(s.times, s.mins, marker="v", color="red")
        maxs = fig.gca().plot(s.times, s.maxs, marker="^", color="blue")
        # come up with a title
        start = gwpy.time.from_gps(self.start)
        end = gwpy.time.from_gps(self.end)
        if self.run is None:
            fmt = '{} from {} to {}\nduring {} Segments (trend: {})'
            title = fmt.format(self.channel_description, start, end,
                               self.dq_flag, self.trends[0])
        else:
            fmt = '{} from {} to {}\nduring {} Segments for {} (trend: {})'
            title = fmt.format(self.channel_description, start, end,
                               self.dq_flag, self.run, self.trends[0])
        fig.legend(handles=[mean, mins, maxs], labels=["Means +/- Std. Dev.",
                                                       "Minima", "Maxima"])
        fig.gca().set_title(title)
        return fig

class CombinedPlotter(Plotter): #TODO
    """A class for plotting a slow channel with all trends. Used when all trends are available to generate a combined plot for a channel over some period of time."""
    def __init__(self, start, end, channel, dq_flag,
                 ext=geco_gwpy_dump.DEFAULT_EXTENSION,
                 run=None, channel_description=None):
        if channel_description is None:
            channel_description = channel
        self.start = start
        self.end = end
        self.channel = channel
        self.dq_flag = dq_flag
        self.trends = COMBINED_TRENDS
        self.ext = ext
        self.run = run
        self.channel_description = channel_description
    def fname(self):
        """Return the filename for the saved plot image."""
        ch = self.queries[0].sanitized_channel
        return '{}__{}__{}.combined.{}'.format(self.start, self.end, ch,
                                               DEFAULT_PLOT_FILETYPE)
    def get_plot(self, fig=None):
        """Generate a ``matplotlib.figure.Figure`` for the channel and
        dq_flag specified in this ``CombinedPlotter``. Optionally pass an
        existing figure as an argument to plot to that figure's axes."""
        if fig is None:
            fig = plt.figure()
        # calculate statistics for each channel and dq_flag active segment
        s = self.stats
        absmaxs = s[self.channel + '.max,m-trend'].maxs
        absmins = s[self.channel + '.min,m-trend'].mins
        means   = s[self.channel + '.mean,m-trend'].means
        times   = s[self.channel + '.mean,m-trend'].times
        stds    = s[self.channel + '.mean,m-trend'].stds
        # plot everything
        mean = fig.gca().errorbar(times, means, marker="o", color="black",
                                  yerr=stds)
        mins = fig.gca().plot(times, absmins, marker="v", color="red")
        maxs = fig.gca().plot(times, absmaxs, marker="^", color="blue")
        # come up with a title
        start = gwpy.time.from_gps(self.start)
        end = gwpy.time.from_gps(self.end)
        if self.run is None:
            fmt = '{} from {} to {}\nduring {} Segments'
            title = fmt.format(self.channel_description, start, end,
                               self.dq_flag)
        else:
            fmt = '{} from {} to {}\nduring {} Segments'
            title = fmt.format(self.channel_description, start, end,
                               self.dq_flag, self.run)
        fig.legend(handles=[mean, mins, maxs], labels=["Means +/- Std. Dev.",
                                                       "Absolute Minima",
                                                       "Absolute Maxima"])
        fig.gca().set_title(title)
        return fig

def main():
    if len(sys.argv) == 1:
        plt_job = PlottingJob.load()
    else:
        plt_job = PlottingJob.load(sys.argv[1])
    pj.pj.make_combined_plots()
    pj.pj.make_individual_plots()

if __name__ == "__main__":
    main()
