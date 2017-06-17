#!/usr/bin/env python
# (c) Stefan Countryman 2017

import matplotlib
if __name__ == '__main__':
    # necessary for headless plotting
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
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
DEFAULT_PLOT_PROPERTIES = {}
DEFAULT_HEIGHT = 7.5
DEFAULT_WIDTH = 10.0
# value that LSC uses to indicate a missing/unrecorded value in EPICS
UNRECORDED_VALUE_CONSTANT = 0.0
# should we subtract mean values from plots where this is relevant?
DEFAULT_SUBTRACT_MEANS = True
# make matplotlib legend fonts smaller so that they take up less space
DEFAULT_LEGEND_FONT = matplotlib.font_manager.FontProperties()
DEFAULT_LEGEND_FONT.set_size('small')
DEFAULT_AXES_POSITION = [0.125, 0.1, 0.775, 0.73]
SEC_PER_DAY = 86400.
NS_PER_SECOND = 10**9
COMBINED_TRENDS = [
    ".mean,m-trend",
    ".min,m-trend",
    ".max,m-trend",
    ".rms,m-trend",
    ".n,m-trend"
]

def get_unrecorded_value_indices(array):
    """Get indices in some array with a unrecorded value according to DAQ
    (determined by checking if the value is equal to the default for unrecorded
    data). Note that these indices correspond to data which, according to the
    DAQ, was not taken, e.g. due to the device in question being inactive."""
    return np.nonzero(array == UNRECORDED_VALUE_CONSTANT)[0]

def get_unlocatable_value_indices(array):
    """Get indices in some array which could not be located on a server or
    frame file. Note that these indices correspond to data which may have been
    taken but which are not included in the locally cached timeseries; plots
    will have to be remade once missing data has been filled in."""
    return np.nonzero(array == geco_gwpy_dump.DEFAULT_PAD)[0]

class PlottingJob(object):
    """A description of the plots that need to be made along with methods
    for making them. Has a similar interface to ``geco_gwpy_dump.Job``, and
    in fact contains a ``Job`` as one of its properties. Uses a superset
    of the data contained in a ``Job`` job specification JSON file. The
    ``geco_gwpy_dump.Job`` specifies what data is necessary to generate
    the plots, while instances of this class provides specific information on
    how to make those plots."""
    def __init__(self, job, run, channel_descriptions, plots,
                 height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
                 subtract_means=DEFAULT_SUBTRACT_MEANS):
        self.job = job
        self.run = run
        self.channel_descriptions = channel_descriptions
        self.plots = plots
        self.height = height
        self.width = width
        self.subtract_means = subtract_means
    def to_dict(self):
        """Return a dict representation of this object."""
        job_dict = self.job.to_dict()
        # we modify the basic job_dict by throwing in an extra dictionary
        # with plotting-specific information
        plot_dict = {'run': self.run,
                     'channel_descriptions': self.channel_descriptions,
                     'plots': self.plots,
                     'height': self.height,
                     'width': self.width,
                     'subtract_means': subtract_means}
        job_dict['slow_channel_plots'] = plot_dict
        return job_dict
    def save(self, jobspecfile):
        """Save a JSON representation of this object."""
        with open(jobspecfile, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    @classmethod
    def from_dict(cls, d):
        """Instantiate a PlottingJob from a suitable dictionary
        representation."""
        # set defaults for height, width, and if mean value should be removed
        plot_dict = {'height': DEFAULT_HEIGHT,
                     'width': DEFAULT_WIDTH,
                     'subtract_means': DEFAULT_SUBTRACT_MEANS}
        # update with other values and user-specified height and width and
        # whether mean values should be subtracted from relevant plots
        plot_dict.update(d['slow_channel_plots'])
        return cls(job = geco_gwpy_dump.Job.from_dict(d),
                   run = plot_dict['run'],
                   channel_descriptions = plot_dict['channel_descriptions'],
                   plots = plot_dict['plots'],
                   height = plot_dict['height'],
                   width = plot_dict['width'])
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
        for plot in self.plots:
            dq_flag = plot['dq_flag']
            channel = plot['channel']
            plot_props = DEFAULT_PLOT_PROPERTIES
            if plot.has_key('plot_properties'):
                plot_props.update(plot_properties)
            for trend in self.trends:
                desc = self.channel_descriptions[channel]
                # only need the first extension since any should have saved data
                p = IndividualPlotter(start = self.start, end = self.end,
                                      channel = channel, dq_flag = dq_flag,
                                      trend = trend, ext = self.exts[0],
                                      run = self.run,
                                      channel_description = desc,
                                      plot_properties = plot_props)
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
        for plot in self.plots:
            dq_flag = plot['dq_flag']
            channel = plot['channel']
            plot_props = DEFAULT_PLOT_PROPERTIES
            if plot.has_key('plot_properties'):
                plot_props.update(plot_properties)
            desc = self.channel_descriptions[channel]
            # only need the first extension since any should have saved data
            p = CombinedPlotter(start = self.start, end = self.end,
                                channel = channel, dq_flag = dq_flag,
                                ext = self.exts[0], run = self.run,
                                channel_description = desc,
                                plot_properties = plot_props)
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
        fig = self.get_plot(save_sidecar=True)
        fig.savefig(self.fname)
        plt.close(fig)
    def read(self):
        """Read a dict of lists of timeseries for this channel corresponding to
        the channel/trend combinations loaded time intervals when this
        Plotter's ``DataQualityFlag`` was active. The dictionary key is just
        the full channel/trend combination."""
        ts = {}
        for q in self.queries:
            ts[q.channel] = q.read_and_split_into_segments(self.dq_segments)
        return ts
    @property
    def days(self):
        """Get the length of this run in days."""
        return (self.end - self.start) / SEC_PER_DAY
    @property
    def time_ticks(self):
        """Get time ticks for this plot. Should always be between 5 and 10
        tickmarcks."""
        logdays = np.log10(self.days)
        logdaysflr = int(np.floor(logdays))
        # pick number of days per tick so that we have 5 - 10 ticks
        if logdays - logdaysflr > 0.65:
            days_per_tick = int(10**logdaysflr)
        elif logdays - logdaysflr > 0.3:
            days_per_tick = int(10**logdaysflr // 2)
        else:
            days_per_tick = int(10**logdaysflr // 5)
        return [int(t) for t in np.arange(0, self.days, days_per_tick)]
    @property
    def sanitized_dq_flag(self):
        """get the DQ Flag name as used in filenames, i.e. with commas (,) and
        colons (:) replaced in order to fit filename conventions."""
        return geco_gwpy_dump.sanitize_for_filename(self.dq_flag)
    @abc.abstractproperty
    def title(self):
        """Get the title for this plot."""
    @abc.abstractproperty
    def fname(self):
        """Return the filename for this plot"""
    @property
    def fname_sidecar(self):
        """Return the filename for this plot's metadata sidecar file"""
        return self.fname + '.sidecar.json'
    @abc.abstractmethod
    def get_plot(self, save_sidecar=False, fig=None):
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
                 run=None, channel_description=None,
                 height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
                 subtract_means=DEFAULT_SUBTRACT_MEANS,
                 plot_properties=DEFAULT_PLOT_PROPERTIES):
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
        self.height = height
        self.width = width
        self.subtract_means = subtract_means
    @property
    def title(self):
        """Get the title for this plot."""
        start = gwpy.time.from_gps(self.start)
        end = gwpy.time.from_gps(self.end)
        # are these the means? mins? maxs? only works for m-trend or s-trend
        trend_stat = self.trends[0].split('.')[1].split(',')[0]
        if self.run is None:
            fmt = '{} from {} to {}\nduring {} Segments (trend: {})'
            return fmt.format(self.channel_description, start, end,
                              self.dq_flag, self.trends[0])
        else:
            fmt = 'Diagnostic {} ({}) during {}'
            return fmt.format(self.channel_description, trend_stat, self.run)
    @property
    def fname(self):
        """Get the filename for this plot."""
        return '{}.{}.{}'.format(self.queries[0].fname, self.sanitized_dq_flag,
                                 DEFAULT_PLOT_FILETYPE)
    def get_plot(self, save_sidecar=False, fig=None):
        """Generate a ``matplotlib.figure.Figure`` for the channel and
        dq_flag specified in this ``Plotter``. Optionally pass an existing
        figure as an argument to plot to that figure's axes."""
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        # get the statistics for this one channel (the only one we will plot)
        s = self.stats[self.queries[0].channel]
        # should we subtract the mean value of the timeseries from each plot?
        if self.subtract_means:
            offset = s.means.mean()
            fmt = ("Difference between {} and Distribution\nSystem Time [ns], "
                   "Mean Value Removed ({:.2f} ns)")
            y_label = fmt.format(self.channel_description,
                                 offset*NS_PER_SECOND)
        else:
            offset = 0
            y_label = "Delay vs. Timing Distribution System [ns]"
        # label the y-axis
        ax.set_ylabel(y_label)
        # use number of days since start of run for t-axis
        t_axis = (s.times - self.start) / SEC_PER_DAY
        # label the t-axis
        t0 = gwpy.time.tconvert(self.start).strftime("%c")
        ax.set_xlabel("Days Since Start of Run ({} UTC)".format(t0))
        ax.set_xlim(left=0, right=t_axis.max())
        ax.set_xticks(self.time_ticks)
        ax.set_xticklabels([str(l) for l in self.time_ticks])
        # mark and remove unrecorded data
        unrecorded_indices = dict()
        t_axis_clean = dict()
        y_axis_clean = dict()
        for array_name in ['means', 'mins', 'maxs']:
            array = s.__getattribute__(array_name)
            unrecorded = get_unrecorded_value_indices(array)
            y_axis_clean[array_name] = np.delete(array, unrecorded)
            t_axis_clean[array_name] = np.delete(t_axis, unrecorded)
            unrecorded_indices[array_name] = list(unrecorded)
        # save unrecorded data indices to a sidecar file
        if save_sidecar:
            with open(self.fname_sidecar, 'w') as f:
                json.dump({'unrecorded_indices': unrecorded_indices},
                          f, indent=2)
        # plot everything; scale up by 10^9 since plots are in ns, not seconds
        stds_unrecorded_removed = np.delete(s.stds, unrecorded_indices['means'])
        ax.errorbar(t_axis_clean['means'],
                    (y_axis_clean['means'] - offset)*NS_PER_SECOND,
                    marker="o", color="green",
                    yerr=stds_unrecorded_removed*NS_PER_SECOND,
                    label="Means +/- Std. Dev.")
        ax.scatter(t_axis_clean['mins'],
                   (y_axis_clean['mins'] - offset)*NS_PER_SECOND,
                   marker="^", color="blue", label="Minima")
        ax.scatter(t_axis_clean['maxs'],
                   (y_axis_clean['maxs'] - offset)*NS_PER_SECOND,
                   marker="v", color="red", label="Maxima")
        # plot the unrecorded points as well
        all_unrecorded = list(set.union(*[set(v) for v in
                                       unrecorded_indices.values()]))
        unrecorded_times = t_axis[all_unrecorded]
        ax.scatter(unrecorded_times, unrecorded_times*0, marker="x",
                   color="orange", label="Data Not Taken",
                   s=(fig.dpi**2)*(0.16**2), zorder=0)
        # set plot size
        fig.set_size_inches((self.width, self.height))
        ax.set_title(self.title, y=1.07)
        ax.set_position(DEFAULT_AXES_POSITION)
        plt.figure(fig.number)
        plt.legend(prop=DEFAULT_LEGEND_FONT, ncol=4,
                   loc='upper center', bbox_to_anchor=(0.5, 1.07))
        return fig

class CombinedPlotter(Plotter): #TODO
    """A class for plotting a slow channel with all trends. Used when all
    trends are available to generate a combined plot for a channel over some
    period of time."""
    def __init__(self, start, end, channel, dq_flag,
                 ext=geco_gwpy_dump.DEFAULT_EXTENSION,
                 run=None, channel_description=None,
                 height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
                 subtract_means=DEFAULT_SUBTRACT_MEANS,
                 plot_properties=DEFAULT_PLOT_PROPERTIES):
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
        self.height = height
        self.width = width
        self.subtract_means = subtract_means
    @property
    def title(self):
        """Get the title for this plot."""
        start = gwpy.time.from_gps(self.start)
        end = gwpy.time.from_gps(self.end)
        if self.run is None:
            fmt = '{} from {} to {}\nduring {} Segments'
            return fmt.format(self.channel_description, start, end,
                              self.dq_flag)
        else:
            fmt = 'Diagnostic {} during {}'
            return fmt.format(self.channel_description, self.run)
    @property
    def fname(self):
        """Return the filename for the saved plot image."""
        ch = self.queries[0].sanitized_channel
        return '{}__{}__{}.{}.combined.{}'.format(self.start, self.end, ch,
                                                  self.sanitized_dq_flag,
                                                  DEFAULT_PLOT_FILETYPE)
    def get_plot(self, save_sidecar=False, fig=None):
        """Generate a ``matplotlib.figure.Figure`` for the channel and
        dq_flag specified in this ``CombinedPlotter``. Optionally pass an
        existing figure as an argument to plot to that figure's axes."""
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        # calculate statistics for each channel and dq_flag active segment
        s = self.stats
        absmaxs = s[self.channel + '.max,m-trend'].maxs
        absmins = s[self.channel + '.min,m-trend'].mins
        means   = s[self.channel + '.mean,m-trend'].means
        times   = s[self.channel + '.mean,m-trend'].times
        stds    = s[self.channel + '.mean,m-trend'].stds
        # should we subtract the mean value of the timeseries from each plot?
        if self.subtract_means:
            offset = means.mean()
            fmt = ("Difference between {} and Distribution\nSystem Time [ns], "
                   "Mean Value Removed ({:.2f} ns)")
            y_label = fmt.format(self.channel_description,
                                 offset*NS_PER_SECOND)
        else:
            offset = 0
            y_label = "Delay vs. Timing Distribution System [ns]"
        # label the y-axis
        ax.set_ylabel(y_label)
        # use number of days since start of run for t-axis
        t_axis = (times - self.start) / SEC_PER_DAY
        # label the t-axis
        t0 = gwpy.time.tconvert(self.start).strftime("%c")
        ax.set_xlabel("Days Since Start of Run ({} UTC)".format(t0))
        ax.set_xlim(left=0, right=t_axis.max())
        ax.set_xticks(self.time_ticks)
        ax.set_xticklabels([str(l) for l in self.time_ticks])
        # mark and remove unrecorded data
        unrecorded_indices = dict()
        t_axis_clean = dict()
        y_axis_clean = dict()
        for array_name in ['means', 'absmins', 'absmaxs']:
            array = locals()[array_name]
            unrecorded = get_unrecorded_value_indices(array)
            y_axis_clean[array_name] = np.delete(array, unrecorded)
            t_axis_clean[array_name] = np.delete(t_axis, unrecorded)
            unrecorded_indices[array_name] = list(unrecorded)
        # save unrecorded data indices to a sidecar file
        if save_sidecar:
            with open(self.fname_sidecar, 'w') as f:
                json.dump({'unrecorded_indices': unrecorded_indices},
                          f, indent=2)
        # plot everything; scale up by 10^9 since plots are in ns, not seconds
        stds_unrecorded_removed = np.delete(stds, unrecorded_indices['means'])
        ax.errorbar(t_axis_clean['means'],
                    (y_axis_clean['means'] - offset)*NS_PER_SECOND,
                    marker="o", color="green",
                    yerr=stds_unrecorded_removed*NS_PER_SECOND,
                    label="Means +/- Std. Dev.")
        ax.scatter(t_axis_clean['absmins'],
                   (y_axis_clean['absmins'] - offset)*NS_PER_SECOND,
                   marker="^", color="blue", label="Abs. Minima")
        ax.scatter(t_axis_clean['absmaxs'],
                   (y_axis_clean['absmaxs'] - offset)*NS_PER_SECOND,
                   marker="v", color="red", label="Abs. Maxima")
        # plot the unrecorded points as well
        all_unrecorded = list(set.union(*[set(v) for v in
                                       unrecorded_indices.values()]))
        unrecorded_times = t_axis[all_unrecorded]
        ax.scatter(unrecorded_times, unrecorded_times*0, marker="x",
                   color="orange", label="Data Not Taken",
                   s=(fig.dpi**2)*(0.16**2), zorder=0)
        # set plot size
        fig.set_size_inches((self.width, self.height))
        ax.set_title(self.title, y=1.07)
        ax.set_position(DEFAULT_AXES_POSITION)
        plt.figure(fig.number)
        plt.legend(prop=DEFAULT_LEGEND_FONT, ncol=4,
                   loc='upper center', bbox_to_anchor=(0.5, 1.07))
        return fig

def main():
    if len(sys.argv) == 1:
        plt_job = PlottingJob.load()
    else:
        plt_job = PlottingJob.load(sys.argv[1])
    plt_job.make_combined_plots()
    plt_job.make_individual_plots()

if __name__ == "__main__":
    main()
