#!/usr/bin/env python
# (c) Stefan Countryman 2017

import matplotlib
if __name__ == '__main__':
    # necessary for headless plotting
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.patches
import multiprocessing
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
DEFAULT_PLOT_PROPERTIES = {
    "save_sidecars": True,
    "omitted_indices": list(),
    "omitted_label": "Omitted Values",
    "missing_label": "Data Not Found",
    "unrecorded_label": "Data Not Taken",
    "omitted_color": "#41e5f4",             # teal, sorta
    "missing_color": "purple",
    "unrecorded_color": "orange",
    "handle_omitted_values": "hide",
    "handle_missing_values": "mark",
    "handle_unrecorded_values": "mark",
    "subtract_means": True,
    "detector_offline": [],
    "find_unrecorded": True,
    "fname_desc": None
}
PLOT_PROPERTY_DESCRIPTIONS = {
    "save_sidecars": ("If true, save a sidecar file with missing and "
                     "unrecorded data saved in it for easier investigation."),
    "omitted_indices": ("Optionally, provide a list of indices that should be "
                        "removed from the timeseries before plotting. This "
                        "data will not be displayed in plots, but "
                        "bad_index_types will still find missing and "
                        "unrecorded values in the original timeseries."),
    "omitted_legend": ("If ``handle_omitted_values`` is set to 'mark`, we "
                       "will mark omitted values on the final plot and will "
                       "label them in the legend using this string."),
    "handle_omitted_values": ("``mark`` will clearly mark omitted indices "
                              "in the final plot but will NOT show "
                              "the value at that index for ANY timeseries "
                              "at that index. ``hide`` will remove those "
                              "indices entirely, including from "
                              "from the plot (DEFAULT). ``ignore`` "
                              "will keep those indices in the plot with "
                              "their placeholder value intact (useful if "
                              "that placeholder value is suspected to be "
                              "the true value at that point)."),
    "handle_missing_values": ("``mark`` will clearly mark missing "
                              "values in the final plot and will show "
                              "the value at that index for any timeseries "
                              "that does not contain missing data "
                              "(DEFAULT). ``hide`` will remove those "
                              "indices entirely from the plot. ``ignore`` "
                              "will keep those indices in the plot with "
                              "their placeholder value intact (useful if "
                              "that placeholder value is suspected to be "
                              "the true value at that point)."),
    "handle_unrecorded_values": ("``mark`` will clearly mark unrecorded "
                                 "values in the final plot and will show "
                                 "the value at that index for any timeseries "
                                 "that does not contain missing data "
                                 "(DEFAULT). ``hide`` will remove those "
                                 "indices entirely from the plot. ``ignore`` "
                                 "will keep those indices in the plot with "
                                 "their placeholder value intact (useful if "
                                 "that placeholder value is suspected to be "
                                 "the true value at that point)."),
    "subtract_means": ("If true, subtract means from plots. If a number, "
                       "subtract that number from the plots."),
    "detector_offline": "List of GPS start/stop tuples when detector was off",
    "find_unrecorded": "Should times when data was not taken be found/plotted?",
    "fname_desc": ("A description of this particular plot, to be appended to "
                   "the plot filename. Useful for cases when multiple "
                   "plots with different options are to be made for the same "
                   "time span, channel, and DQ flag."),
    "ylim_top": "Upper y-limit for the axes of plots for this channel",
    "ylim_bottom": "Lower y-limit for the axes of plots for this channel",
    "xlim_left": "Leftward x-limit for the axes of plots for this channel",
    "xlim_right": "Rightward x-limit for the axes of plots for this channel"
}
DEFAULT_HEIGHT = 7.5
DEFAULT_WIDTH = 10.0
NUM_THREADS = 6
# value that LSC uses to indicate a unrecorded value in EPICS
UNRECORDED_VALUE_CONSTANT = 0.0
# value that we have GWPY use to indicate a missing value in NDS2
MISSING_VALUE_CONSTANT = -1.0
# should we subtract mean values from plots where this is relevant?
DEFAULT_SUBTRACT_MEANS = True
# make matplotlib legend fonts smaller so that they take up less space
DEFAULT_LEGEND_FONT = matplotlib.font_manager.FontProperties()
DEFAULT_LEGEND_FONT.set_size('small')
DEFAULT_AXES_POSITION = [0.125, 0.1, 0.775, 0.73]
DEFAULT_TITLE_OFFSET = 1.07
SEC_PER_DAY = 86400.
NS_PER_SECOND = 10**9
COMBINED_TRENDS = [
    ".mean,m-trend",
    ".min,m-trend",
    ".max,m-trend",
    ".rms,m-trend",
    ".n,m-trend"
]

class Cacheable(object):
    """An object that can store the output of property getter functions in a
    "private" cache to avoid regeneration times. The cache can be flushed if
    data needs to be regenerated from sources."""
    @staticmethod
    def _cacheable(property_func):
        """A decorator that makes an object property cacheable, i.e. the
        function generating the data will not be called if the data has
        already been generated. This can be cancelled by flushing cache."""
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_cache'):
                setattr(self, '_cache', dict())
            prop_key = str(hash(property_func))
            if not self._cache.has_key(prop_key):
                self._cache[prop_key] = property_func(self, *args, **kwargs)
            return self._cache[prop_key]
        return wrapper
    def _clear_cache(obj):
        """Delete the _cache property used by the @_cacheable decorator (if
        present)."""
        if hasattr(obj, '_cache'):
            delattr('_cache')

def get_unrecorded_indices(array):
    """Get indices in some array with a unrecorded value according to DAQ
    (determined by checking if the value is equal to the default for unrecorded
    data). Note that these indices correspond to data which, according to the
    DAQ, was not taken, e.g. due to the device in question being inactive."""
    return np.nonzero(array == UNRECORDED_VALUE_CONSTANT)[0]

def get_missing_indices(array):
    """Get indices in some array where the value cannot be found in any file
    according to NDS2 (determined by checking if the value is equal to the
    pad value for unrecorded data). Note that these indices correspond to data
    which, according to NDS2, cannot be found saved anywhere, though it may
    have been saved somewhere."""
    return np.nonzero(array == MISSING_VALUE_CONSTANT)[0]

def multiprocessing_traceback(func):
    """A decorator for formatting exception traceback into a string to aid in
    debugging when using the ``multiprocessing`` module."""
    import traceback, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            raise type(e)(msg)
    return wrapper

class PlottingJob(object):
    """A description of the plots that need to be made along with methods
    for making them. Has a similar interface to ``geco_gwpy_dump.Job``, and
    in fact contains a ``Job`` as one of its properties. Uses a superset
    of the data contained in a ``Job`` job specification JSON file. The
    ``geco_gwpy_dump.Job`` specifies what data is necessary to generate
    the plots, while instances of this class provides specific information on
    how to make those plots."""
    def __init__(self, job, run, channel_descriptions, plots,
                 height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH):
        self.job = job
        self.run = run
        self.channel_descriptions = channel_descriptions
        self.plots = plots
        self.height = height
        self.width = width
    def to_dict(self):
        """Return a dict representation of this object."""
        job_dict = self.job.to_dict()
        # we modify the basic job_dict by throwing in an extra dictionary
        # with plotting-specific information
        plot_dict = {'run': self.run,
                     'channel_descriptions': self.channel_descriptions,
                     'plots': self.plots,
                     'height': self.height,
                     'width': self.width}
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
        plot_dict = {'height': DEFAULT_HEIGHT, 'width': DEFAULT_WIDTH}
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
            plot_props = DEFAULT_PLOT_PROPERTIES.copy()
            if plot.has_key('plot_properties'):
                plot_props.update(plot['plot_properties'])
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
        mapf = multiprocessing.Pool(processes=NUM_THREADS).map
        mapf(_save_plot, self.individual_plotters)
    @property
    def combined_plotters(self):
        """Return a list of ``CombinedPlotter``s for this job."""
        plotters = []
        for plot in self.plots:
            dq_flag = plot['dq_flag']
            channel = plot['channel']
            plot_props = DEFAULT_PLOT_PROPERTIES.copy()
            if plot.has_key('plot_properties'):
                plot_props.update(plot['plot_properties'])
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
        mapf = multiprocessing.Pool(processes=NUM_THREADS).map
        mapf(_save_plot, self.combined_plotters)

class Plotter(Cacheable):
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
    # define a named tuple for statistics on saved values
    Stats = collections.namedtuple('Stats', ['means', 'mins', 'maxs',
                                             'stds', 'times', 'ns'])
    # a way of storing indices of types of bad data.
    BadIndices = collections.namedtuple('BadIndices',
                                        ['unrecorded', 'missing', 'omitted'])
    # a way of grouping t and y axis arrays.
    AxisArrays = collections.namedtuple('AxisArrays', ['y_axis', 't_axis'])
    @abc.abstractproperty
    def PlotVars(self):
        """Define a named tuple class for the data arrays that will actually be
        plotted. Really just a naming convention; the properties of this class
        can be timeseries, indices for a timeseries, etc. PlotVars must include
        a ``times`` attribute or else ``t_axis`` and perhaps other methods
        for a given plotter will not work."""
    @property
    def bad_index_types(self):
        """Get the indices of missing, unrecorded, and omitted times. Indices
        refer to the original timeseries before any data has been deleted.
        Indices are named on a per-variable basis for the variables used in
        each plot. For example, access the missing values in the mean
        timerseries using ``plotter.bad_index_types.missing.means``"""
        bad = {'unrecorded': dict(), 'missing': dict(), 'omitted': dict()}
        plot_vars = self.plot_vars
        # get the bad indices of each type
        for array_name in self.PlotVars._fields:
            array = getattr(plot_vars, array_name)
            bad['unrecorded'][array_name] = list(get_unrecorded_indices(array))
            bad['missing'][array_name] = list(get_missing_indices(array))
            bad['omitted'][array_name] = self.plot_properties['omitted_indices']
        # save this data to a sidecar file for separate inspection
        if self.plot_properties['save_sidecars']:
            with open(self.fname_sidecar, 'w') as f:
                json.dump(bad, f, indent=2)
        # put the bad indices in PlotVars namedtuples inside a BadIndices
        # namedtuple
        return self.BadIndices(unrecorded=self.PlotVars(**bad['unrecorded']),
                          missing=self.PlotVars(**bad['missing']),
                          omitted=self.PlotVars(**bad['omitted']))
    @property
    def bad_indices(self):
        """Separate out missing and unrecorded values from cleaned timeseries
        in a named tuple and give the indices of those bad values. The indices
        that are returned depend on which types of data are supposed to be
        removed as specified in the plot_properties. Uses the PlotVars
        ``namedtuple`` along with AxisArrays to provide an interface to the
        plot variables in the form of e.g. ``plotter.bad_indices.means``."""
        # if there are any indices that should be removed from all plot
        # variables, put them in this set:
        shared_bad_set = set()
        bad_sets = {}
        for array_name in self.PlotVars._fields:
            all_bad_inds = set()
            for badness_type in self.BadIndices._fields:
                handle_method_key = 'handle_{}_values'.format(badness_type)
                ind_type = getattr(self.bad_index_types, badness_type)
                bad_inds = getattr(ind_type, array_name)
                if self.plot_properties[handle_method_key] == "hide":
                    shared_bad_set = shared_bad_set.union(bad_inds)
                elif self.plot_properties[handle_method_key] == "mark":
                    if array_name == 'omitted':
                        shared_bad_set = shared_bad_set.union(bad_inds)
                    else:
                        all_bad_inds = all_bad_inds.union(bad_inds)
            bad_sets[array_name] = all_bad_inds
        # now that all plot variables and their corresponding types of missing
        # data have contributed to the shared set of bad indices, we can fold
        # the shared bad indices in to each list of plot-value-specific bad
        # indices.
        for array_name in self.PlotVars._fields:
            bad_sets[array_name] = bad_sets[array_name].union(shared_bad_set)
            # make sure we are returning numpy arrays for our index lists
            bad_sets[array_name] = np.sort(list(bad_sets[array_name]))
        # put everything into a plot variable namedtuple
        return self.PlotVars(**bad_sets)
    @property
    @Cacheable._cacheable
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
        - number of sample values per segment (ns)
        
        Access like e.g. plotter.stats[channel_name].means"""
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
    @abc.abstractproperty
    def plot_vars(self):
        """Get the actual data arrays that will go into this plot."""
    @property
    def dq_segments(self):
        """Get the ``gwpy.segments.DataQualityFlag`` time segments
        corresponding to this ``Plotter``."""
        return self.job.get_dq_segments()[self.dq_flag]
    # must be a static method so that we can use multiprocessing on it
    @staticmethod
    def _save_plot(plotter):
        """Save this plot as an image file."""
        fig = plotter.get_plot()
        fig.savefig(plotter.fname)
        plt.close(fig)
    def save_plot(self):
        """Save this plot as an image file."""
        self._save_plot(self)
    def read(self):
        """Read a dict of lists of timeseries for this channel corresponding to
        the channel/trend combinations loaded time intervals when this
        Plotter's ``DataQualityFlag`` was active. The dictionary key is just
        the full channel/trend combination."""
        ts = {}
        for q in self.queries:
            ts[q.channel] = q.read_and_split_into_segments(self.dq_segments)
        return ts
    def size_and_label(self, ax):
        """Resize and label axes for a given matplotlib Axes instance according
        to the plot_properties and class methods of this Plotter instance."""
        # label axes and set scales and limits of the plot
        ax.set_title(self.title, y=DEFAULT_TITLE_OFFSET)
        ax.set_ylabel(self.y_label)
        ax.set_xlabel(self.t_label)
        ax.set_xticks(self.time_ticks)
        ax.set_xticklabels([str(l) for l in self.time_ticks])
        ax.set_xlim(self.t_lim)
        ax.set_position(DEFAULT_AXES_POSITION)
        # add a legend
        ax.legend(prop=DEFAULT_LEGEND_FONT, ncol=4,
                  loc='upper center',
                  bbox_to_anchor=(0.5, DEFAULT_TITLE_OFFSET))
        # these don't have defaults, so only update if specified
        if self.plot_properties.has_key('ylim_bottom'):
            ax.set_ylim(bottom=self.plot_properties['ylim_bottom'])
        if self.plot_properties.has_key('ylim_top'):
            ax.set_ylim(top=self.plot_properties['ylim_top'])
    def plot_bad_value_markers(self, ax):
        """If there are bad values (missing, unrecorded, or purposefully
        omitted), plot any that are supposed to be marked on the given
        matplotlib Axes instance."""
        for badness_type in self.BadIndices._fields:
            handle_method_key = 'handle_{}_values'.format(badness_type)
            color_key = '{}_color'.format(badness_type)
            label_key = '{}_label'.format(badness_type)
            if self.plot_properties[handle_method_key] == "mark":
                bad_inds = getattr(self.bad_index_types, badness_type)
                # get all bad indices (mins, maxs, etc.) for this badness type
                all_bad_inds = list(set.union(*[set(b) for b in bad_inds]))
                if len(all_bad_inds) != 0:
                    # we will want to set the original y limits after adding
                    # these markers, so store them now
                    ylim = ax.get_ylim()
                    color = self.plot_properties[color_key]
                    label = self.plot_properties[label_key]
                    ax.errorbar(self.t_axis[all_bad_inds],
                                len(all_bad_inds)*[0], marker="x",
                                color=color, label=label,
                                zorder=0, linestyle='none',
                                yerr=4*max([abs(l) for l in ylim]))
                    ax.set_ylim(ylim)
    @property
    def y_label(self):
        """Get a label for the y-axis based on trend type."""
        if self.plot_properties['subtract_means'] is True:
            fmt = ("Difference between {} and Distribution\nSystem Time [ns], "
                   "Mean Value Removed ({:.2f} ns)")
            y_label = fmt.format(self.channel_description,
                                 self.trend*NS_PER_SECOND)
        elif self.plot_properties['subtract_means'] is False:
            fmt = "Difference between {} and Distribution\nSystem Time [ns]"
            y_label = fmt.format(self.channel_description)
        else:
            fmt = ("Difference between {} and Distribution\nSystem Time [ns], "
                   "Offset Removed ({:.2f} ns)")
            y_label = fmt.format(self.channel_description,
                                 self.trend*NS_PER_SECOND)
        return y_label
    @property
    def t_axis(self):
        """Return the default t-axis, which is scaled to be in days from the
        start of the observation run. ``Matplotlib`` refers to this as the
        x-axis (the horizontal one)."""
        return (self.plot_vars.times - self.start) / SEC_PER_DAY
    @property
    def t_lim(self):
        """Return a tuple containing the left and right t-limits for this
        plot."""
        if self.plot_properties.has_key('xlim_left'):
            left = self.plot_properties['xlim_left']
        else:
            left = 0
        if self.plot_properties.has_key('xlim_right'):
            right = self.plot_properties['xlim_right']
        else:
            right = self.t_axis.max()
        return (left, right)
    @property
    def t_label(self):
        """Return the default t-axis label for a t-axis measuring days since
        the start of an observing run. ``Matplotlib`` refers to this as the
        x-axis (the horizontal one)."""
        t0 = gwpy.time.tconvert(self.start).strftime("%c")
        return "Days Since Start of Run ({} UTC)".format(t0)
    @property
    def time_ticks(self):
        """Get time ticks for this plot. Should always be between 5 and 10
        tickmarcks."""
        start_day, end_day = self.t_lim
        logdays = np.log10(end_day - start_day)
        logdaysflr = int(np.floor(logdays))
        # pick number of days per tick so that we have 5 - 10 ticks
        if logdays - logdaysflr > 0.65:
            days_per_tick = int(10**logdaysflr)
        elif logdays - logdaysflr > 0.3:
            days_per_tick = int(10**logdaysflr // 2)
        else:
            days_per_tick = int(10**logdaysflr // 5)
        return [int(t) for t in np.arange(np.ceil(start_day), end_day,
                                          days_per_tick)]
    @property
    def sanitized_dq_flag(self):
        """get the DQ Flag name as used in filenames, i.e. with commas (,) and
        colons (:) replaced in order to fit filename conventions."""
        return geco_gwpy_dump.sanitize_for_filename(self.dq_flag)
    @property
    def trend(self):
        """Should we subtract the mean value of the timeseries from each plot?
        if subtract means is defined as a number, then we will subtract that
        value out instead of the mean."""
        if self.plot_properties['subtract_means'] is True:
            trend = self.plot_vars.means.mean()
        elif self.plot_properties['subtract_means'] is False:
            trend = 0
        else:
            trend = self.plot_properties['subtract_means']
        return trend
    @abc.abstractproperty
    def title(self):
        """Get the title for this plot."""
    @abc.abstractproperty
    def fname(self):
        """Get the filename for this plot. If a filename description is
        provided as the fname_desc key in the plot_properties for this plot in
        the jobspec file, this description should be appended to the title."""
    @property
    def fname_sidecar(self):
        """Return the filename for this plot's metadata sidecar file"""
        return self.fname + '.sidecar.json'
    @abc.abstractmethod
    def plot_timeseries(self, ax):
        """Take a Matplotlib Axes object and plot the timeries associated with
        this Plotter subclass to those axes. This method is used by ``get_plot``
        to specify what curves and styles should actually be plotted."""
    def get_plot(self, fig=None):
        """Generate a ``matplotlib.figure.Figure`` for the channel and
        dq_flag specified in this ``Plotter``. Optionally pass an existing
        figure as an argument to plot to that figure's axes."""
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()
        # plot the actual curves that make this Plotter subclass unique."""
        self.plot_timeseries(ax)
        # mark any bad values as specified in plot_properties
        self.plot_bad_value_markers(ax)
        # label, scale, and resize the plot
        fig.set_size_inches((self.width, self.height))
        self.size_and_label(ax)
        return fig

@multiprocessing_traceback
def _save_plot(plotter):
    """Must define this at the global level to allow for multiprocessing."""
    type(plotter)._save_plot(plotter)

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
        self.plot_properties = plot_properties
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
        """Get the filename for this plot. If a filename description is
        provided as the fname_desc key in the plot_properties for this plot in
        the jobspec file, this description should be appended to the title."""
        if self.plot_properties['fname_desc'] is None:
            desc = ""
        else:
            desc = "." + self.plot_properties['fname_desc']
        return '{}.{}{}.{}'.format(self.queries[0].fname, self.sanitized_dq_flag,
                                   desc, DEFAULT_PLOT_FILETYPE)
    @property
    def PlotVars(self):
        return self.Stats
    @property
    def plot_vars(self):
        return self.stats[self.queries[0].channel]
    def plot_timeseries(self, ax):
        """Scale up by 10^9 since plots are in ns, not seconds.
        Remove any indices considered bad in ``plot_properties``"""
        ax.errorbar(np.delete(self.t_axis, self.bad_indices.means),
                    (  np.delete(self.plot_vars.means, self.bad_indices.means)
                     - self.trend  ) * NS_PER_SECOND,
                    marker="o", color="green",
                    linestyle='none',
                    yerr=np.delete(self.plot_vars.stds,
                                   self.bad_indices.means) * NS_PER_SECOND,
                    label="Means +/- Std. Dev.")
        ax.scatter(np.delete(self.t_axis, self.bad_indices.mins),
                   (  np.delete(self.plot_vars.mins, self.bad_indices.mins)
                    - self.trend) * NS_PER_SECOND,
                   marker="^", color="blue", label="Minima")
        ax.scatter(np.delete(self.t_axis, self.bad_indices.maxs),
                   (  np.delete(self.plot_vars.maxs, self.bad_indices.maxs)
                    - self.trend) * NS_PER_SECOND,
                   marker="v", color="red", label="Maxima")

class CombinedPlotter(Plotter):
    """A class for plotting a slow channel with all trends. Used when all
    trends are available to generate a combined plot for a channel over some
    period of time."""
    def __init__(self, start, end, channel, dq_flag,
                 ext=geco_gwpy_dump.DEFAULT_EXTENSION,
                 run=None, channel_description=None,
                 height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
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
        self.plot_properties = plot_properties
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
        """Get the filename for this plot. If a filename description is
        provided as the fname_desc key in the plot_properties for this plot in
        the jobspec file, this description should be appended to the title."""
        ch = self.queries[0].sanitized_channel
        if self.plot_properties['fname_desc'] is None:
            desc = ""
        else:
            desc = "." + self.plot_properties['fname_desc']
        return '{}__{}__{}.{}.combined{}.{}'.format(self.start, self.end, ch,
                                                    self.sanitized_dq_flag,
                                                    desc,
                                                    DEFAULT_PLOT_FILETYPE)
    PlotVars = collections.namedtuple('CombinedPlotterVars',
                                      ['means', 'absmins', 'absmaxs',
                                       'stds', 'times'])
    @property
    def plot_vars(self):
        """Calculate statistics for each channel and dq_flag active segment
        and return the actual values that will go into this plot."""
        s = self.stats
        absmaxs = s[self.channel + '.max,m-trend'].maxs
        absmins = s[self.channel + '.min,m-trend'].mins
        means   = s[self.channel + '.mean,m-trend'].means
        times   = s[self.channel + '.mean,m-trend'].times
        stds    = s[self.channel + '.mean,m-trend'].stds
        return self.PlotVars(means=means, absmins=absmins, absmaxs=absmaxs,
                             stds=stds, times=times)
    def plot_timeseries(self, ax):
        """Scale up by 10^9 since plots are in ns, not seconds.
        Remove any indices considered bad in ``plot_properties``"""
        ax.errorbar(np.delete(self.t_axis, self.bad_indices.means),
                    (  np.delete(self.plot_vars.means, self.bad_indices.means)
                     - self.trend  ) * NS_PER_SECOND,
                    marker="o", color="green",
                    linestyle='none',
                    yerr=np.delete(self.plot_vars.stds,
                                   self.bad_indices.means) * NS_PER_SECOND,
                    label="Means +/- Std. Dev.")
        ax.scatter(np.delete(self.t_axis, self.bad_indices.absmins),
                   (  np.delete(self.plot_vars.absmins,
                                self.bad_indices.absmins)
                    - self.trend  ) * NS_PER_SECOND,
                   marker="^", color="blue", label="Abs. Minima")
        ax.scatter(np.delete(self.t_axis, self.bad_indices.absmaxs),
                   (  np.delete(self.plot_vars.absmaxs,
                                self.bad_indices.absmaxs)
                    - self.trend  ) * NS_PER_SECOND,
                   marker="v", color="red", label="Abs. Maxima")

def main():
    if len(sys.argv) == 1:
        plt_job = PlottingJob.load()
    else:
        plt_job = PlottingJob.load(sys.argv[1])
    plt_job.make_combined_plots()
    plt_job.make_individual_plots()

if __name__ == "__main__":
    main()
