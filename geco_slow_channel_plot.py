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
import scipy.stats
import geco_gwpy_dump
import gwpy.segments
import gwpy.time
import collections
import logging
import json
import sys
import abc

###############################################################################
#
# CONSTANTS
#
###############################################################################

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
    "detrend": 'mean',
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
    "detrend": ("If equals 'mean', subtract means from plots. If equals "
                "'none', do not shift the plots. If equals 'linear', remove "
                "the linear trend from the plots. If a number, "
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
# make a dictionary of conversion factors between seconds and other time units
# returned by ``Plotter.t_units``
SEC_PER = {
    "ns": 1e-9,
    "s": 1.,
    "days": 86400.
}
COMBINED_TRENDS = [
    ".mean,m-trend",
    ".min,m-trend",
    ".max,m-trend",
    ".rms,m-trend",
    ".n,m-trend"
]

###############################################################################
#
# METHODS AND DECORATORS
#
###############################################################################

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

def fetch_data_first(fetch_first):
    """A decorator that checks whether a given Plotter's job is done (i.e.
    whether data has been downloaded) and fetches any missing data if it
    is not yet finished. Methods that don't need to dump a ton of data before
    being run should have this decorator applied so that they can dynamically
    fetch data from NDS2 even if it hasn't been downloaded yet. Longer data
    dumps should avoid this decorator so as to avoid blocking. This can be
    overridden when the function is called by passing the ``fetch_data_first``
    argument to the original function."""
    def real_decorator(func):
        def wrapper(plotter, *args, **kwargs):
            # can override the default fetching behavior by passing the
            # ``fetch_data_first`` kwarg to the function in question
            if kwargs.has_key('fetch_data_first'):
                fetch_first = kwargs['fetch_data_first']
            j = plotter.job
            if not all([q.file_exists for q in j.queries]):
                if fetch_first:
                    logging.info('Fetching data for job: {}'.format(j))
                    logging.debug('Running queries for job: {}'.format(j))
                    j.run_queries()
                    logging.debug('Concatenating data for job: {}'.format(j))
                    j.concatenate_files()
                    # this will run even if we don't have an m-trend, in which
                    # case it will just harmlessly return.
                    logging.debug('Find missing m-trend for job: {}'.format(j))
                    j.fill_in_missing_m_trend()
                else:
                    msg = ('Data missing for job: {}\n'
                           'Use ``fetch_data_first=True`` to automatically '
                           'fetch missing data.').format(j)
                    raise IOError(msg)
            return func(plotter, *args, **kwargs)
        return wrapper
    return real_decorator

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

def plot_vertical_marker(ax, t, label="marked", color="pink", marker="x"):
    """Plot a vertical marker (of the sort used to mark bad indices) to the
    supplied ``matplotlib`` Axes object."""
    # we will want to reset the original y limits after adding these markers,
    # so store them now
    ylim = ax.get_ylim()
    y_offset = (ylim[1] + ylim[0]) / 2.
    y_span = (ylim[1] - ylim[0])
    ax.errorbar(t, len(t)*[y_offset], color=color, linestyle='none',
                zorder=100, marker=marker, label=label, yerr=4*y_span)
    ax.set_ylim(ylim)

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

@multiprocessing_traceback
def _save_plot(plotter):
    """Must define this at the global level to allow for multiprocessing."""
    type(plotter)._save_plot(plotter)

def trend_var(attr):
    """Returns a function that takes a plotter and channel and fetches the
    given attribute from the plotter.stats for that channel. Used for
    TrendDataPlotter plot_vars."""
    def getter(plotter, channels):
        return getattr(plotter.stats[channels[0]], attr)

def trend_chan(trend, ext=',m-trend'):
    """Returns a function that gets the channel name from a given plotter by
    specifying the trend to append to that channel. Used for TrendDataPlotter
    plot_vars."""
    def channelmethod(plotter):
        return ['{}.{}{}'.format(plotter.channel, trend, ext)]

###############################################################################
#
# ABSTRACT CLASSES AND FACTORIES
#
###############################################################################

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
        a ``times`` attribute or else ``t_axis`` and other methods
        for a given plotter will not work. It must also include a ``means``
        attribute, which, for full data plots, just corresponds to the actual
        value of the timeseries at that point."""
    @property
    def plot_vars(self):
        """Calculate statistics for each channel and dq_flag active segment
        and return the actual values that will go into this plot."""
        return self.PlotVars(self)
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
    @abc.abstractmethod
    def read(self):
        """Read in the data needed for this Plotter. The exact form of the
        returned object depends on the type of plotter."""
    def size_and_label(self, ax):
        """Resize and label axes for a given matplotlib Axes instance according
        to the plot_properties and class methods of this Plotter instance."""
        # label axes and set scales and limits of the plot
        ax.set_title(self.title, y=DEFAULT_TITLE_OFFSET)
        ax.set_ylabel(self.y_label)
        ax.set_xlabel(self.t_label)
        ax.set_xticks(self.t_ticks)
        ax.set_xticklabels([str(l) for l in self.t_ticks])
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
                    plot_vertical_marker(ax, self.t_axis[all_bad_inds],
                                         label=self.plot_properties[label_key],
                                         color=self.plot_properties[color_key])
    @property
    def y_label(self):
        """Get a label for the y-axis based on trend type."""
        if self.plot_properties['detrend'] == 'mean':
            fmt = ("Difference between {} and Distribution\nSystem Time [ns], "
                   "Mean Value Removed ({:.2f} ns)")
            y_label = fmt.format(self.channel_description,
                                 self.trend / SEC_PER['ns'])
        elif self.plot_properties['detrend'] == 'none':
            fmt = "Difference between {} and Distribution\nSystem Time [ns]"
            y_label = fmt.format(self.channel_description)
        elif self.plot_properties['detrend'] == 'linear':
            fmt = ("Difference between {} and Distribution System Time [ns]\n"
                   "Linear Trend Removed ({:.2f} ns intercept, "
                   "{:.2E} drift coefficient)")
            bestfit, driftcoeff, linregress = self.linregress
            y_label = fmt.format(self.channel_description,
                                 linregress.intercept / SEC_PER['ns'],
                                 driftcoeff)
        else:
            fmt = ("Difference between {} and Distribution\nSystem Time [ns], "
                   "Offset Removed ({:.2f} ns)")
            y_label = fmt.format(self.channel_description,
                                 self.trend / SEC_PER['ns'])
        return y_label
    @property
    def t_axis(self):
        """Return the default t-axis, which is scaled to be in days from the
        start of the observation run. ``Matplotlib`` refers to this as the
        x-axis (the horizontal one)."""
        return (self.plot_vars.times - self.start) / SEC_PER[self.t_units]
    @property
    def t_lim(self):
        """Return a tuple containing the left and right t-limits for this
        plot."""
        tmin = self.t_axis.min()
        tmax = self.t_axis.max()
        if self.plot_properties.has_key('xlim_left'):
            left = self.plot_properties['xlim_left']
        # if the start is very close to zero, i.e. less than 2% of the full
        # timespan, round the left limit to zero
        elif float(tmin) / (tmax - tmin) < 2e-2:
            left = 0
        else:
            left = tmin
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
        if self.run is None:
            fmt = "Time Since {} UTC [{}]"
        else:
            fmt = "Time Since Start of Run ({} UTC) [{}]"
        return fmt.format(t0, self.t_units)
    @property
    def t_ticks(self):
        """Get time ticks for this plot. Should always be between 5 and 10
        tickmarcks."""
        start, end = self.t_lim
        logduration = np.log10(end - start)
        logdurationflr = int(np.floor(logduration))
        # pick number of days per tick so that we have 5 - 10 ticks
        if logduration - logdurationflr > 0.65:
            tick_duration = int(10**logdurationflr)
        elif logduration - logdurationflr > 0.3:
            tick_duration = int(10**logdurationflr // 2)
        else:
            tick_duration = int(10**logdurationflr // 5)
        return [int(t) for t in np.arange(np.ceil(start), end, tick_duration)]
    @property
    def t_units(self):
        """Get units to use for the t-axis; for very short plots, don't use
        days; use seconds."""
        if self.end - self.start < 2*SEC_PER['days']:
            return 's'
        else:
            return 'days'
    @property
    def sanitized_dq_flag(self):
        """get the DQ Flag name as used in filenames, i.e. with commas (,) and
        colons (:) replaced in order to fit filename conventions."""
        return geco_gwpy_dump.sanitize_for_filename(self.dq_flag)
    LinRegress = collections.namedtuple('LinRegress',
                                        ['bestfit', 'driftcoeff', 'linregress'])
    @property
    @Cacheable._cacheable
    def linregress(self):
        """Get the linear regression of the mean values in this plot. Returns
        a tuple containing the best-fit line y-values for this plotter's
        t_axis, the drift coefficient, and the ``linregress`` named tuple from
        scipy.stats.linregress."""
        r = scipy.stats.linregress(self.t_axis, self.plot_vars.means)
        bestfit = r.slope * self.t_axis + r.intercept
        driftcoeff = r.slope / SEC_PER[self.t_units]
        return self.LinRegress(bestfit=bestfit, driftcoeff=driftcoeff,
                               linregress=r)
    @property
    def trend(self):
        """Should we subtract the mean value of the timeseries from each plot?
        if subtract means is defined as a number, then we will subtract that
        value out instead of the mean."""
        if self.plot_properties['detrend'] == 'mean':
            trend = self.plot_vars.means.mean()
        elif self.plot_properties['detrend'] == 'none':
            trend = 0
        elif self.plot_properties['detrend'] == 'linear':
            trend, driftcoeff, linregress = self.linregress
        else:
            trend = self.plot_properties['detrend']
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
        # label, scale, and resize the plot
        fig.set_size_inches((self.width, self.height))
        # mark any bad values as specified in plot_properties
        self.plot_bad_value_markers(ax)
        self.size_and_label(ax)
        return fig

class TrendDataPlotter(Plotter):
    """A plotter with a ``stats`` property that can be used to calculate
    statistics on a trend timeseries."""
    __metaclass__ = abc.ABCMeta
    @Cacheable._cacheable
    @fetch_data_first(False)
    def read(self, **kwargs):
        """Read a dict of lists of timeseries for this channel corresponding to
        the channel/trend combinations loaded time intervals when this
        Plotter's ``DataQualityFlag`` was active. The dictionary key is just
        the full channel/trend combination."""
        ts = {}
        for q in self.queries:
            ts[q.channel] = q.read_and_split_into_segments(self.dq_segments)
        return ts
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

class FullDataPlotter(Plotter):
    """A plotter that is designed to plot full timeseries data rather than
    trend data. Accomplishes this by overriding a few of the default Plotter
    methods."""
    __metaclass__ = abc.ABCMeta
    class PlotVars(AbstractPlotVarHolder):
        # NB we don't actually use the list of channels in our getter funcs
        # because this Plotter subclass is so simple and only has one channel.
        @staticmethod
        def _channels(plotter):
            assert len(plotter.trends) == 1
            return [plotter.channel + plotter.trends[0]]
        @staticmethod
        def _means(plotter, channels):
            return plotter.read()[0].value
        @staticmethod
        def _times(plotter, channels):
            # we don't actually use the list of channels because this Plotter
            # subclass is so simple.
            return plotter.read()[0].times.value
        plot_var_generators = [PlotVarGenerator(_means, _channels, 'means'),
                               PlotVarGenerator(_times, _channels, 'times')]
    @Cacheable._cacheable
    @fetch_data_first(True)
    def read(self, **kwargs):
        """Read a dict of timeseries for all channels associated with this
        plotter."""
        ts = {}
        for q in self.queries:
            ts[q.channel] = q.read()
        return ts

class PlotVarGenerator(object)
    """A class for containing variables (that will be plotted) in a structured
    way that preserves information about the data set the variables were
    generated from as well as the methods used to generate the data. Each
    PlotVarGenerator is independent of any specific ``Plotter`` instance."""
    def __init__(self, channelmethod, method, name):
        """Specify a method for generating the list of EPICS channels needed to
        generate this variable using a ``Plotter`` instance as input, a method
        for generating this particular variable from the plotter and the
        channel name, and a name for this particular variable as it is likely
        to appear in a plot.  The method should take a plotter instance
        followed by the list of channels as its arguments."""
        self.channelmethod = channelmethod
        self.method = method
        self.name = name
    @property
    def value(self, plotter):
        return self.method(plotter, channelmethod(plotter))

class AbstractPlotVarHolder(object):
    """Calculate and store the values bunch of PlotVarGenerators and provide a
    clean interface to them, e.g. plotter.plot_vars.means, while also providing
    an interface to the methods, channels, and names of those plotters through
    the ``methods``, ``channels``, and ``names`` attributes. Each PlotVarHolder
    is tied to a specific ``Plotter`` instance."""
    __metaclass__ = abc.ABCMeta
    def __init__(self, plotter):
        """Simply pass the plotter. Values will be calculated at initialization
        time, but the generators and other data used to make those values will
        still be stored in the PlotVarHolder instance for later use."""
        self.plotter = plotter
        for pvg in self.plot_var_generators:
            setattr(self, pvg.value(plotter))
    @property
    def methods(self):
        """Get a namedtuple with the methods used for each plot_var. These
        should each take a ``Plotter`` instance followed by a list of the
        necessary channel names."""
        methods = dict()
        for pvg in self.plot_var_generators:
            methods[pvg.name] = pvg.method
        return self.namedtuple(**methods)
    @property
    def channels(self):
        """Get a namedtuple with the channels used for each plot_var."""
        channels = dict()
        for pvg in self.plot_var_generators:
            channels[pvg.name] = pvg.channelmethod(self.plotter)
        return self.namedtuple(**channels)
    @property
    def channelmethods(self):
        """Get a namedtuple with the channel methods (i.e. the methods used to
        generate each channel name given a plotter) used for each plot_var."""
        channelmethods = dict()
        for pvg in self.plot_var_generators:
            channelmethods[pvg.name] = pvg.channelmethod
        return self.namedtuple(**channelmethods)
    @property
    def namedtuple(self):
        """Get a namedtuple corresponding to the plot_vars variable names."""
        return collections.namedtuple(name, self.varnames)
    @property
    def varnames(self):
        """Names of the plot variables."""
        return [pvg.name for pvg in self.plot_var_generators]
    @abc.abstractproperty
    def plot_var_generators(self):
        """A list of plot_var_generators used by this class."""

def PlotVarHolderFactory(newclassname, *plot_var_generators):
    """Generate a PlotVarHolder subclass that will take a ``Plotter`` as an
    initialization argument."""
    return type(newclassname, (AbstractPlotVarHolder,),
                {'plot_var_generators': plot_var_generators})

###############################################################################
#
# IMPLEMENTATION CLASSES
#
###############################################################################

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

class IndividualPlotter(TrendDataPlotter):
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
        return '{}.{}{}.{}'.format(self.queries[0].fname,
                                   self.sanitized_dq_flag, desc,
                                   DEFAULT_PLOT_FILETYPE)
    @property
    def PlotVars(self):
        return self.Stats
    @property
    def plot_vars(self):
        return self.stats[self.queries[0].channel]
    @fetch_data_first(False)
    def plot_timeseries(self, ax, **kwargs):
        """Scale up by 10^9 since plots are in ns, not seconds.
        Remove any indices considered bad in ``plot_properties``"""
        ax.errorbar(np.delete(self.t_axis, self.bad_indices.means),
                    np.delete(self.plot_vars.means - self.trend,
                              self.bad_indices.means) / SEC_PER['ns'],
                    marker="o", color="green",
                    linestyle='none',
                    yerr=np.delete(self.plot_vars.stds,
                                   self.bad_indices.means) / SEC_PER['ns'],
                    label="Means +/- Std. Dev.")
        ax.scatter(np.delete(self.t_axis, self.bad_indices.mins),
                   np.delete(self.plot_vars.mins - self.trend,
                             self.bad_indices.mins) / SEC_PER['ns'],
                   marker="^", color="blue", label="Minima")
        ax.scatter(np.delete(self.t_axis, self.bad_indices.maxs),
                   np.delete(self.plot_vars.maxs - self.trend,
                             self.bad_indices.maxs) / SEC_PER['ns'],
                   marker="v", color="red", label="Maxima")

class CombinedPlotter(TrendDataPlotter):
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
        if self.run is None:
            start = gwpy.time.from_gps(self.start)
            end = gwpy.time.from_gps(self.end)
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
        fmt = '{}__{}__{}.{}.combined{}.{}'
        return fmt.format(self.start, self.end, ch, self.sanitized_dq_flag,
                          desc, DEFAULT_PLOT_FILETYPE)
    PlotVars = collections.namedtuple('CombinedPlotterVars',
                                      ['means', 'absmins', 'absmaxs',
                                       'stds', 'times'])
    class PlotVars(AbstractPlotVarHolder):
        plot_var_generators = [
            PlotVarGenerator(trend_var('maxs'),  trend_chan('max'),  'absmaxs'),
            PlotVarGenerator(trend_var('mins'),  trend_chan('min'),  'absmins'),
            PlotVarGenerator(trend_var('means'), trend_chan('mean'), 'means'),
            PlotVarGenerator(trend_var('times'), trend_chan('mean'), 'times'),
            PlotVarGenerator(trend_var('stds'),  trend_chan('stds'), 'stds')
        ]
    @fetch_data_first(False)
    def plot_timeseries(self, ax, **kwargs):
        """Scale up by 10^9 since plots are in ns, not seconds.
        Remove any indices considered bad in ``plot_properties``"""
        ax.errorbar(np.delete(self.t_axis, self.bad_indices.means),
                    np.delete(self.plot_vars.means - self.trend,
                              self.bad_indices.means) / SEC_PER['ns'],
                    marker="o", color="green",
                    linestyle='none',
                    yerr=np.delete(self.plot_vars.stds,
                                   self.bad_indices.means) / SEC_PER['ns'],
                    label="Means +/- Std. Dev.")
        ax.scatter(np.delete(self.t_axis, self.bad_indices.absmins),
                   np.delete(self.plot_vars.absmins - self.trend,
                             self.bad_indices.absmins) / SEC_PER['ns'],
                   marker="^", color="blue", label="Abs. Minima")
        ax.scatter(np.delete(self.t_axis, self.bad_indices.absmaxs),
                   np.delete(self.plot_vars.absmaxs - self.trend,
                             self.bad_indices.absmaxs) / SEC_PER['ns'],
                   marker="v", color="red", label="Abs. Maxima")

class BadTimesZoomPlotter(FullDataPlotter):
    """A Plotter that makes zoomed in plots meant to be used for seeing
    missing data. To be more precise, this Plotter is used to show the full
    data from a segment that has missing data; the start and end should be
    the start and end of that data segment."""
    def __init__(self, start, end, channel, dq_flag, trends,
                 days_from_start, dq_segment, dq_segment_index, 
                 ext=geco_gwpy_dump.DEFAULT_EXTENSION,
                 run=None, channel_description=None,
                 height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
                 plot_properties=DEFAULT_PLOT_PROPERTIES):
        """Similar to other ``Plotter`` __init__ interfaces, but with the
        ability to optionally specify the number of days from the start of
        an observation run as ``days_from_start`` and the ability to specify
        the index of the dq_segment which is here being plotted in full. The
        actual start and end of the dq_segment can be labelled if a tuple or
        list with the start and end times of this dq_segment is passed as the
        ``dq_segment`` variable. The dq_segment index within the parent
        Plotter's list of segments can also be passed as the
        ``dq_segment_index`` variable.  In this case, it will be added to the
        title, making it quicker to find which dq segment contained the bad
        times just by looking at the plots."""
        if channel_description is None:
            channel_description = channel
        self.start = start
        self.end = end
        self.channel = channel
        self.dq_flag = dq_flag
        self.days_from_start = days_from_start
        self.dq_segment = dq_segment
        self.dq_segment_index = dq_segment_index
        self.trends = trends
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
        fmt = 'Bad Segment in {} from {} to {}\nSegment no. {} from Flag {}'
        if not self.run is None:
            fmt += ' during {}'.format(self.run)
        return fmt.format(self.channel_description, start, end,
                          self.dq_segment_index, self.dq_flag)
    @property
    def fname(self):
        if self.plot_properties['fname_desc'] is None:
            desc = ""
        else:
            desc = "." + self.plot_properties['fname_desc']
        fmt = '{}__{}__{}.{}.badtime.ind__{}{}.{}'
        return fmt.format(self.start, self.end, ch, self.sanitized_dq_flag,
                          self.dq_segment_index, desc, DEFAULT_PLOT_FILETYPE)
    @fetch_data_first(True)
    def plot_timeseries(self, ax, **kwargs):
        ax.plot(np.delete(self.t_axis, self.bad_indices.means),
                np.delete(self.plot_vars.means, self.bad_indices.means),
                marker="o", color="green", label="Recorded Signal")
        # put the start and/or end time in the plot as a vertical line
        if self.plot_vars.times.min() <= self.start:
            forest_green = '#228B22'
            plot_vertical_marker(ax, self.start / SEC_PER[self.t_units],
                                 label="Start of Segment", color=forest_green)
        if self.end <= self.plot_vars.times.max():
            midnight_blue = '#191970'
            plot_vertical_marker(ax, self.end / SEC_PER[self.t_units],
                                 label="End of Segment", color=midnight_blue)
    def t_label(self):
        """Return the default t-axis label for a t-axis measuring days since
        the start of an observing run. ``Matplotlib`` refers to this as the
        x-axis (the horizontal one)."""
        t0 = gwpy.time.tconvert(self.start).strftime("%c")
        fmt = "Time Since Start of Fault during Segment {} at {} UTC [{}]"
        if not self.run is None:
            fmt += ' during {}'.format(self.run)
        return fmt.format(self.dq_segment_index, t0, self.t_units)

def main():
    if len(sys.argv) == 1:
        plt_job = PlottingJob.load()
    else:
        plt_job = PlottingJob.load(sys.argv[1])
    plt_job.make_combined_plots()
    plt_job.make_individual_plots()

if __name__ == "__main__":
    main()
