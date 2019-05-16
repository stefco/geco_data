#!/usr/bin/env python
# (c) Stefan Countryman 2017

###############################################################################
#
# IMPORT ARGUMENT PARSER
#
###############################################################################

import argparse
import textwrap
import collections
# ALL OTHER IMPORTS LISTED AFTER ARGUMENT PARSING; this allows for fast
# documentation access. Variables are declared first; import follow.

###############################################################################
#
# CONSTANTS
#
###############################################################################

DESC = """Generate plots for months worth of aLIGO timing diagnostic data
along with zoomed plots highlighting anomalous sections of each channel.
Almost everything about the plots is specified in ``jobspec.json``."""
DEFAULT_TREND = ''
DEFAULT_PLOT_FILETYPE = 'png'
DEFAULT_PLOT_PROPERTIES = {
    "omitted_indices": list(),
    "outliers_lower_bound": -1e-6,
    "outliers_upper_bound": 1e-6,
    "omitted_zorder": -2,
    "missing_zorder": 101,
    "unrecorded_zorder": 100,
    "outliers_zorder": -1,
    "start_end_zorder": 200,
    "omitted_label": "Omitted Values",
    "missing_label": "Data Not Found",
    "unrecorded_label": "Data Not Taken",
    "outliers_label": "Outliers",
    "omitted_color": "#41e5f4",             # teal, sorta
    "missing_color": "purple",
    "unrecorded_color": "orange",
    "outliers_color": "black",
    "handle_omitted_values": "hide",
    "handle_missing_values": "mark",
    "handle_unrecorded_values": "mark",
    "handle_outliers_values": "markshow",
    "detrend": 'mean',
    "detector_offline": [],
    "fname_desc": None
}
PLOT_PROPERTY_DESCRIPTIONS = {
    "outliers_lower_bound": "Smallest value not considered an outlier [s].",
    "outliers_upper_bound": "Largest value not considered an outlier [s].",
    "omitted_indices": ("Optionally, provide a list of indices that should be "
                        "removed from the timeseries before plotting. This "
                        "data will not be displayed in plots, but "
                        "bad_index_types will still find missing and "
                        "unrecorded values in the original timeseries."),
    "omitted_zorder": "Higher numbers plot this marker over other markers.",
    "missing_zorder": "Higher numbers plot this marker over other markers.",
    "unrecorded_zorder": "Higher numbers plot this marker over other markers.",
    "outliers_zorder": "Higher numbers plot this marker over other markers.",
    "start_end_zorder": ("The z-order of data quality segment start/end "
                         "markers. Higher numbers plot this marker over other "
                         "markers."),
    "unrecorded_label": ("If ``handle_unrecorded_values`` is set to 'mark`, "
                         "mark unrecorded values on the final plot and "
                         "label them in the legend using this string."),
    "missing_label": ("If ``handle_missing_values`` is set to 'mark`, we "
                      "will mark missing values on the final plot and "
                      "label them in the legend using this string."),
    "omitted_label": ("If ``handle_omitted_values`` is set to 'mark`, we "
                      "will mark omitted values on the final plot and will "
                      "label them in the legend using this string."),
    "outliers_label": ("If ``handle_outliers_values`` is set to 'mark`, we "
                       "will mark outlier values on the final plot and "
                       "label them in the legend using this string."),
    "omitted_color": "Color for omitted value time markers on plot.",
    "missing_color": "Color for missing value time markers on plot.",
    "unrecorded_color": "Color for unrecorded value time markers on plot.",
    "outliers_color": "Color for outlier time markers on plot.",
    "handle_omitted_values": ("``mark`` will clearly mark omitted indices "
                              "in the final plot but will NOT show "
                              "the value at that index for ANY timeseries "
                              "at that index. ``hide`` will remove those "
                              "indices entirely, including from "
                              "from the plot (DEFAULT). ``ignore`` "
                              "will keep those indices in the plot with "
                              "their placeholder value intact (useful if "
                              "that placeholder value is suspected to be "
                              "the true value at that point). ``markshow`` "
                              "will mark the time of the omitted value but "
                              "will keep the omitted data points in."),
    "handle_missing_values": ("``mark`` will clearly mark missing "
                              "values in the final plot and will show "
                              "the value at that index for any timeseries "
                              "that does not contain missing data "
                              "(DEFAULT). ``hide`` will remove those "
                              "indices entirely from the plot. ``ignore`` "
                              "will keep those indices in the plot with "
                              "their placeholder value intact (useful if "
                              "that placeholder value is suspected to be "
                              "the true value at that point). ``markshow`` "
                              "will mark the time of the missing value but "
                              "will keep the missing data points in."),
    "handle_unrecorded_values": ("``mark`` will clearly mark unrecorded "
                                 "values in the final plot and will show "
                                 "the value at that index for any timeseries "
                                 "that does not contain missing data "
                                 "(DEFAULT). ``hide`` will remove those "
                                 "indices entirely from the plot. ``ignore`` "
                                 "will keep those indices in the plot with "
                                 "their placeholder value intact (useful if "
                                 "that placeholder value is suspected to be "
                                 "the true value at that point). ``markshow`` "
                                 "will mark the time of the unrecorded value "
                                 "but will keep the data points in."),
    "handle_outliers_values": ("``mark`` will clearly mark outlier "
                               "values in the final plot and will show "
                               "the value at that index for any timeseries "
                               "that does not contain outliers "
                               "(DEFAULT). ``hide`` will remove those "
                               "indices entirely from the plot. ``ignore`` "
                               "will keep those indices in the plot with "
                               "their outlier value intact (useful if "
                               "that outlier value is suspected to be "
                               "the true value at that point). ``markshow`` "
                               "will mark the time of the outlier but "
                               "will keep the outlier data points in."),
    "detrend": ("If equals 'mean', subtract means from plots. If equals "
                "'none', do not shift the plots. If equals 'linear', remove "
                "the linear trend from the plots. If a number, "
                "subtract that number from the plots."),
    "detector_offline": "List of GPS start/stop tuples when detector was off",
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
DEFAULT_AXES_POSITION = [0.125, 0.1, 0.775, 0.73]
# DEFAULT_LEGEND_FONT
DEFAULT_TITLE_OFFSET = 1.07
COMBINED_TRENDS = [
    ".mean,m-trend",
    ".min,m-trend",
    ".max,m-trend",
    ".rms,m-trend",
    ".n,m-trend"
]
# a dict mapping CLI plot names to PlottingJob bound plotting methods
CLI_PLOTTING_OPTIONS = collections.OrderedDict([
    ('individual', 'make_individual_plots'),
    ('combined',   'make_combined_plots'),
    ('zoom',       'make_bad_time_zoom_plots'),
    ('doublezoom', 'make_bad_time_double_zoom_plots')
])
CLI_DEFAULT_JOBSPEC = "jobspec.json"

###############################################################################
#
# CONFIGURE ARGUMENT PARSER
#
###############################################################################

# a function for documenting plot_properties
def print_plot_properties():
    """Print out a list of plot properties and their descriptions."""
    for key, val in PLOT_PROPERTY_DESCRIPTIONS.items():
        print(textwrap.fill('``{}``: {}'.format(key, val), width=72,
                            initial_indent='- ', subsequent_indent='  '))

# quits immediately on --help or -h flags to avoid slow imports. only runs
# if this module is used as a script and called from the command line.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("-p", "--plots",
                        choices=tuple(CLI_PLOTTING_OPTIONS.keys()),
                        nargs='*', default=CLI_PLOTTING_OPTIONS.keys(),
                        help=("Which types of plots to make? Defaults to "
                              "{}").format(CLI_PLOTTING_OPTIONS.keys()))
    parser.add_argument("-f", "--faults", action='store_true',
                        help=("Print out a detailed taxonomy of all anomalous "
                              "time segments in all plots defined in this "
                              "jobspec, then immediately quit. Very useful for "
                              "debugging plots and methodically identifying "
                              "faults."))
    parser.add_argument("-j", "--jobspec", default=CLI_DEFAULT_JOBSPEC,
                        help=("Which job specification file to load from?"
                              "Defaults to {}").format(CLI_DEFAULT_JOBSPEC))
    parser.add_argument("--helpplotprops", action='store_true',
                        help=("Print out a list of "
                              "``plot_properties`` keys and descriptions of "
                              "what they mean, then quit immediately."))
    args = parser.parse_args()
    if args.helpplotprops:
        print_plot_properties()
        exit()

###############################################################################
#
# IMPORTS
#
###############################################################################

import matplotlib
if __name__ == '__main__':
    # necessary for headless plotting
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
# make matplotlib legend fonts smaller so that they take up less space
DEFAULT_LEGEND_FONT = matplotlib.font_manager.FontProperties()
DEFAULT_LEGEND_FONT.set_size('small')
import matplotlib.patches
import multiprocessing
import numpy as np
import scipy.stats
import geco_gwpy_dump
from geco_gwpy_dump import SEC_PER
import gwpy.segments
import gwpy.time
import h5py
import logging
import json
import sys
import abc

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

def fetch_data(job, multiproc=False, getmethod='fetch'):
    """Fetch data specified for a given job."""
    logging.info('Fetching data for job: {}'.format(job))
    logging.debug('Running queries for job: {}'.format(job))
    geco_gwpy_dump._run_queries(job, multiproc=multiproc, getmethod=getmethod)
    logging.debug('Concatenating data for job: {}'.format(job))
    job.concatenate_files()
    # this will run even if we don't have an m-trend, in which
    # case it will just harmlessly return.
    logging.debug('Find missing m-trend for job: {}'.format(job))
    job.fill_in_missing_m_trend()

def fetch_data_first(fetch_first, multiproc=False, getmethod='fetch'):
    """A decorator that checks whether a given Plotter's job is done (i.e.
    whether data has been downloaded) and fetches any missing data if it
    is not yet finished. Methods that don't need to dump a ton of data before
    being run should have this decorator applied so that they can dynamically
    fetch data from NDS2 even if it hasn't been downloaded yet. Longer data
    dumps should avoid this decorator so as to avoid blocking. This can be
    overridden when the function is called by passing the ``fetch_data_first``
    argument to the original function. Can explicitly defer the decision on
    fetching data as needed to subclasses and instances, or to the user,
    by setting ``fetch_first`` to be 'defer'."""
    def real_decorator(func):
        def wrapper(plotter, *args, **kwargs):
            fetch = fetch_first
            # can override the default fetching behavior by passing the
            # ``fetch_data_first`` kwarg to the function in question
            if kwargs.has_key('fetch_data_first'):
                fetch = kwargs['fetch_data_first']
            # can also automatically download data if the plotter has a
            # ``fetch_data_first`` attribute that is set to ``True``
            elif hasattr(plotter, 'fetch_data_first'):
                if getattr(plotter, 'fetch_data_first') is True:
                    fetch = True
            # can also implicitly pass the buck to instances by omitting a
            # default with fetch_first='defer', but if this is done, we must
            # get the decision on the behavior from the instance or the calling
            # user. otherwise, raise an exception.
            elif fetch == 'defer':
                msg = ("when deferring fetch_first, must specify value in "
                       "instance or via kwarg.")
                raise ValueError(msg)
            elif not (fetch is True) and not (fetch is False):
                msg = "``fetch_first`` must equal True, False, or 'defer'."
                raise ValueError(msg)
            j = plotter.job
            if not j.is_finished:
                if fetch:
                    fetch_data(j, multiproc=multiproc, getmethod=getmethod)
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

def get_outlier_indices(array, minval, maxval):
    """Get indices in some array whose values are outside of some interval.
    Used, of course, to find outliers."""
    # plus operator is elementwise logical or to numpy; times is and.
    return np.nonzero(   (   (array != MISSING_VALUE_CONSTANT)
                           * (array != UNRECORDED_VALUE_CONSTANT) )
                       * ( (array <= minval) + (maxval <= array) ) )[0]

def plot_vertical_marker(ax, t, label="marked", color="pink", marker="x",
                         zorder=100):
    """Plot a vertical marker (of the sort used to mark bad indices) to the
    supplied ``matplotlib`` Axes object."""
    # we will want to reset the original y limits after adding these markers,
    # so store them now
    ylim = ax.get_ylim()
    y_offset = (ylim[1] + ylim[0]) / 2.
    y_span = (ylim[1] - ylim[0])
    ax.errorbar(t, len(t)*[y_offset], color=color, linestyle='none',
                zorder=zorder, marker=marker, label=label, yerr=4*y_span)
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

def get_channel_trend(full_channel_name):
    """Split an EPICS channel name with trend extension into a tuple containing
    the bare EPICS channel name (full data) and a string for the
    trend-extension."""
    if '.' in full_channel_name:
        i = full_channel_name.index('.')
        return (full_channel_name[0:i], full_channel_name[i:])
    else:
        return (full_channel_name, '')

@multiprocessing_traceback
def _save_plot(plotter):
    """Must define this at the global level to allow for multiprocessing."""
    type(plotter)._save_plot(plotter)

###############################################################################
#
# PLOT VARIABLE CLASSES (For Plotter subclasses)
#
###############################################################################

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
            setattr(self, pvg.name, pvg.value(plotter))
    @property
    def channels(self):
        """Get a namedtuple with the channels used for each plot_var."""
        channels = dict()
        for pvg in self.plot_var_generators:
            channels[pvg.name] = pvg.channelmethod(self.plotter)
        return self.namedtuple(**channels)
    @property
    def methods(self):
        """Get a namedtuple with the methods used for each plot_var. These
        should each take a ``Plotter`` instance followed by a list of the
        necessary channel names."""
        return self._methods()
    @classmethod
    def _methods(cls):
        methods = dict()
        for pvg in cls._plot_var_generators():
            methods[pvg.name] = pvg.method
        return cls._namedtuple()(**methods)
    @property
    def channelmethods(self):
        """Get a namedtuple with the channel methods (i.e. the methods used to
        generate each channel name given a plotter) used for each plot_var."""
        return self._channelmethods()
    @classmethod
    def _channelmethods(cls):
        channelmethods = dict()
        for pvg in cls._plot_var_generators():
            channelmethods[pvg.name] = pvg.channelmethod
        return cls._namedtuple()(**channelmethods)
    @property
    def namedtuple(self):
        """Get a namedtuple corresponding to the plot_vars variable names."""
        return self._namedtuple()
    @classmethod
    def _namedtuple(cls):
        return collections.namedtuple('PlotVars', cls._names())
    @property
    def names(self):
        """Names of the plot variables."""
        return self._names()
    @classmethod
    def _names(cls):
        return [pvg.name for pvg in cls._plot_var_generators()]
    @property
    def plot_var_generators(self):
        """A list of plot_var_generators used by this class."""
        return self._plot_var_generators()
    #classmethod
    @abc.abstractmethod
    def _plot_var_generators(cls):
        """A list of plot_var_generators that can be accessed as a class
        method."""

class FullDataPlotVars(AbstractPlotVarHolder):
    @staticmethod
    def _channels(plotter):
        assert len(plotter.trends) == 1
        return [plotter.channel + plotter.trends[0]]
    @staticmethod
    def _means(plotter, channels):
        return plotter.read()[channels[0]].value
    @staticmethod
    def _times(plotter, channels):
        # we don't actually use the list of channels because this Plotter
        # subclass is so simple.
        return plotter.read()[channels[0]].times.value
    @classmethod
    def _plot_var_generators(cls):
        return [PlotVarGenerator(cls._means, cls._channels, 'means'),
                PlotVarGenerator(cls._times, cls._channels, 'times')]

class IndividualPlotVars(AbstractPlotVarHolder):
    @staticmethod
    def _chan(plotter):
        return [plotter.channel + plotter.trends[0]]
    @staticmethod
    def _var(attr):
        """Returns a function that takes a plotter and channel and fetches the
        given attribute from the plotter.stats for that channel. Used for
        TrendDataPlotter plot_vars."""
        def getter(plotter, channels):
            return getattr(plotter.stats[channels[0]], attr)
        return getter
    @classmethod
    def _plot_var_generators(cls):
        return [PlotVarGenerator(cls._var(k), cls._chan, k)
                for k in ['maxs', 'mins', 'means', 'stds', 'times']]

class CombinedPlotVars(AbstractPlotVarHolder):
    """Plot variable container for CombinedPlotter."""
    @staticmethod
    def _var(attr):
        """Returns a function that takes a plotter and channel and fetches the
        given attribute from the plotter.stats for that channel. Used for
        TrendDataPlotter plot_vars."""
        def getter(plotter, channels):
            return getattr(plotter.stats[channels[0]], attr)
        return getter
    @staticmethod
    def _chan(trend, ext=',m-trend'):
        """Returns a function that gets the channel name from a given plotter by
        specifying the trend to append to that channel. Used for TrendDataPlotter
        plot_vars."""
        def channelmethod(plotter):
            return ['{}.{}{}'.format(plotter.channel, trend, ext)]
        return channelmethod
    @classmethod
    def _plot_var_generators(c):
        return [
            PlotVarGenerator(c._var('maxs'),  c._chan('max'),  'absmaxs'),
            PlotVarGenerator(c._var('mins'),  c._chan('min'),  'absmins'),
            PlotVarGenerator(c._var('means'), c._chan('mean'), 'means'),
            PlotVarGenerator(c._var('times'), c._chan('mean'), 'times'),
            PlotVarGenerator(c._var('stds'),  c._chan('mean'), 'stds')
        ]

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
                                  trends = self.trends,
                                  max_chunk_length = self.max_chunk_length)
    def fetch_data(self, multiproc=True, getmethod='fetch'):
        """Fetch the data needed for this plot from local gravitational wave
        frame files or from NDS2 using GWpy-based methods found in
        geco_gwpy_dump. Since this is liable to be called by a user
        interactively, it defaults to multiprocess operation."""
        fetch_data(self.job, multiproc=multiproc, getmethod=getmethod)
    @property
    def max_chunk_length(self):
        """Get the maximum chunk length to download at once for this channel.
        In units of seconds. Based off of the types of trend data needed;
        higher sample rate data should be fetched in smaller chunks."""
        if '' in self.trends:
            return geco_gwpy_dump.DEFAULT_MAX_CHUNK
        elif any(['s-trend' in trend for trend in self.trends]):
            return SEC_PER['minutes'] * 30
        else:
            return SEC_PER['days']
    @property
    def queries(self):
        """Get the ``geco_gwpy_dump.Query`` objects corresponding to this
        ``Plotter``."""
        return self.job.full_queries
    # a way of storing indices of types of bad data.
    BadIndices = collections.namedtuple('BadIndices',
                                        ['unrecorded', 'missing', 'omitted',
                                         'outliers'])
    # a way of grouping t and y axis arrays.
    AxisArrays = collections.namedtuple('AxisArrays', ['y_axis', 't_axis'])
    @abc.abstractproperty
    def bad_time_zoom_plots(self):
        """Make plots showing zoomed views of the bad regions of this
        ``Plotter`` instance's data."""
    def make_bad_time_zoom_plots(self):
        """Save all individual plots, i.e. every channel and every dq_flag gets
        its own plot."""
        #mapf = multiprocessing.Pool(processes=NUM_THREADS).map
        mapf = map
        mapf(_save_plot, self.bad_time_zoom_plots)
    @abc.abstractproperty
    def PlotVars(self):
        """Define a class for the data arrays that will actually be
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
        bad = {'unrecorded': dict(), 'missing': dict(), 'omitted': dict(),
               'outliers': dict()}
        minval = self.plot_properties['outliers_lower_bound']
        maxval = self.plot_properties['outliers_upper_bound']
        plot_vars = self.plot_vars
        # get the bad indices of each type
        for varname in self.PlotVars._names():
            array = getattr(plot_vars, varname)
            # unfortunately, when we plot standard deviations, they can often
            # be zero coincidentally (especially in the case of n trends,
            # which are usually solidly e.g. 960 samples per minute, leading
            # to spurious warnings.
            if varname == 'stds':
                bad['unrecorded'][varname] = list()
            else:
                bad['unrecorded'][varname] = list(get_unrecorded_indices(array))
            bad['missing'][varname] = list(get_missing_indices(array))
            bad['omitted'][varname] = self.plot_properties['omitted_indices']
            # times will obviously be outside of the range of accepted
            # outliers. also exclude "n" trend values, since these should
            # usually be in the hundreds.
            if varname == 'times' or self.trends[0][0:2] == '.n':
                bad['outliers'][varname] = list()
            else:
                bad['outliers'][varname] = list(get_outlier_indices(array,
                                                                    minval,
                                                                    maxval))
        # put the bad indices in PlotVars namedtuples inside a BadIndices
        # namedtuple
        nt = self.PlotVars._namedtuple()
        return self.BadIndices(unrecorded = nt(**bad['unrecorded']),
                               missing = nt(**bad['missing']),
                               omitted = nt(**bad['omitted']),
                               outliers = nt(**bad['outliers']))
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
        for array_name in self.PlotVars._names():
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
        for array_name in self.PlotVars._names():
            bad_sets[array_name] = bad_sets[array_name].union(shared_bad_set)
            # make sure we are returning numpy arrays for our index lists
            bad_sets[array_name] = np.sort(list(bad_sets[array_name]))
        # put everything into a plot variable namedtuple
        return self.PlotVars._namedtuple()(**bad_sets)
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
        """If there are bad values (missing, unrecorded, purposefully omitted,
        or outliers), plot any that are supposed to be marked on the given
        matplotlib Axes instance."""
        for badness_type in self.BadIndices._fields:
            handle_method_key = 'handle_{}_values'.format(badness_type)
            color = self.plot_properties['{}_color'.format(badness_type)]
            label = self.plot_properties['{}_label'.format(badness_type)]
            zorder = self.plot_properties['{}_zorder'.format(badness_type)]
            # get all omitted indices. this approach is future safe, in case
            # i allow for omissions of specific plot_vars in the future.
            omitted = list(set.union(*[set(o) for o in
                                       self.bad_index_types.omitted]))
            if self.plot_properties[handle_method_key] in ["mark", "markshow"]:
                bad_inds = getattr(self.bad_index_types, badness_type)
                # get all bad indices (mins, maxs, etc.) for this badness type
                all_inds = list(set.union(*[set(b) for b in bad_inds]))
                # don't mark omitted indices as part of other badness types;
                # assume the user omitted them for a reason.
                if not badness_type is 'omitted':
                    all_inds = filter(lambda i: not i in omitted, all_inds)
                if len(all_inds) != 0:
                    plot_vertical_marker(ax, self.t_axis[all_inds],
                                         zorder=zorder, label=label,
                                         color=color)
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
            bestfit, driftcoeff, intercept = self.linregress
            y_label = fmt.format(self.channel_description,
                                 intercept / SEC_PER['ns'],
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
        deltat = end - start
        if deltat <= 0:
            sys.stderr.write(("Got deltat = 0 in t_ticks for {}, start={}, "
                              "end={}").format(self, start, end))
            return list(np.arange(np.ceil(start), np.ceil.start+1))
        logduration = np.log10(deltat)
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
                                        ['bestfit', 'driftcoeff', 'intercept'])
    @property
    @Cacheable._cacheable
    def linregress(self):
        """Get the linear regression of the mean values in this plot. Returns
        a tuple containing the best-fit line y-values for this plotter's
        t_axis, the drift coefficient, and the ``linregress`` named tuple from
        scipy.stats.linregress."""
        cleandata  = np.delete(self.plot_vars.means, self.bad_indices.means)
        cleantimes = np.delete(self.t_axis, self.bad_indices.means)
        if len(cleandata) != 0:
            slope, intercept, r_value, p_value, stderr = scipy.stats.linregress(
                cleantimes, cleandata)
            bestfit = slope * self.t_axis + intercept
            driftcoeff = slope / SEC_PER[self.t_units]
        else:
            bestfit = 0
            driftcoeff = 0
        return self.LinRegress(bestfit=bestfit, driftcoeff=driftcoeff,
                               intercept=intercept)
    @property
    def trend(self):
        """Subtract the trend specified in
        ``Plotter.plot_properties['detrend']`` from each plot. Trend can be 
        the 'mean' value of the plot, the 'linear' least squares best fit, a
        custom-specified number, or simply 'none' if no trend should be
        removed."""
        if self.plot_properties['detrend'] == 'mean':
            # delete bad indices before calculating the trend, since they
            # can skew the trend.
            cleandata = np.delete(self.plot_vars.means, self.bad_indices.means)
            if len(cleandata) != 0:
                trend = cleandata.mean()
            else:
                trend = 0
        elif self.plot_properties['detrend'] == 'none':
            trend = 0
        elif self.plot_properties['detrend'] == 'linear':
            trend, driftcoeff, intercept = self.linregress
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

class PlotVarGenerator(object):
    """A class for containing variables (that will be plotted) in a structured
    way that preserves information about the data set the variables were
    generated from as well as the methods used to generate the data. Each
    PlotVarGenerator is independent of any specific ``Plotter`` instance."""
    def __init__(self, method, channelmethod, name):
        """Specify a method for generating this particular variable from the
        plotter and the channel name, a method for generating the list of EPICS
        channels needed to generate this variable using a ``Plotter`` instance
        as input, and a name for this particular variable as it is likely
        to appear in a plot.  The method should take a plotter instance
        followed by the list of channels as its arguments."""
        self.method = method
        self.channelmethod = channelmethod
        self.name = name
    def value(self, plotter):
        """Get the value of this plot variable from a given plotter."""
        return self.method(plotter, self.channelmethod(plotter))

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
    # define a named tuple for statistics on saved values
    Stats = collections.namedtuple('Stats', ['means', 'mins', 'maxs',
                                             'stds', 'times', 'ns'])
    @property
    def fname_stats(self):
        return self.fname + '.stats.hdf5'
    def save_stats(self, stats):
        """Save the ``CombinedPlotter.Stats`` for this Plotter to file for
        speedier recovery in the future."""
        with h5py.File(self.fname_stats, 'w') as f:
            for ch in stats.keys():
                grp = f.create_group(ch)
                for field in stats[ch]._fields:
                    grp.create_dataset(field, data=getattr(stats[ch], field))
    def load_stats(self):
        """Try loading cached ``CombinedPlotter.Stats`` from file for this
        Plotter. If the file does not exist, an IOError will be raised."""
        with h5py.File(self.fname_stats, 'r') as f:
            stats = dict()
            for q in self.queries:
                if not set(f[q.channel].keys()) == set(self.Stats._fields):
                    raise IOError('Missing fields from cached Stats file.')
                stat_dict = dict()
                for field in f[q.channel].keys():
                    stat_dict[field] = f[q.channel][field][:]
                stats[q.channel] = self.Stats(**stat_dict)
            return stats
    def del_stats(self):
        """Delete the cached ``CombinedPlotter.Stats`` hdf5 file for this
        Plotter from disk."""
        if os.path.isfile(self.fname_stats):
            os.remove(self.fname_stats)
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
        try:
            return self.load_stats()
        except IOError:
            pass
        ts = self.read()
        stats = dict()
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
        self.save_stats(stats)
        return stats
    @property
    def bad_time_zoom_plots(self):
        """Get a list of ``Plotters`` showing zoomed views of the messed up
        time spans for this channel whether they be missing, unrecorded,
        or omitted data). Will show the raw m-trend data in the time
        segment with a fault for the trend type in which the fault was
        discovered."""
        zoomed_plots = list()
        all_bad_indices = self.bad_indices
        bad_index_types = self.bad_index_types
        t_axis = self.t_axis
        tlim = self.t_lim
        segs = self.dq_segments.active
        desc = self.channel_description
        # make sure that the way we define outliers does not change
        # for these zoomed plots; we don't want to mark spurious
        # outliers. For BadTimesZoomPlotter, these special surviving
        # plot properties will be folded into the defaults.
        plot_props = dict()
        for pp in ['outliers_lower_bound', 'outliers_upper_bound']:
            if self.plot_properties.has_key(pp):
                plot_props[pp] = self.plot_properties[pp]
        for pvg in self.PlotVars._plot_var_generators():
            varname = pvg.name
            omitted = getattr(bad_index_types.omitted, varname)
            if len(getattr(all_bad_indices, varname)) == 0:
                continue
            channel, trend = get_channel_trend(pvg.channelmethod(self)[0])
            trends = [trend]
            for badness_type in self.BadIndices._fields:
                # don't bother with omitted indices
                if badness_type == 'omitted':
                    continue
                bad_plot_vars = getattr(bad_index_types, badness_type)
                all_inds = getattr(bad_plot_vars, varname)
                # exclude omitted times
                incl_inds = filter(lambda i: not i in omitted, all_inds)
                # only show zoomed plots for data within the plot window
                bad_inds = filter(lambda i: tlim[0] < t_axis[i] < tlim[1],
                                  incl_inds)
                for i in bad_inds:
                    start = segs[i].start.gpsSeconds
                    end = segs[i].end.gpsSeconds + SEC_PER['minutes']
                    dq_segment = segs[i]
                    p = BadTimesZoomPlotter(start=start, end=end,
                                            channel=channel,
                                            dq_flag=self.dq_flag, trends=trends,
                                            days_from_start=t_axis[i],
                                            dq_segment=dq_segment,
                                            dq_segment_index=i, ext=self.ext,
                                            badness_type=badness_type,
                                            run=self.run,
                                            channel_description=desc,
                                            height=self.height,
                                            width=self.width,
                                            plot_properties=plot_props)
                    zoomed_plots.append(p)
        return zoomed_plots

class FullDataPlotter(Plotter):
    """A plotter that is designed to plot full timeseries data rather than
    trend data. Accomplishes this by overriding a few of the default Plotter
    methods."""
    __metaclass__ = abc.ABCMeta
    PlotVars = FullDataPlotVars
    @Cacheable._cacheable
    @fetch_data_first(True)
    def read(self, **kwargs):
        """Read a dict of timeseries for all channels associated with this
        plotter."""
        ts = {}
        for q in self.queries:
            ts[q.channel] = q.read()
        return ts

###############################################################################
#
# IMPLEMENTATION CLASSES
#
###############################################################################

class PlottingJob(Cacheable):
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
    def fault_taxonomy(self, fault_types=['missing', 'unrecorded', 'outliers']):
        """Print a list of bad times and their segment numbers for each
        combined plotter along with the type of bad time and the type of
        plot variable that exhibits the bad values."""
        title_fmt = geco_gwpy_dump._GREEN + '{}' + geco_gwpy_dump._CLEAR
        varname_fmt = geco_gwpy_dump._RED + '  {}:' + geco_gwpy_dump._CLEAR
        for c in self.combined_plotters:
            start, end = c.t_lim
            title_not_yet_printed = True
            t_axis = c.t_axis
            rounded_times = t_axis.round(1)
            for v in c.PlotVars._names():
                varname_not_yet_printed = True
                omitted = getattr(c.bad_index_types.omitted, v)
                for bt in fault_types:
                    # don't bother with omitted indices; these have been
                    # looked at by a human already.
                    all_segs = getattr(getattr(c.bad_index_types, bt), v)
                    # only include segments that are not omitted
                    incl_segs = filter(lambda i: not i in omitted, all_segs)
                    # only show bad times for data within the plot window
                    segs = filter(lambda i: start < t_axis[i] < end, incl_segs)
                    if len(segs) != 0:
                        int_seg_fmt = ', '.join(['{:>5d}']*len(segs))
                        flt_seg_fmt = ', '.join(['{:>5.1f}']*len(segs))
                        if title_not_yet_printed:
                            title_not_yet_printed = False
                            print(title_fmt.format(c.title))
                            print c.channel_description + ': ' + c.channel
                            print 'BAD SEGMENTS (units: ' + c.t_units + ')'
                        if varname_not_yet_printed:
                            varname_not_yet_printed = False
                            print varname_fmt.format(v)
                        segments = int_seg_fmt.format(*segs)
                        times = flt_seg_fmt.format(*rounded_times[segs])
                        print '    {}:'.format(bt)
                        print '      segments: [{}]'.format(segments)
                        print '      times:    [{}]'.format(times)
    @property
    @Cacheable._cacheable
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
        #mapf = multiprocessing.Pool(processes=NUM_THREADS).map
        mapf = map
        mapf(_save_plot, self.individual_plotters)
    @property
    @Cacheable._cacheable
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
        #mapf = multiprocessing.Pool(processes=NUM_THREADS).map
        mapf = map
        mapf(_save_plot, self.combined_plotters)
    @property
    @Cacheable._cacheable
    def bad_time_zoom_plots(self):
        """Get all zoomed plots from bad times in the ``CombinedPlotter``s
        associated with this ``PlottingJob``. Don't bother with the bad
        times from ``IndividualPlotter``s, though."""
        return sum([c.bad_time_zoom_plots for c in self.combined_plotters], [])
    @property
    @Cacheable._cacheable
    def bad_time_double_zoom_plots(self):
        """Get the extra-zoomed plots with full (i.e. non-trend) data for
        the bad time regions identified in the first level bad time zoom
        plots. This extra zoom level combined with the use of raw data
        makes is possible to distinguish true anomalies from trend-taking
        artifacts. These plots are hence the final step in checking data
        quality at seemingly anomalous times."""
        return sum([z.bad_time_zoom_plots for z in self.bad_time_zoom_plots],
                   [])
    def make_bad_time_zoom_plots(self):
        """Save all zoomed plots from bad times for this ``PlottingJob``. See
        ``PlottingJob.bad_time_zoom_plots`` for details."""
        #mapf = multiprocessing.Pool(processes=NUM_THREADS).map
        mapf = map
        mapf(_save_plot, self.bad_time_zoom_plots)
    def make_bad_time_double_zoom_plots(self):
        """Save all doubly zoomed plots from bad times for this
        ``PlottingJob``. See ``PlottingJob.bad_time_double_zoom_plots`` for
        details."""
        mapf = map
        mapf(_save_plot, self.bad_time_double_zoom_plots)

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
    PlotVars = IndividualPlotVars
    @fetch_data_first(False)
    def plot_timeseries(self, ax, **kwargs):
        """Scale up by 10^9 since plots are in ns, not seconds.
        Remove any indices considered bad in ``plot_properties``"""
        # define the variables for our plots
        y = np.delete(self.plot_vars.means - self.trend,
                      self.bad_indices.means) / SEC_PER['ns']
        t = np.delete(self.t_axis, self.bad_indices.means)
        yerr = np.delete(self.plot_vars.stds,
                         self.bad_indices.means) / SEC_PER['ns']
        mint = np.delete(self.t_axis, self.bad_indices.mins)
        miny = np.delete(self.plot_vars.mins - self.trend,
                         self.bad_indices.mins) / SEC_PER['ns']
        maxt = np.delete(self.t_axis, self.bad_indices.maxs)
        maxy = np.delete(self.plot_vars.maxs - self.trend,
                         self.bad_indices.maxs) / SEC_PER['ns']
        # plot everything, but only if the plotted data has nonzero length
        # in order to avoid an annoying matplotlib bug when adding legends.
        if len(t) != 0:
            ax.errorbar(t, y, marker="o", color="green", linestyle='none',
                        yerr=yerr, label="Means +/- Std. Dev.")
        if len(mint) != 0:
            ax.scatter(mint, miny, marker="^", color="blue", label="Minima")
        if len(maxt) != 0:
            ax.scatter(maxt, maxy, marker="v", color="red", label="Maxima")

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
    PlotVars = CombinedPlotVars
    @fetch_data_first(False)
    def plot_timeseries(self, ax, **kwargs):
        """Scale up by 10^9 since plots are in ns, not seconds.
        Remove any indices considered bad in ``plot_properties``"""
        # define the variables for our plots
        t = np.delete(self.t_axis, self.bad_indices.means)
        y = np.delete(self.plot_vars.means - self.trend,
                      self.bad_indices.means) / SEC_PER['ns']
        yerr = np.delete(self.plot_vars.stds,
                         self.bad_indices.means) / SEC_PER['ns']
        mint = np.delete(self.t_axis, self.bad_indices.absmins)
        miny = np.delete(self.plot_vars.absmins - self.trend,
                         self.bad_indices.absmins) / SEC_PER['ns']
        maxt = np.delete(self.t_axis, self.bad_indices.absmaxs)
        maxy = np.delete(self.plot_vars.absmaxs - self.trend,
                         self.bad_indices.absmaxs) / SEC_PER['ns']
        # plot everything, but only if the plotted data has nonzero length
        # in order to avoid an annoying matplotlib bug when adding legends.
        if len(t) != 0:
            ax.errorbar(t, y, marker="o", color="green", linestyle='none',
                        yerr=yerr, label="Means +/- Std. Dev.")
        if len(mint) != 0:
            ax.scatter(mint,miny,marker="^", color="blue", label="Abs. Minima")
        if len(maxt) != 0:
            ax.scatter(maxt,maxy,marker="v", color="red", label="Abs. Maxima")

class BadTimesZoomPlotter(FullDataPlotter):
    """A Plotter that makes zoomed in plots meant to be used for seeing
    missing data. To be more precise, this Plotter is used to show the full
    data from a segment that has missing data; the start and end should be
    the start and end of that data segment."""
    def __init__(self, start, end, channel, dq_flag, trends,
                 days_from_start, dq_segment, dq_segment_index, 
                 badness_type='bad',
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
        self.badness_type = badness_type
        self.trends = trends
        self.ext = ext
        self.run = run
        self.channel_description = channel_description
        self.height = height
        self.width = width
        self.plot_properties = DEFAULT_PLOT_PROPERTIES
        self.plot_properties.update(plot_properties)
        self.plot_properties['detrend'] = 'none'
    @property
    def title(self):
        """Get the title for this plot."""
        start = gwpy.time.from_gps(self.start)
        end = gwpy.time.from_gps(self.end)
        fmt = 'Fault: {} in {} in Segment no. {}\nFlag {} from {} to {}'
        if not self.run is None:
            fmt += ' during {}'.format(self.run)
        return fmt.format(self.badness_type, self.channel_description,
                          self.dq_segment_index, self.dq_flag, start, end)
    @property
    def fname(self):
        ch = self.queries[0].sanitized_channel
        if self.plot_properties['fname_desc'] is None:
            desc = ""
        else:
            desc = "." + self.plot_properties['fname_desc']
        fmt = '{}__{}__{}.{}.badtime.{}.ind__{}{}.{}'
        return fmt.format(self.start, self.end, ch, self.sanitized_dq_flag,
                          self.badness_type, self.dq_segment_index, desc,
                          DEFAULT_PLOT_FILETYPE)
    @property
    def t_lim(self):
        """Return a tuple containing the left and right t-limits for this plot.
        Same property as seen in ``Plotter``, but adds a bit of padding so
        that the first and last points are clearly visible."""
        tlim_left, tlim_right = Plotter.t_lim.fget(self)
        margin_factor = 3e-2
        t_span = tlim_right - tlim_left
        tlim_left -= margin_factor * t_span
        tlim_right += margin_factor * t_span
        return (tlim_left, tlim_right)
    @fetch_data_first(True)
    def plot_timeseries(self, ax, **kwargs):
        ax.plot(np.delete(self.t_axis, self.bad_indices.means),
                np.delete(self.plot_vars.means - self.trend,
                          self.bad_indices.means) / SEC_PER['ns'],
                marker="o", color="green", label="Recorded Signal")
        # put the start and/or end time in the plot as a vertical line
        unitfactor = SEC_PER[self.t_units]
        dq_start = (self.dq_segment.start.gpsSeconds - self.start) / unitfactor
        dq_end = (self.dq_segment.end.gpsSeconds - self.start) / unitfactor
        zorder = self.plot_properties['start_end_zorder']
        if self.t_lim[0] <= dq_start:
            deep_pink = '#FF1493'
            plot_vertical_marker(ax, [dq_start], zorder=zorder,
                                 label="Start of Segment", color=deep_pink)
        if dq_end <= self.t_lim[1]:
            midnight_blue = '#191970'
            plot_vertical_marker(ax, [dq_end], zorder=zorder,
                                 label="End of Segment", color=midnight_blue)
    @property
    def t_label(self):
        """Return the default t-axis label for a t-axis measuring days since
        the start of an observing run. ``Matplotlib`` refers to this as the
        x-axis (the horizontal one)."""
        t0 = gwpy.time.tconvert(self.start).strftime("%c")
        fmt = "Time Since Start of Fault during Segment {} at {} UTC [{}]"
        if not self.run is None:
            fmt += ' during {}'.format(self.run)
        return fmt.format(self.dq_segment_index, t0, self.t_units)
    @property
    def bad_time_zoom_plots(self):
        """Get a list of ``Plotters`` showing zoomed views of the messed up
        time spans for this channel whether they be missing, unrecorded,
        omitted, or outlier data). If this ``Plotter`` is using raw data, the
        list will be empty, since getting higher resolution data is impossible.
        If this ``Plotter`` is using some sort of trend data, though, the list
        of zoomed views will show contiguous timespans from this plotter that
        had bad values, only with the full data rather than minute trends."""
        zoomed_plots = list()
        if self.trends == ['']:
            return zoomed_plots
        all_bad_indices = self.bad_indices
        bad_index_types = self.bad_index_types
        t_axis = self.t_axis
        tlim = self.t_lim
        segs = self.dq_segments.active
        desc = self.channel_description
        # make sure that the way we define outliers does not change
        # for these zoomed plots; we don't want to mark spurious
        # outliers. For BadTimesZoomPlotter, these special surviving
        # plot properties will be folded into the defaults.
        plot_props = dict()
        for pp in ['outliers_lower_bound', 'outliers_upper_bound']:
            if self.plot_properties.has_key(pp):
                plot_props[pp] = self.plot_properties[pp]
        dq_segment = self.dq_segment
        dq_ind = self.dq_segment_index
        channel = self.channel
        trends = ['']
        days_from_start = self.days_from_start
        for pvg in self.PlotVars._plot_var_generators():
            varname = pvg.name
            if len(getattr(all_bad_indices, varname)) == 0:
                continue
            for badness_type in self.BadIndices._fields:
                bad_plot_vars = getattr(bad_index_types, badness_type)
                # only show zoomed plots for data within the plot window
                bad_inds = filter(lambda i: tlim[0] < t_axis[i] < tlim[1],
                                  getattr(bad_plot_vars, varname))
                bad_intervals = geco_gwpy_dump.indices_to_intervals(bad_inds)
                bad_time_intervals = self.plot_vars.times[bad_intervals]
                for i in range(len(bad_time_intervals) // 2):
                    start = bad_time_intervals[2*i]
                    end = bad_time_intervals[2*i + 1] + SEC_PER['minutes']
                    p = BadTimesZoomPlotter(start=start, end=end,
                                            channel=channel,
                                            dq_flag=self.dq_flag, trends=trends,
                                            days_from_start=days_from_start,
                                            badness_type=badness_type,
                                            dq_segment=dq_segment,
                                            dq_segment_index=dq_ind,
                                            ext=self.ext, run=self.run,
                                            channel_description=desc,
                                            height=self.height,
                                            width=self.width,
                                            plot_properties=plot_props)
                    zoomed_plots.append(p)
        return zoomed_plots

def main(args):
    plt_job = PlottingJob.load(args.jobspec)
    if args.faults:
        plt_job.fault_taxonomy()
        exit()
    for plot_type in args.plots:
        getattr(plt_job, CLI_PLOTTING_OPTIONS[plot_type])()

if __name__ == "__main__":
    main(args)
