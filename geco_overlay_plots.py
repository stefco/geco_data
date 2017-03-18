#!/usr/bin/env python
# (c) Stefan Countryman, 2017

import argparse
# ALL OTHER IMPORTS LISTED AFTER ARGUMENT PARSING; this allows for fast
# documentation access.

DESC = """Make IRIG-B and DuoTone overlay plots for a time interval surrounding
an event. These overlay plots are made by plotting one second worth of data
at a time on the same axes. For periodic signals (or quasi-periodic signals
with recurring features), the resulting plot should be (almost)
indistinguishible from a plot of a single second worth of data, provided the
assumption of (quasi) periodicity holds in the actual signal. This provides a
quick visual check of the quality of the signal."""
FAST_BITRATE = 16384      # bitrate for fast channels, like ADC DuoTone & IRIG-B
DEFAULT_DELTA_T = (30*60) # 30 minutes
OVERLAY_DPI = 400         # overlay plots need higher resolution to show detail
# OVERLAY_PLOT_WIDTH = 24
# OVERLAY_PLOT_HEIGHT_PER_ZOOM = 4
# OVERLAY_PLOT_TITLE_HEIGHT = 2.4
DEFAULT_OFFSET = DEFAULT_DELTA_T // 2
DEFAULT_MAX_FETCH_SIZE = 60 # max number of seconds to fetch from NDS2 at once
DUOTONE_START = -2        # data points before start of second to include?
DUOTONE_END   = 6         # data points after end of second to include?
ZOOMS = {
    'duotone': 10,        # how many zoomed in subplots for DuoTone overlays?
    'irigb':   5          # how many zoomed in subplots for IRIG-B overlays?
}
# OVERLAY_PLOT_HEIGHTS = {
#     'duotone': (  OVERLAY_PLOT_HEIGHT_PER_ZOOM * ZOOMS['duotone']
#                 + OVERLAY_PLOT_TITLE_HEIGHT),
#     'irigb':   (  OVERLAY_PLOT_HEIGHT_PER_ZOOM * ZOOMS['irigb']
#                 + OVERLAY_PLOT_TITLE_HEIGHT)
# }

# quits immediately on --help or -h flags to avoid slow imports
if __name__ == "__main__":
    """Define an argument parser with default values for this script."""
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("-t", "--gpstime", type=float,
                        help="The GPS time of this event.")
    parser.add_argument("-d", "--deltat", type=int, default=DEFAULT_DELTA_T,
                        help=("How large of a time window to make plots for, "
                              "in seconds. Defaults to "
                              "{}").format(DEFAULT_DELTA_T))
    parser.add_argument("-o", "--offset", type=int, default=DEFAULT_OFFSET,
                        help=("How far to offset the start of this plot from "
                              "the --gpstime. Defaults to "
                              "{}").format(DEFAULT_OFFSET))
    parser.add_argument("-p", "--plots", choices=('irigb', 'duotone'),
                        nargs='*', default=['irigb', 'duotone'],
                        help="Which plots to make?")
    parser.add_argument("-i", "--ifos", choices=('H1', 'L1'),
                        nargs='*', default=['H1', 'L1'],
                        help="Which detectors to make plots for?")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="If provided, print verbose progress information.")
    args = parser.parse_args()

import gwpy.timeseries
import geco_irig_plot
import matplotlib
matplotlib.use('Agg') # necessary for plotting on headless machines
import matplotlib.pyplot as plt
import numpy as np
import abc

class SecondStats(object):
    """Keep track of some sort of data on a per-second basis. This way you
    can accumulate these statistics after downloading many seconds of data,
    and then use the final result without having to have ever stored the full
    data set on which the stats are based. Very naive, with no checking for
    redundancy, so make sure not to use the same times twice."""
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def __init__(self, channel, timeseries=None, stats=None, n=None):
        """Take a second of data and calculate some statistics for it. Can be
        initialized with custom data by omitting the timeseries argument and
        providing custom statistics and sample number."""
    @abc.abstractmethod
    def __add__(self, other):
        """Add two SecondStat objects together. Equivalent to generating
        multiple seconds worth of statistics. Allows you to reduce multiple
        SecondStats objects into a single instance."""
    @abc.abstractmethod
    def stat_plots(self, start_time, end_time):
        """Do any plotting that is related to the statistics calculated."""

class IRIGBSecondStats(SecondStats):
    """Keep track of a second of IRIG-B statistics. Not currently used."""
    def __init__(self, channel, timeseries=None, stats=None, n=None): pass
    def __add__(self, other): return self
    def stat_plots(self, start_time, end_time): pass

class DuoToneSecondStats(SecondStats):
    """Keep track of a second of DuoTone statistics. Note that this looks at the
    start and end of each second, so if you reduce a bunch of contiguous
    seconds' stats, you have the end of the second before the first second and
    the start of the second after the final second included in your stats. This
    could only possibly matter in edge cases, since DuoTone signals are expected
    to be perfectly periodic, but in case you see weird behavior in the
    zero-crossing plots, it's worth keeping this in mind as a possible
    culprit."""
    def __init__(self, channel, timeseries=None, stats=None, n=None):
        """If timeseries is None, then we are manually initializing with custom
        stat values. Otherwise, calculate stat values from the timeseries.
        Input is assumed to be a GWpy timeseries."""
        self.channel = channel
        if timeseries is None:
            self.n = n
            self.stats = stats
        else:
            self.n = 1
            # wrap the end of the second onto the start of that same second.
            # obviously this is only okay with quasi periodic signals!
            zero_crossing = np.concatenate((timeseries.value[DUOTONE_START:],
                                            timeseries.value[:DUOTONE_END]))
            self.stats = {
                "sum": zero_crossing,
                "sum_sq": np.square(zero_crossing),
                "mean": zero_crossing,
                "sigma": zero_crossing * 0., # i.e. zeros
                "max": zero_crossing,
                "min": zero_crossing
            }
    def __add__(self, other):
        n = self.n + other.n
        sum = self.stats['sum'] + other.stats['sum']
        sum_sq = self.stats['sum_sq'] + other.stats['sum_sq']
        stats = {
            "sum":     sum,
            "sum_sq":  sum_sq,
            "mean":    sum / n,
            "sigma":   np.sqrt((sum_sq / n) - np.square(sum / n)),
            "max":     np.maximum(self.stats['max'], other.stats['max']),
            "min":     np.minimum(self.stats['max'], other.stats['max'])
        }
        return type(self)(channel=self.channel, stats=stats, n=n)
    def stat_plots(self, start_time, end_time):
        """Make 3 plots with different levels of zoom and save them as PNG
        files."""
        t = np.arange(DUOTONE_START, DUOTONE_END, dtype=float) / FAST_BITRATE
        min_delta = self.stats['mean'] - self.stats['min']
        max_delta = self.stats['max']  - self.stats['mean']
        title = ('Mean DuoTone Signal, {}, near Zero Crossing\n'
                 'with Max/Min Values and Std. Dev.').format(self.channel)
        xlabel = r'Time since start of second ($\mu s$)'
        fname_fmt_fmt = '{}-DuoTone-Statistics-{}-from-{}-to-{}.png'
        fname_fmt = fname_fmt_fmt.format(self.channel.replace(':', '..'),
                                         '{}', start_time, end_time)
        # first figure, full view
        plt.close(1)
        plt.figure(1)
        plt.plot(t, self.stats['mean'], 'k')
        plt.errorbar(t, self.stats['mean'], yerr=[min_delta, max_delta],
                     fmt='k')
        plt.errorbar(t, self.stats['mean'], yerr=self.stats['sigma'],
                     fmt='go', linewidth=4)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.grid(b=True, which='major', color='#262626', linestyle='--')
        plt.savefig(fname_fmt.format('full'))
        # second file, zoomed on zero-crossing best fit line
        plt.xlim(49.5, 52.)
        plt.ylim(-50, 50)
        plt.title(title + ' at zero crossing')
        plt.savefig(fname_fmt.format('zero-crossing-zoom'))
        # third file, zoomed on data point nearest zero-crossing
        plt.title(title + ', Error Bars Emphasized')
        plt.xlim(61, 61.1)
        plt.ylim(250, 260)
        plt.savefig(fname_fmt.format('super-zoom'))

def get_second_stats_class(plottype):
    """Return the appropriate second statistics class for this plot type.
    Use this to reduce data that was collected in parallel."""
    if plottype == "irigb":
        return IRIGBSecondStats
    elif plottype == "duotone":
        return DuoToneSecondStats

def get_channels(ifo, plottype):
    """Get a list of channels to plot for a given IFO. Plot Type must be
    either 'irigb' or 'duotone'."""
    if plottype == "irigb":
        return ['{}:CAL-PCAL{}_IRIGB_OUT_DQ'.format(ifo, arm)
                for arm in ['X', 'Y']]
    elif plottype == "duotone":
        return ['{}:CAL-PCAL{}_FPGA_DTONE_IN1_DQ'.format(ifo, arm)
                for arm in ['X', 'Y']]
    else:
        raise ValueError("Must specify 'irigb' or 'duotone' for plottype.")

def make_plots(gps, ifo, plottype, deltat=DEFAULT_DELTA_T,
               offset=DEFAULT_OFFSET, max_fetch_size=DEFAULT_MAX_FETCH_SIZE,
               verbose=False):
    """Make the necessary plots for this plottype."""
    gps_start = int(gps) + int(offset)
    gps_end = int(gps_start) + int(deltat)
    chans = get_channels(ifo, plottype)
    n_subdivs = ZOOMS[plottype]
    StatClass = get_second_stats_class(plottype)
    stats_instances = {}
    # we will fetch a chunk of data at a time
    time_intervals = range(gps_start, gps_end, int(max_fetch_size)) + [gps_end]
    if verbose:
        print('gps_start: {}'.format(gps_start))
        print('gps_end: {}'.format(gps_end))
        print('chans: {}'.format(chans))
        print('n_subdivs: {}'.format(n_subdivs))
        print('StatClass: {}'.format(StatClass))
        print('stats_instances: {}'.format(stats_instances))
        print('time_intervals: {}'.format(time_intervals))
    plt.close('all')

    # download data in chunks
    for i in range(len(time_intervals) - 1):
        if verbose:
            msg_fmt = 'fetching {} bufs from {} to {} ({}/{})...'
            print(msg_fmt.format(chans, time_intervals[i], time_intervals[i+1],
                                 i+1, len(time_intervals)-1))
        bufs = gwpy.timeseries.TimeSeriesDict.fetch(chans, time_intervals[i],
                                                    time_intervals[i+1],
                                                    verbose=verbose)
        # process 1 second at a time from each chunk
        for j in range(time_intervals[i+1] - time_intervals[i]):
            gpst = time_intervals[i] + j
            timeseries = [bufs[c][ FAST_BITRATE*j:FAST_BITRATE*(j+1) ]
                          for c in chans]
            titles = ['Overlay of {}\n from {} to {}'.format(c, gpst, gpst+1)
                      for c in chans]
            # make overlay plots for each channel and accumulate statistics
            for k in range(len(chans)):
                plt.figure(k+1)
                geco_irig_plot.plot_with_zoomed_views(timeseries[k], titles[k],
                                                      num_subdivs=n_subdivs,
                                                      overlay=True,
                                                      linewidth=0.2)
                # initialize the statistics instance if not already initialized
                try:
                    stats_instances[chans[k]] += StatClass(chans[k],
                                                           timeseries[k])
                except KeyError:
                    stats_instances[chans[k]]  = StatClass(chans[k],
                                                           timeseries[k])

    # save overlay plots
    for ii in range(len(chans)):
        plt.figure(ii+1)
        fname_fmt='{}-Overlay-{}-to-{}.png'
        if verbose:
            print(('saving {} OVERLAY plots for {} from '
                   '{} to {}.').format(plottype, c, gps_start, gps_end))
        # plt.gcf().set_figwidth(OVERLAY_PLOT_WIDTH)
        # plt.gcf().set_figheight(OVERLAY_PLOT_HEIGHTS[plottype])
        plt.savefig(fname_fmt.format(chans[ii].replace(':', '..'),
                                     gps_start, gps_end),
                    dpi=OVERLAY_DPI)
        if verbose:
            print('done making OVERLAY plots.')
    plt.close('all')
    # save statistics plots
    for c in chans:
        if verbose:
            print(('making {} STATS plots for {} from '
                   '{} to {}.').format(plottype, c, gps_start, gps_end))
        stats_instances[c].stat_plots(gps_start, gps_end)
        if verbose:
            print('done making STATS plots.')

if __name__ == "__main__":
    for plottype in args.plots:   # parser defined at top if in __main__
        for ifo in args.ifos:
            make_plots(args.gpstime, ifo, plottype, args.deltat, args.offset,
                       verbose=args.verbose)
