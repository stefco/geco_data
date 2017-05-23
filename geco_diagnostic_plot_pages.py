#!/usr/bin/env python
# (c) Stefan Countryman 2017

DESC="""Plot a list of timing diagnostic channels, which will be read from
stdin as a newline-delimited channel list, for a time window around a given
GPS time. This script can also generate a summary webpage for easy viewing of
plot results. Use this script when there is a timing error to make plots on a
bunch of timing channels to quickly find the source of the problem."""
DT = 30
MAX_SIMULTANEOUS_CHANS = 5

# THE REST OF THE IMPORTS ARE AFTER THIS IF STATEMENT.
# Quits immediately on --help or -h flags to skip slow imports when you just
# want to read the help documentation.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("-t", "--gpstime", type=int,
                        help=("The GPS time about which the plots should be"
                              "centered."))
    parser.add_argument("-o", "--outdir", default=".",
                        help=("Where should the files be saved? Defaults to "
                              "the current directory."))
    parser.add_argument("-l", "--channellist",
                        help=("Path to a text file containing the list of"
                              "channels to plot. If not provided, this script"
                              "will try to read the channel list in from STDIN."
                              "The channel list should be a newline-delimited"
                              "list of valid EPICS channel names, though"
                              "badly-formed or invalid channel names will be"
                              "skipped."))
    parser.add_argument("-d", "--deltat", type=int, default=DT,
                        help=("The size of the time window to be plotted. The"
                              "generated plots will show +/- deltat seconds of"
                              "data around the central GPS time. Defaults"
                              "to: {}").format(DT))
    parser.add_argument("-w", "--skipwebpage", action='store_true',
                        help=("If this flag is provided, no preview webpage"
                              "will be generated. By default, the webpage is"
                              "generated."))
    parser.add_argument("-p", "--skipplots", action='store_true',
                        help=("If this flag is provided, no plots will be"
                              "generated (useful if you already made plots and"
                              "just want to regenerate the webpage). By"
                              "default, the plots are generated."))
    parser.add_argument("-s", "--maxsimultaneous", type=int,
                        default=MAX_SIMULTANEOUS_CHANS,
                        help=("The maximum number of simultaneous channels for"
                              "which to fetch data at a given time. It is"
                              "generally faster to fetch a batch of channels"
                              "at a time, but getting too many channels at once"
                              "might slow things down. Defaults to: "
                              "{}").format(MAX_SIMULTANEOUS_CHANS))
    parser.add_argument("-v", "--verbose", action='store_true',
                        help=("Print verbose output for status monitoring"
                              "while plotting. Defaults to False."))
    args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')
import gwpy.timeseries
import sys
import os


def channel_fname(gps, chan):
    return '{}_{}.png'.format(gps, chan.replace(':', '..'))

def image_link_html(gps, chan):
    """return an HTML string for an image element for this plot"""
    return ('<div>\n'
            '    <p>{}</p>\n'
            '    <br>\n'
            '    <img src="./{}">\n'
            '</div>\n').format(chan, channel_fname(gps, chan))

def read_channels(filedescriptor):
    return list(set(filedescriptor.read().split('\n')) - {''})

def get_channel_list():
    """Get the channel list from the file descriptor specified via the CLI."""
    if args.channellist is None:
        return read_channels(sys.stdin)
    else:
        with open(args.channellist, 'r') as f:
            return read_channels(f)

def make_preview_webpate(outdir, chans, gps):
    """Make a webpage called index.html displaying all generated plots and save
    it to the specified output directory."""
    HTML_TOP = """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Plots around {}</title>
            <style type="text/css">
                img, p {{
                    max-width: 400px;
                }}
                div {{
                    display: inline
                }}
            </style>
        </head>
        <body>
    """.replace('\n    ', '\n').format(gps) # replace will remove 1st indent
    HTML_BOTTOM = """
        </body>
    </html>
    """.replace('\n    ', '\n') # replace will remove 1st indent
    filepath = os.path.join(outdir, 'index.html')
    with open(filepath, 'w') as f:
        f.write(HTML_TOP)
        f.write('\n')
        for chan in chans:
            f.write(image_link_html(gps, chan))
        f.write(HTML_BOTTOM)

def make_channel_plots(outdir, chans, gps, dt=DT, verbose=False,
                       max_simultaneous_chans=MAX_SIMULTANEOUS_CHANS):
    for i in range(0, len(chans), max_simultaneous_chans):
        if verbose:
            print('plotting {} thru {} of {}'.format(i,
                                                     i+max_simultaneous_chans,
                                                     len(chans)))
        chans_sublist = chans[i:i+max_simultaneous_chans]
        try:
            bufs = gwpy.timeseries.TimeSeriesDict.fetch(chans_sublist, gps-dt,
                                                        gps+dt,
                                                        verbose=verbose)
        except RuntimeError:
            if verbose:
                print('Could not fetch all at once:\n{}'.format(chans_sublist))
            bufs = {}
            for chan in chans_sublist:
                try:
                    buf = gwpy.timeseries.TimeSeries.fetch(chan, gps-dt, gps+dt,
                                                        verbose=verbose)
                    bufs[chan] = buf
                except RuntimeError:
                    if verbose: print('Bad channel: {}'.format(chan))
        for chan in bufs:
            buf = bufs[chan]
            filepath = os.path.join(outdir, channel_fname(gps, chan))
            plot = buf.plot()
            plot.set_title(buf.channel.name.replace('_', '\_'))
            plot.savefig(filepath)

if __name__ == '__main__':
    chans = get_channel_list()
    if args.verbose: print('channels to plot: {}'.format(chans))

    if not args.skipwebpage:
        make_preview_webpate(args.outdir, chans, args.gpstime)

    if not args.skipplots:
        make_channel_plots(args.outdir, chans, args.gpstime, args.deltat,
                           args.verbose, args.maxsimultaneous)
