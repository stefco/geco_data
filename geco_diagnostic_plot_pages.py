#!/usr/bin/env python
# (c) Stefan Countryman 2017

DESC="""Plot a list of timing diagnostic channels, which will be read from
stdin as a newline-delimited channel list, for a time window around a given
GPS time. This script can also generate a summary webpage for easy viewing of
plot results. Use this script when there is a timing error to make plots on a
bunch of timing channels to quickly find the source of the problem."""
DT = 30
MAX_SIMULTANEOUS_CHANS = 5
# channels to use if there are none specified via CLI
DEFAULT_CHANNELS=[
    "L1:SYS-TIMING_C_GPS_A_ERROR_FLAG",
    "L1:SYS-TIMING_C_GPS_A_ERROR_FLAG",
    "L1:SYS-TIMING_C_GPS_A_LATITUDE",
    "L1:SYS-TIMING_C_GPS_A_LONGITUDE",
    "L1:SYS-TIMING_C_GPS_A_ALTITUDE",
    "L1:SYS-TIMING_C_GPS_A_SURVEYPROGRESS",
    "L1:SYS-TIMING_C_GPS_A_HOLDOVERDURATION",
    "L1:SYS-TIMING_C_GPS_A_DACVOLTAGE",
    "L1:SYS-TIMING_C_GPS_A_TEMPERATURE",
    "L1:SYS-TIMING_C_GPS_A_RECEIVERMODE",
    "L1:SYS-TIMING_C_GPS_A_DECODINGSTATUS",
    "L1:SYS-TIMING_C_GPS_A_TIMEVALID",
    "L1:SYS-TIMING_C_GPS_A_UTCOFFSET",
    "L1:SYS-TIMING_C_GPS_A_TIMESOURCE",
    "L1:SYS-TIMING_C_GPS_A_PPSSOURCE",
    "L1:SYS-TIMING_C_GPS_A_ALMANACINCOMPLETE",
    "L1:SYS-TIMING_C_GPS_A_NOTTRACKINGSATTELITES",
    "L1:SYS-TIMING_C_GPS_A_SURVEYINPROGRESS",
    "L1:SYS-TIMING_C_GPS_A_GPS",
    "L1:SYS-TIMING_C_GPS_A_YEAR",
    "L1:SYS-TIMING_C_GPS_A_MONTH",
    "L1:SYS-TIMING_C_GPS_A_DAY",
    "L1:SYS-TIMING_C_GPS_A_HOUR",
    "L1:SYS-TIMING_C_GPS_A_MINUTE",
    "L1:SYS-TIMING_C_GPS_A_SECOND",
    "L1:SYS-TIMING_C_GPS_A_LEAP",
    "L1:SYS-TIMING_C_GPS_A_LEAPSECONDPENDING",
    "L1:SYS-TIMING_C_GPS_A_WEEK",
    "L1:SYS-TIMING_C_GPS_A_TOW",
    "L1:SYS-TIMING_C_GPS_A_PPSOFFSET",
    "L1:SYS-TIMING_C_GPS_A_DISCIPLININGMODE",
    "L1:SYS-TIMING_C_GPS_A_DISCIPLININGACTIVITY",
    "L1:SYS-TIMING_C_GPS_A_DACATRAIL",
    "L1:SYS-TIMING_C_GPS_A_DACNEARRAIL",
    "L1:SYS-TIMING_C_GPS_A_NOSTOREDPOSITION",
    "L1:SYS-TIMING_C_GPS_A_POSITIONQUESTIONABLE",
    "L1:SYS-TIMING_C_GPS_A_NOPPS",
    "L1:SYS-TIMING_C_GPS_A_ANTENNAOPEN",
    "L1:SYS-TIMING_C_GPS_A_ANTENNASHORTED",
    "L1:SYS-TIMING_C_GPS_A_ERROR_FLAG",
    "L1:SYS-TIMING_C_GPS_A_ERROR_CODE",

    "L1:SYS-TIMING_X_GPS_A_ERROR_FLAG",
    "L1:SYS-TIMING_X_GPS_A_LATITUDE",
    "L1:SYS-TIMING_X_GPS_A_LONGITUDE",
    "L1:SYS-TIMING_X_GPS_A_ALTITUDE",
    "L1:SYS-TIMING_X_GPS_A_SPEED3D",
    "L1:SYS-TIMING_X_GPS_A_SPEED2D",
    "L1:SYS-TIMING_X_GPS_A_HEADING",
    "L1:SYS-TIMING_X_GPS_A_DOP",
    "L1:SYS-TIMING_X_GPS_A_VISSATELLITES",
    "L1:SYS-TIMING_X_GPS_A_TRACKSATELLITES",
    "L1:SYS-TIMING_X_GPS_A_TIMEVALID",
    "L1:SYS-TIMING_X_GPS_A_RECEIVERMODE",
    "L1:SYS-TIMING_X_GPS_A_UTCOFFSET",
    "L1:SYS-TIMING_X_GPS_A_TIMESOURCE",
    "L1:SYS-TIMING_X_GPS_A_ALMANACINCOMPLETE",
    "L1:SYS-TIMING_X_GPS_A_GPS",
    "L1:SYS-TIMING_X_GPS_A_YEAR",
    "L1:SYS-TIMING_X_GPS_A_MONTH",
    "L1:SYS-TIMING_X_GPS_A_DAY",
    "L1:SYS-TIMING_X_GPS_A_HOUR",
    "L1:SYS-TIMING_X_GPS_A_MINUTE",
    "L1:SYS-TIMING_X_GPS_A_SECOND",
    "L1:SYS-TIMING_X_GPS_A_LEAP",
    "L1:SYS-TIMING_X_GPS_A_WEEK",
    "L1:SYS-TIMING_X_GPS_A_TOW",
    "L1:SYS-TIMING_X_GPS_A_NOTTRACKINGSATTELITES",
    "L1:SYS-TIMING_X_GPS_A_SURVEYINPROGRESS",
    "L1:SYS-TIMING_X_GPS_A_ANTENNAOPEN",
    "L1:SYS-TIMING_X_GPS_A_ANTENNASHORTED",
    "L1:SYS-TIMING_X_GPS_A_NARROWBAND",
    "L1:SYS-TIMING_X_GPS_A_FASTACQUISITION",
    "L1:SYS-TIMING_X_GPS_A_FILTERRESET",
    "L1:SYS-TIMING_X_GPS_A_POSITIONLOCK",
    "L1:SYS-TIMING_X_GPS_A_DIFFERENTIALFIX",

    "L1:SYS-TIMING_Y_GPS_A_ERROR_FLAG",
    "L1:SYS-TIMING_Y_GPS_A_LATITUDE",
    "L1:SYS-TIMING_Y_GPS_A_LONGITUDE",
    "L1:SYS-TIMING_Y_GPS_A_ALTITUDE",
    "L1:SYS-TIMING_Y_GPS_A_SPEED3D",
    "L1:SYS-TIMING_Y_GPS_A_SPEED2D",
    "L1:SYS-TIMING_Y_GPS_A_HEADING",
    "L1:SYS-TIMING_Y_GPS_A_DOP",
    "L1:SYS-TIMING_Y_GPS_A_VISSATELLITES",
    "L1:SYS-TIMING_Y_GPS_A_TRACKSATELLITES",
    "L1:SYS-TIMING_Y_GPS_A_TIMEVALID",
    "L1:SYS-TIMING_Y_GPS_A_RECEIVERMODE",
    "L1:SYS-TIMING_Y_GPS_A_UTCOFFSET",
    "L1:SYS-TIMING_Y_GPS_A_TIMESOURCE",
    "L1:SYS-TIMING_Y_GPS_A_ALMANACINCOMPLETE",
    "L1:SYS-TIMING_Y_GPS_A_GPS",
    "L1:SYS-TIMING_Y_GPS_A_YEAR",
    "L1:SYS-TIMING_Y_GPS_A_MONTH",
    "L1:SYS-TIMING_Y_GPS_A_DAY",
    "L1:SYS-TIMING_Y_GPS_A_HOUR",
    "L1:SYS-TIMING_Y_GPS_A_MINUTE",
    "L1:SYS-TIMING_Y_GPS_A_SECOND",
    "L1:SYS-TIMING_Y_GPS_A_LEAP",
    "L1:SYS-TIMING_Y_GPS_A_WEEK",
    "L1:SYS-TIMING_Y_GPS_A_TOW",
    "L1:SYS-TIMING_Y_GPS_A_NOTTRACKINGSATTELITES",
    "L1:SYS-TIMING_Y_GPS_A_SURVEYINPROGRESS",
    "L1:SYS-TIMING_Y_GPS_A_ANTENNAOPEN",
    "L1:SYS-TIMING_Y_GPS_A_ANTENNASHORTED",
    "L1:SYS-TIMING_Y_GPS_A_NARROWBAND",
    "L1:SYS-TIMING_Y_GPS_A_FASTACQUISITION",
    "L1:SYS-TIMING_Y_GPS_A_FILTERRESET",
    "L1:SYS-TIMING_Y_GPS_A_POSITIONLOCK",
    "L1:SYS-TIMING_Y_GPS_A_DIFFERENTIALFIX",

    "L1:SYS-TIMING_C_PPS_B_SIGNAL_0_DIFF",
    "L1:SYS-TIMING_C_PPS_B_SIGNAL_1_DIFF",
    "L1:SYS-TIMING_C_PPS_B_SIGNAL_2_DIFF",
    "L1:SYS-TIMING_C_PPS_B_SIGNAL_3_DIFF",
    "L1:SYS-TIMING_C_PPS_B_SIGNAL_4_DIFF",
    "L1:SYS-TIMING_C_PPS_B_SIGNAL_5_DIFF",
    "L1:SYS-TIMING_C_PPS_B_SIGNAL_6_DIFF",
    "L1:SYS-TIMING_C_PPS_B_SIGNAL_7_DIFF",

    "L1:SYS-TIMING_X_PPS_A_SIGNAL_0_DIFF",
    "L1:SYS-TIMING_X_PPS_A_SIGNAL_1_DIFF",
    "L1:SYS-TIMING_X_PPS_A_SIGNAL_2_DIFF",
    "L1:SYS-TIMING_X_PPS_A_SIGNAL_3_DIFF",
    "L1:SYS-TIMING_X_PPS_A_SIGNAL_4_DIFF",
    "L1:SYS-TIMING_X_PPS_A_SIGNAL_5_DIFF",
    "L1:SYS-TIMING_X_PPS_A_SIGNAL_6_DIFF",
    "L1:SYS-TIMING_X_PPS_A_SIGNAL_7_DIFF",

    "L1:SYS-TIMING_Y_PPS_A_SIGNAL_0_DIFF",
    "L1:SYS-TIMING_Y_PPS_A_SIGNAL_1_DIFF",
    "L1:SYS-TIMING_Y_PPS_A_SIGNAL_2_DIFF",
    "L1:SYS-TIMING_Y_PPS_A_SIGNAL_3_DIFF",
    "L1:SYS-TIMING_Y_PPS_A_SIGNAL_4_DIFF",
    "L1:SYS-TIMING_Y_PPS_A_SIGNAL_5_DIFF",
    "L1:SYS-TIMING_Y_PPS_A_SIGNAL_6_DIFF",
    "L1:SYS-TIMING_Y_PPS_A_SIGNAL_7_DIFF",

    "H1:SYS-TIMING_C_GPS_A_ERROR_FLAG",
    "H1:SYS-TIMING_C_GPS_A_ERROR_FLAG",
    "H1:SYS-TIMING_C_GPS_A_LATITUDE",
    "H1:SYS-TIMING_C_GPS_A_LONGITUDE",
    "H1:SYS-TIMING_C_GPS_A_ALTITUDE",
    "H1:SYS-TIMING_C_GPS_A_SURVEYPROGRESS",
    "H1:SYS-TIMING_C_GPS_A_HOLDOVERDURATION",
    "H1:SYS-TIMING_C_GPS_A_DACVOLTAGE",
    "H1:SYS-TIMING_C_GPS_A_TEMPERATURE",
    "H1:SYS-TIMING_C_GPS_A_RECEIVERMODE",
    "H1:SYS-TIMING_C_GPS_A_DECODINGSTATUS",
    "H1:SYS-TIMING_C_GPS_A_TIMEVALID",
    "H1:SYS-TIMING_C_GPS_A_UTCOFFSET",
    "H1:SYS-TIMING_C_GPS_A_TIMESOURCE",
    "H1:SYS-TIMING_C_GPS_A_PPSSOURCE",
    "H1:SYS-TIMING_C_GPS_A_ALMANACINCOMPLETE",
    "H1:SYS-TIMING_C_GPS_A_NOTTRACKINGSATTELITES",
    "H1:SYS-TIMING_C_GPS_A_SURVEYINPROGRESS",
    "H1:SYS-TIMING_C_GPS_A_GPS",
    "H1:SYS-TIMING_C_GPS_A_YEAR",
    "H1:SYS-TIMING_C_GPS_A_MONTH",
    "H1:SYS-TIMING_C_GPS_A_DAY",
    "H1:SYS-TIMING_C_GPS_A_HOUR",
    "H1:SYS-TIMING_C_GPS_A_MINUTE",
    "H1:SYS-TIMING_C_GPS_A_SECOND",
    "H1:SYS-TIMING_C_GPS_A_LEAP",
    "H1:SYS-TIMING_C_GPS_A_LEAPSECONDPENDING",
    "H1:SYS-TIMING_C_GPS_A_WEEK",
    "H1:SYS-TIMING_C_GPS_A_TOW",
    "H1:SYS-TIMING_C_GPS_A_PPSOFFSET",
    "H1:SYS-TIMING_C_GPS_A_DISCIPLININGMODE",
    "H1:SYS-TIMING_C_GPS_A_DISCIPLININGACTIVITY",
    "H1:SYS-TIMING_C_GPS_A_DACATRAIL",
    "H1:SYS-TIMING_C_GPS_A_DACNEARRAIL",
    "H1:SYS-TIMING_C_GPS_A_NOSTOREDPOSITION",
    "H1:SYS-TIMING_C_GPS_A_POSITIONQUESTIONABLE",
    "H1:SYS-TIMING_C_GPS_A_NOPPS",
    "H1:SYS-TIMING_C_GPS_A_ANTENNAOPEN",
    "H1:SYS-TIMING_C_GPS_A_ANTENNASHORTED",
    "H1:SYS-TIMING_C_GPS_A_ERROR_FLAG",
    "H1:SYS-TIMING_C_GPS_A_ERROR_CODE",

    "H1:SYS-TIMING_X_GPS_A_ERROR_FLAG",
    "H1:SYS-TIMING_X_GPS_A_LATITUDE",
    "H1:SYS-TIMING_X_GPS_A_LONGITUDE",
    "H1:SYS-TIMING_X_GPS_A_ALTITUDE",
    "H1:SYS-TIMING_X_GPS_A_SPEED3D",
    "H1:SYS-TIMING_X_GPS_A_SPEED2D",
    "H1:SYS-TIMING_X_GPS_A_HEADING",
    "H1:SYS-TIMING_X_GPS_A_DOP",
    "H1:SYS-TIMING_X_GPS_A_VISSATELLITES",
    "H1:SYS-TIMING_X_GPS_A_TRACKSATELLITES",
    "H1:SYS-TIMING_X_GPS_A_TIMEVALID",
    "H1:SYS-TIMING_X_GPS_A_RECEIVERMODE",
    "H1:SYS-TIMING_X_GPS_A_UTCOFFSET",
    "H1:SYS-TIMING_X_GPS_A_TIMESOURCE",
    "H1:SYS-TIMING_X_GPS_A_ALMANACINCOMPLETE",
    "H1:SYS-TIMING_X_GPS_A_GPS",
    "H1:SYS-TIMING_X_GPS_A_YEAR",
    "H1:SYS-TIMING_X_GPS_A_MONTH",
    "H1:SYS-TIMING_X_GPS_A_DAY",
    "H1:SYS-TIMING_X_GPS_A_HOUR",
    "H1:SYS-TIMING_X_GPS_A_MINUTE",
    "H1:SYS-TIMING_X_GPS_A_SECOND",
    "H1:SYS-TIMING_X_GPS_A_LEAP",
    "H1:SYS-TIMING_X_GPS_A_WEEK",
    "H1:SYS-TIMING_X_GPS_A_TOW",
    "H1:SYS-TIMING_X_GPS_A_NOTTRACKINGSATTELITES",
    "H1:SYS-TIMING_X_GPS_A_SURVEYINPROGRESS",
    "H1:SYS-TIMING_X_GPS_A_ANTENNAOPEN",
    "H1:SYS-TIMING_X_GPS_A_ANTENNASHORTED",
    "H1:SYS-TIMING_X_GPS_A_NARROWBAND",
    "H1:SYS-TIMING_X_GPS_A_FASTACQUISITION",
    "H1:SYS-TIMING_X_GPS_A_FILTERRESET",
    "H1:SYS-TIMING_X_GPS_A_POSITIONLOCK",
    "H1:SYS-TIMING_X_GPS_A_DIFFERENTIALFIX",

    "H1:SYS-TIMING_Y_GPS_A_ERROR_FLAG",
    "H1:SYS-TIMING_Y_GPS_A_LATITUDE",
    "H1:SYS-TIMING_Y_GPS_A_LONGITUDE",
    "H1:SYS-TIMING_Y_GPS_A_ALTITUDE",
    "H1:SYS-TIMING_Y_GPS_A_SPEED3D",
    "H1:SYS-TIMING_Y_GPS_A_SPEED2D",
    "H1:SYS-TIMING_Y_GPS_A_HEADING",
    "H1:SYS-TIMING_Y_GPS_A_DOP",
    "H1:SYS-TIMING_Y_GPS_A_VISSATELLITES",
    "H1:SYS-TIMING_Y_GPS_A_TRACKSATELLITES",
    "H1:SYS-TIMING_Y_GPS_A_TIMEVALID",
    "H1:SYS-TIMING_Y_GPS_A_RECEIVERMODE",
    "H1:SYS-TIMING_Y_GPS_A_UTCOFFSET",
    "H1:SYS-TIMING_Y_GPS_A_TIMESOURCE",
    "H1:SYS-TIMING_Y_GPS_A_ALMANACINCOMPLETE",
    "H1:SYS-TIMING_Y_GPS_A_GPS",
    "H1:SYS-TIMING_Y_GPS_A_YEAR",
    "H1:SYS-TIMING_Y_GPS_A_MONTH",
    "H1:SYS-TIMING_Y_GPS_A_DAY",
    "H1:SYS-TIMING_Y_GPS_A_HOUR",
    "H1:SYS-TIMING_Y_GPS_A_MINUTE",
    "H1:SYS-TIMING_Y_GPS_A_SECOND",
    "H1:SYS-TIMING_Y_GPS_A_LEAP",
    "H1:SYS-TIMING_Y_GPS_A_WEEK",
    "H1:SYS-TIMING_Y_GPS_A_TOW",
    "H1:SYS-TIMING_Y_GPS_A_NOTTRACKINGSATTELITES",
    "H1:SYS-TIMING_Y_GPS_A_SURVEYINPROGRESS",
    "H1:SYS-TIMING_Y_GPS_A_ANTENNAOPEN",
    "H1:SYS-TIMING_Y_GPS_A_ANTENNASHORTED",
    "H1:SYS-TIMING_Y_GPS_A_NARROWBAND",
    "H1:SYS-TIMING_Y_GPS_A_FASTACQUISITION",
    "H1:SYS-TIMING_Y_GPS_A_FILTERRESET",
    "H1:SYS-TIMING_Y_GPS_A_POSITIONLOCK",
    "H1:SYS-TIMING_Y_GPS_A_DIFFERENTIALFIX",

    "H1:SYS-TIMING_C_PPS_B_SIGNAL_0_DIFF",
    "H1:SYS-TIMING_C_PPS_B_SIGNAL_1_DIFF",
    "H1:SYS-TIMING_C_PPS_B_SIGNAL_2_DIFF",
    "H1:SYS-TIMING_C_PPS_B_SIGNAL_3_DIFF",
    "H1:SYS-TIMING_C_PPS_B_SIGNAL_4_DIFF",
    "H1:SYS-TIMING_C_PPS_B_SIGNAL_5_DIFF",
    "H1:SYS-TIMING_C_PPS_B_SIGNAL_6_DIFF",
    "H1:SYS-TIMING_C_PPS_B_SIGNAL_7_DIFF",

    "H1:SYS-TIMING_X_PPS_A_SIGNAL_0_DIFF",
    "H1:SYS-TIMING_X_PPS_A_SIGNAL_1_DIFF",
    "H1:SYS-TIMING_X_PPS_A_SIGNAL_2_DIFF",
    "H1:SYS-TIMING_X_PPS_A_SIGNAL_3_DIFF",
    "H1:SYS-TIMING_X_PPS_A_SIGNAL_4_DIFF",
    "H1:SYS-TIMING_X_PPS_A_SIGNAL_5_DIFF",
    "H1:SYS-TIMING_X_PPS_A_SIGNAL_6_DIFF",
    "H1:SYS-TIMING_X_PPS_A_SIGNAL_7_DIFF",

    "H1:SYS-TIMING_Y_PPS_A_SIGNAL_0_DIFF",
    "H1:SYS-TIMING_Y_PPS_A_SIGNAL_1_DIFF",
    "H1:SYS-TIMING_Y_PPS_A_SIGNAL_2_DIFF",
    "H1:SYS-TIMING_Y_PPS_A_SIGNAL_3_DIFF",
    "H1:SYS-TIMING_Y_PPS_A_SIGNAL_4_DIFF",
    "H1:SYS-TIMING_Y_PPS_A_SIGNAL_5_DIFF",
    "H1:SYS-TIMING_Y_PPS_A_SIGNAL_6_DIFF",
    "H1:SYS-TIMING_Y_PPS_A_SIGNAL_7_DIFF"
]


# THE REST OF THE IMPORTS ARE AFTER THIS IF STATEMENT.
# Quits immediately on --help or -h flags to skip slow imports when you just
# want to read the help documentation.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("-t", "--gpstime", type=int,
                        help=("The GPS time about which the plots should be "
                              "centered."))
    parser.add_argument(
        "-o",
        "--outdir",
        default=".",
        help="""
             Where should the files be saved? Defaults to current directory
             """
    )
    parser.add_argument(
        "-l",
        "--channellist",
        help="""
             Path to a text file containing the list of channels to plot. If
             not provided, this script will try to read the channel list in
             from STDIN. If nothing is available from stdin, a default,
             comprehensive channel list will be used. The channel list
             should be a newline delimited list of valid EPICS channel names,
             though badly-formed or invalid channel names will be skipped.
             """
    )
    parser.add_argument(
        "-d",
        "--deltat",
        type=int,
        default=DT,
        help="""
             The size of the time window to be plotted. The generated plots
             will show +/- deltat seconds of data around the central GPS time.
             Defaults to: {}
             """.format(DT)
    )
    parser.add_argument(
        "-w",
        "--skipwebpage",
        action='store_true',
        help="""
             If this flag is provided, no preview webpage will be generated.
             By default, the webpage is generated.
             """
    )
    parser.add_argument(
        "-p",
        "--skipplots",
        action='store_true',
        help="""
             If this flag is provided, no plots will be generated (useful if
             you already made plots and just want to regenerate the webpage).
             By default, the plots are generated.
             """
    )
    parser.add_argument(
        "-s",
        "--maxsimultaneous",
        type=int,
        default=MAX_SIMULTANEOUS_CHANS,
        help="""
             The maximum number of simultaneous channels for which to fetch
             data at a given time. It is generally faster to fetch a batch of
             channels at a time, but getting too many channels at once might
             slow things down. Defaults to: {}
             """.format(MAX_SIMULTANEOUS_CHANS)
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help="""
             Print verbose output for status monitoring while plotting.
             Defaults to False.
             """
    )
    parser.add_argument(
        "--print-default-channels",
        action='store_true',
        help="""
            Print the default list of channels as a newline-delimited list
            and exit.
            """
    )
    args = parser.parse_args()
    if args.print_default_channels:
        print('\n'.join(DEFAULT_CHANNELS))
        exit(0)

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
    """Get the channel list from the file descriptor specified via the CLI.
    If no channel list is provided, try to read from sys.stdin. Otherwise,
    use the default comprehensive channel list."""
    if args.channellist is None:
        # if stdin is a tty, then nothing is being piped in
        if sys.stdin.isatty():
            return DEFAULT_CHANNELS
        else:
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
