#!/usr/bin/env python
# (c) Stefan Countryman, 2016-2017

# plot an IRIG-B signal read from stdin or from a textfile
# assumes that the input values each appear on a new line

# Force matplotlib to not use any Xwindows backend. NECESSARY ON CLUSTER.
import matplotlib
matplotlib.use('Agg')
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import geco_irig_decode

FAST_CHANNEL_BITRATE = 16384  # for IRIG-B, DuoTone, etc.

def main():
    """Get arguments from command line and figure out which plot to make."""
    args = get_parser().parse_args()
    timeseries = read_timeseries_stdin(FAST_CHANNEL_BITRATE,
                                       cat_to_stdout=args.timeseries)
    title = irigb_decoded_title(timeseries, args.detector, args.actualtime)
    output_filename = irigb_output_filename(args.outfile)
    plot_with_zoomed_views(timeseries, title, num_subdivs=5, dt=1.,
                           output_filename=output_filename)

def get_parser():
    """Define an argument parser for this script."""
    parser = argparse.ArgumentParser()
    # TODO: make this -i and --ifo instead of detector.
    parser.add_argument("--detector",
                        help=("the detector; used in the title of the output "
                              "plot"))
    parser.add_argument("-O", "--outfile",
                        help="the filename of the generated plot")
    parser.add_argument("-T", "--timeseries",
                        help="copy from stdin to stdout while reading",
                        action="store_true")
    parser.add_argument("-A", "--actualtime",
                        help=("actual time signal was recorded "
                              "(appears in title)"))
    return parser

def read_timeseries_stdin(num_lines, cat_to_stdout=False):
    """Read in newline-delimited numerical data from stdin; don't read more
    than a second worth of data. If cat_to_stdout is True, print data that
    has been read in back to stdout (useful for piped commands)."""
    timeseries = np.zeros(num_lines)
    line = ""
    i = 0
    while i < num_lines:
        line = float(sys.stdin.readline())
        timeseries[i] = line
        if cat_to_stdout:
            print(line)
        i += 1
    return timeseries

def irigb_decoded_title(timeseries, IFO=None, actual_time=None):
    """Get a title for an IRIG-B timeseries plot that includes the decoded
    time in the timeseries itself."""
    # get the detector name
    if IFO is None:
        detector_suffix = ""
    else:
        detector_suffix = " at " + IFO

    # get the actual time of recording, if provided
    if actual_time is None:
        actual_time_str = ""
    else:
        actual_time_str = "\nActual Time: {}".format(actual_time)

    # add title and so on
    try:
        decoded_time = geco_irig_decode.get_date_from_timeseries(timeseries)
        decoded_time_str = decoded_time.strftime('%a %b %d %X %Y')
    except ValueError as e:
        decoded_time_str = "COULD NOT DECODE TIME"
    fmt = "One Second of IRIG-B Signal{}\nDecoded Time: {}{}"
    return fmt.format(detector_suffix, decoded_time_str, actual_time_str)

def irigb_output_filename(outfile=None):
    """Get the output filename for an IRIG-B plot."""
    if outfile is None:
        output_filename = "irigb-plot-made-at-" + str(time.time()) + ".png"
    else:
        output_filename = outfile
        # append .png if not already there
        if output_filename.split(".")[-1] != "png":
            output_filename += ".png"
    return output_filename

def plot_with_zoomed_views(timeseries, title, num_subdivs=5, dt=1.,
                           output_filename=None):
    """Plot a timeseries and produce num_subdivs subplots that show equal-sized
    subdivisions of the full timeseries data to show details (good for
    high-bitrate timeseries)."""
    bitrate = int(len(timeseries) / float(dt))
    times = np.linspace(0, 1, num=bitrate, endpoint=False)
    
    # find max and min values in timeseries; use these to set plot boundaries
    yrange = timeseries.max() - timeseries.min()
    ymax = timeseries.max() + 0.1*yrange
    ymin = timeseries.min() - 0.1*yrange
    
    # print("making plot")
    plt.close('all')
    plt.figure(1, figsize=(2+num_subdivs,10)) # approx. 1 inch per zoomed plot
    # plot the full second on the first row; lines should be black ('k' option).
    plt.subplot(num_subdivs + 1, 1, 1)
    plt.ylim(ymin, ymax)
    plt.plot(times, timeseries, 'k')
    plt.tick_params(axis='y', labelsize='small')
    # make num_subdivs subplots to better show the full second
    for i in range(num_subdivs):
        # print("making plot " + str(i))
        plt.subplot(num_subdivs+1, 1, i+2)
        plt.ylim(ymin, ymax)
        plt.xlim(float(i)/num_subdivs, (float(i)+1)/num_subdivs)
        start = bitrate*i     // num_subdivs
        end   = bitrate*(i+1) // num_subdivs
        plt.plot(times[start:end], timeseries[start:end], 'k')
        plt.tick_params(axis='y', labelsize='small')
    plt.suptitle(title)
    plt.xlabel("Time since start of second [$s$]")
    # print("saving plot")
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2,
                        hspace=0.5)
    if not (output_filename is None):
        plt.savefig(output_filename)

    return plt

if __name__ == '__main__':
    main()
