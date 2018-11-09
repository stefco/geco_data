#!/usr/bin/env python
# (c) Stefan Countryman 2017

DESC = """Perform some IRIG-B checks for an event:
  1. Decode IRIG-B signals for the times surrounding the event
  2. Check that the times are correct.
  3. Make plots of the event time.
"""
EPILOG = """If no gpstime or graceid are provided as arguments to
the script, the script will look for files named "starttime.txt"
and "graceid.txt" for these values (which should be the sole
contents of their respective files).
"""
BITRATE = 16384
DT = 30 # number of seconds before and after the event to check
IFOS = ['H1', 'L1']
ARMS = ['X', 'Y']
CHANS = ['{}:CAL-PCAL{}_IRIGB_OUT_DQ'.format(ifo, arm)
         for ifo in IFOS for arm in ARMS]
TFORMAT = '%a %b %d %X %Y'

# THE REST OF THE IMPORTS ARE AFTER THIS IF STATEMENT.
# Quits immediately on --help or -h flags to skip slow imports when you just
# want to read the help documentation.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=DESC,
                                     epilog=EPILOG)
    parser.add_argument("-t", "--gpstime", type=float,
                        help="The gps time of the event.")
    parser.add_argument("-g", "--graceid",
                        help="The GraceDB ID of the event.")
    args = parser.parse_args()

import geco_irig_plot
import geco_irig_decode
import gwpy.timeseries
import astropy.time

def get_leap_seconds(gps):
    """Find the number of leap seconds at a given gps time using astropy's
    time module (by comparing UTC to TAI and subtracting 19, the base difference
    between TAI and GPS scales)."""
    t = astropy.time.Time(gps, format='gps')
    t.format = 'iso'
    return (t.tai.to_datetime() - t.utc.to_datetime()).seconds - 19

def make_plots(start_time):
    """Make IRIG-B plots for this particular GPS start time, including the
    decoded IRIG-B time (as well as the correct, intended IRIG-B time) in
    the titles of the plots."""
    t = astropy.time.Time(start_time, format='gps', scale='utc')
    utc_start_time = t.to_datetime().strftime(TFORMAT)
    print('utc_start_time: {}'.format(utc_start_time))
    for chan in CHANS:
        print('plotting {}'.format(chan))
        timeseries = gwpy.timeseries.TimeSeries.fetch(chan, start_time,
                                                      start_time + 1).value
        title = geco_irig_plot.irigb_decoded_title(timeseries, chan,
                                                   utc_start_time)
        outfile = '{}_{}.png'.format(start_time, chan).replace(':', '_')
        geco_irig_plot.plot_with_zoomed_views(timeseries, title,
                                              output_filename=outfile)

def check_decoded_times(start_time, graceid, DT=DT):
    """Check the decoded IRIG-B times within a window of size DT surrounding the
    event time and make sure the times are all correct (i.e. they are either
    the correct UTC or GPS time; either one is considered correct). Save
    the results of the check to a text file for upload to LIGO's EVNT log."""
    with open('{}-decoded-times.txt'.format(graceid), 'w') as f:
        f.write('Times are allowed to be off by current number of leap\n')
        f.write('seconds. This most likely indicates that the device\n')
        f.write('producing the IRIG-B signal is outputting GPS time.\n\n')
        for chan in CHANS:
            format_string = 'Decoding {}s surrounding {} at {} on channel {}.\n'
            msg = format_string.format(DT, graceid, start_time, chan)
            f.write(msg)
            print(msg[:-1])
            timeseries = gwpy.timeseries.TimeSeries.fetch(chan, start_time-DT,
                                                          start_time+DT+1).value
            # deal with one second at a time, writing results to file
            for i in range(2*DT + 1):
                timeseries_slice = timeseries[i*BITRATE:(i+1)*BITRATE]
                gps_actual = (start_time - DT) + i
                leap_seconds = get_leap_seconds(gps_actual)
                t_actual = astropy.time.Time(gps_actual, format='gps',
                                             scale='utc')
                t = geco_irig_decode.get_date_from_timeseries(timeseries_slice)
                dt = (t - t_actual.to_datetime()).seconds
                # check whether the times agree, or whether they are off by the
                # current number of leap seconds
                if dt == 0:
                    f.write('{} OK, IS UTC\n'.format(t.strftime(TFORMAT)))
                elif dt == leap_seconds:
                    f.write('{} OK, IS GPS\n'.format(t.strftime(TFORMAT)))
                else:
                    t_actual_str = t_actual.to_datetime().strftime(TFORMAT)
                    f.write('{} IS WRONG! '.format(t.strftime(TFORMAT)))
                    f.write('SHOULD BE {}\n'.format(t_actual_str))

if __name__ == "__main__":
    if args.graceid is None:
        with open('graceid.txt', 'r') as f:
            graceid = f.read().strip()
    else:
        graceid = args.graceid

    if args.gpstime is None:
        with open('starttime.txt', 'r') as f:
            start_time = int(float(f.read().strip()))
    else:
        start_time = int(args.gpstime) # already a float via parser

    print('BITRATE: {}'.format(BITRATE))
    print('DT: {}'.format(DT))
    print('IFOS: {}'.format(IFOS))
    print('ARMS: {}'.format(ARMS))
    print('CHANS: {}'.format(CHANS))
    print('graceid: {}'.format(graceid))
    print('start_time: {}'.format(start_time))

    make_plots(start_time)
    check_decoded_times(start_time, graceid)
