#!/usr/bin/env python
# (c) Stefan Countryman, 2017
# pylint: disable=superfluous-parens

"""
USAGE: geco_irigb_midnight_bug_check.py START END

Check +/- 30 seconds around midnight in the given date range. START and END
should both be something like 2018-10-03. All dates between these two dates
(endpoints inclusive) will be checked at 00:00:00 (the start of each date).
"""

import sys

# spit out help string before slow imports if necessary
if __name__ == "__main__" and {"-h", "--help"}.intersection(sys.argv):
    print(__doc__)

# pylint: disable=wrong-import-position
from datetime import timedelta
from astropy.time import Time
from gwpy.timeseries import TimeSeries
from dateutil.parser import parse as parse_datetime
import geco_irig_decode

RED = '\033[91m'
CLEAR = '\033[0m'
BITRATE = 16384
IFOS = ['H1', 'L1']
ARMS = ['X', 'Y']
CHANS = ['{}:CAL-PCAL{}_IRIGB_OUT_DQ'.format(ifo, arm)
         for ifo in IFOS for arm in ARMS]
TFORMAT = '%a %b %d %X %Y'


def get_leap_seconds(gps):
    """Find the number of leap seconds at a given gps time using astropy's time
    module (by comparing UTC to TAI and subtracting 19, the base difference
    between TAI and GPS scales)."""
    t = Time(gps, format='gps')
    t.format = 'iso'
    return (t.tai.to_datetime() - t.utc.to_datetime()).seconds - 19


def check_decoded_times(start_time, timewindow=30):
    """Check the decoded IRIG-B times within a window of size timewindow
    surrounding the event time and make sure the times are all correct (i.e.
    they are either the correct UTC or GPS time; either one is considered
    correct). Save the results of the check to a text file for upload to LIGO's
    EVNT log."""
    for chan in CHANS:
        format_string = 'Decoding {}+/-{}s on channel {}.'
        msg = format_string.format(start_time, timewindow, chan)
        print(msg)
        # print a header, since output will be arranged tabularly, e.g.
        # 1225152018 | 1225152018 |   18 | GPS   |   0 |   0 |  0 | 306 | ...
        # 2018 | Thu Nov 02 00:00:00 2018 | Fri Nov 02 23:59:42 2018
        print("Actual GPS | Decode GPS | Leap | Scale | Sec | Min | Hr | Day "
              "| Year | Decoded Date/Time        | Actual UTC Date/Time")
        print("-----------+------------+------+-------+-----+-----+----+-----"
              "+------+--------------------------+-------------------------")
        row_fmt = ("{gps_actual:>10d} | {gps_decoded:>10d} | {leap:>4d} | "
                   "{scale:<5} | {second:>3d} | {minute:>3d} | {hour:>2d} | "
                   "{day:>3d} | {year:>4d} | {datetime_decoded:<24} | "
                   "{datetime_actual:<24}")
        timeseries = TimeSeries.fetch(chan, start_time-timewindow,
                                      start_time+timewindow+1).value
        # deal with one second at a time, writing results to file
        for i in range(2*timewindow + 1):
            timeseries_slice = timeseries[i*BITRATE:(i+1)*BITRATE]
            gps_actual = (start_time - timewindow) + i
            leap_seconds = get_leap_seconds(gps_actual)
            t_actual = Time(gps_actual, format='gps', scale='utc')
            decoded = geco_irig_decode.decode_timeseries(timeseries_slice)
            t = decoded['datetime_decoded']
            dt = (t - t_actual.to_datetime()).seconds
            datetime_actual = t_actual.to_datetime().strftime(TFORMAT)
            # check whether the times agree, or whether they are off by the
            # current number of leap seconds
            if dt == 0:
                scale = "UTC"
            elif dt == leap_seconds:
                scale = "GPS"
            else:
                scale = "ERROR"
            print(row_fmt.format(gps_actual=gps_actual,
                                 gps_decoded=Time(t).gps, leap=leap_seconds,
                                 scale=scale, datetime_actual=datetime_actual,
                                 **decoded))


def main():
    """Check the specified date range; see module docstring."""
    start = parse_datetime(sys.argv[1]).replace(hour=0, minute=0, second=0)
    end = parse_datetime(sys.argv[2]).replace(hour=0, minute=0, second=0)
    days = (end - start).days + 1
    print('Processing {} days between {} and {}.'.format(days, start, end))
    print('Times are allowed to be off by current number of leap')
    print('seconds. This most likely indicates that the device')
    print('producing the IRIG-B signal is outputting GPS time.')
    if days < 1:
        sys.stderr.write("end day cannot be earlier than start day.")
        exit(1)
    for day in (start + timedelta(d) for d in range(days)):
        print("CHECKING DATE: {}".format(day.isoformat()))
        check_decoded_times(Time(day).gps)


if __name__ == "__main__":
    main()
