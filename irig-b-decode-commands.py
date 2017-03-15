#!/usr/bin/env python
# (c) Stefan Countryman
# Perform some IRIG-B checks for an event:
#   1. Decode IRIG-B signals for the times surrounding the event
#   2. Check that the times are correct.
#   3. Make plots of the event time.

import geco_irig_plot
import geco_irig_decode
import gwpy.timeseries
import astropy.time

BITRATE = 16384
print('BITRATE: {}'.format(BITRATE))
DT = 30 # number of seconds before and after the event to check
print('DT: {}'.format(DT))
IFOS = ['H1', 'L1']
print('IFOS: {}'.format(IFOS))
ARMS = ['X', 'Y']
print('ARMS: {}'.format(ARMS))
CHANS = ['{}:CAL-PCAL{}_IRIGB_OUT_DQ'.format(ifo, arm)
         for ifo in IFOS for arm in ARMS]
print('CHANS: {}'.format(CHANS))

with open('graceid.txt', 'r') as f:
    graceid = f.read().strip()
print('graceid: {}'.format(graceid))

with open('starttime.txt', 'r') as f:
    start_time = int(float(f.read().strip()))
print('start_time: {}'.format(start_time))

t = astropy.time.Time(start_time, format='gps', scale='utc')
print('t: {}'.format(t))
utc_start_time = t.to_datetime().strftime('%a %b %d %X %Y')
print('utc_start_time: {}'.format(utc_start_time))

def get_leap_seconds(gps):
    """Find the number of leap seconds at a given gps time using astropy's
    time module (by comparing UTC to TAI and subtracting 19, the base difference
    between TAI and GPS scales)."""
    t = astropy.time.Time(gps, format='gps')
    t.format = 'iso'
    return (t.tai.to_datetime() - t.utc.to_datetime()).seconds - 19

leap_seconds = get_leap_seconds(start_time)

# make plots
for chan in CHANS:
    print('plotting {}'.format(chan))
    timeseries = gwpy.timeseries.TimeSeries.fetch(chan, start_time,
                                                  start_time + 1).value
    title = geco_irig_plot.irigb_decoded_title(timeseries, chan, utc_start_time)
    of = '{}_{}.png'.format(start_time, chan).replace(':', '_')
    geco_irig_plot.plot_with_zoomed_views(timeseries, title, output_filename=of)

# decode times
with open('{}-decoded-times.txt'.format(graceid), 'w') as f:
    f.write('Times are allowed to be off by current number of leap seconds.\n')
    f.write('This most likely indicates that the device producing the IRIG-B\n')
    f.write('signal is outputting GPS time.\n')
    for chan in CHANS:
        format_string = 'Decoding {}s surrounding {} at {} on channel {}.\n'
        msg = format_string.format(DT, graceid, start_time, chan)
        f.write(msg)
        print(msg[:-1])
        timeseries = gwpy.timeseries.TimeSeries.fetch(chan, start_time - DT,
                                                      start_time + DT + 1).value
        # deal with one second at a time, writing results to file
        for i in range(2*DT + 1):
            timeseries_slice = timeseries[i*BITRATE:(i+1)*BITRATE]
            gps_actual = (start_time - DT) + i
            t_actual = astropy.time.Time(gps_actual, format='gps', scale='utc')
            t = geco_irig_decode.get_date_from_timeseries(timeseries_slice)
            dt = (t - t_actual.to_datetime()).seconds
            # check whether the times agree, or whether they are off by the
            # current number of leap seconds
            if dt == 0:
                f.write('{} OK, IS UTC\n'.format(t.strftime('%a %b %d %X %Y')))
            elif dt == leap_seconds:
                f.write('{} OK, IS GPS\n'.format(t.strftime('%a %b %d %X %Y')))
            else:
                t_actual_str = t_actual.to_datetime().strftime('%a %b %d %X %Y')
                f.write('{} IS WRONG! '.format(t.strftime('%a %b %d %X %Y')))
                f.write('SHOULD BE {}\n'.format(t_actual_str))
