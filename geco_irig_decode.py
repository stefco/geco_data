#!/usr/bin/env python
# (c) Stefan Countryman, 2016-2017

import sys
import datetime
from textwrap import fill
import numpy as np
# import scipy.ndimage.filters as scf

if len(sys.argv) > 1:
    print(
        "Usage: {} <input_file.txt\n\n" +
        fill(
            """Read a raw IRIG-B signal from STDIN and print out decoded
            timestamps.  Data must be a newline-delimited list of floating
            point values representing the value of the IRIG-B signal at each
            point in time.  Sample rate must be 16,384 (2^14) Hz. The input
            file must contain an integer number of seconds worth of data, i.e.
            it must have (Sample Rate) x (N) values, where N is the number of
            seconds that must be decoded, and the data must start at the
            beginning of a second.
            """.format(sys.argv[0])
        )
    )
    exit(1)

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------

# max and min of histogram, and number of bins
SAMPLE_RATE = 16384     # ADC sample rate

# --------------------------------------------------------------------------
# IRIG-B RELATED CONSTANTS
# --------------------------------------------------------------------------
BITS_PER_SECOND = 100   # bits per second in IRIG-B spec
CONVOLUTION_SIGMA = 1e-4
HIGH_SIGNAL_THRESHOLD = 3500

# find the high/low test points for each type of bit (0, 1, or control);
# measured as indices from start of each bit
# TODO measure in fractions of a second and convert with SAMPLE_RATE
TEST_POINTS = np.array([20, 60, 110, 150])

# find the start of each bit
BIT_STARTS = np.round(np.arange(0, 1, 1./BITS_PER_SECOND) * SAMPLE_RATE)
ALL_TEST_POINT_INDICES = (BIT_STARTS[:, np.newaxis] + TEST_POINTS).astype(int)

# representations of each type of bit (0, 1, or control)
REP_0 = [1, 0, 0, 0]
REP_1 = [1, 1, 0, 0]
REP_C = [1, 1, 1, 0]
CURRENT_CENTURY = 20

# how many seconds does each bit represent?
seconds = np.zeros(BITS_PER_SECOND, dtype=int)
seconds[1] = 1
seconds[2] = 2
seconds[3] = 4
seconds[4] = 8
seconds[6] = 10
seconds[7] = 20
seconds[8] = 40

# how many minutes does each bit represent?
minutes = np.zeros(BITS_PER_SECOND, dtype=int)
minutes[10] = 1
minutes[11] = 2
minutes[12] = 4
minutes[13] = 8
minutes[15] = 10
minutes[16] = 20
minutes[17] = 40

# how many hours does each bit represent?
hours = np.zeros(BITS_PER_SECOND, dtype=int)
hours[20] = 1
hours[21] = 2
hours[22] = 4
hours[23] = 8
hours[25] = 10
hours[26] = 20

# how many days does each bit represent?
days = np.zeros(BITS_PER_SECOND, dtype=int)
days[30] = 1
days[31] = 2
days[32] = 4
days[33] = 8
days[35] = 10
days[36] = 20
days[37] = 40
days[38] = 80
days[40] = 100
days[41] = 200

# how many years does each bit represent?
years = np.zeros(BITS_PER_SECOND, dtype=int)
years[50] = 1
years[51] = 2
years[52] = 4
years[53] = 8
years[55] = 10
years[56] = 20
years[57] = 40
years[58] = 80

control_bits = np.zeros(BITS_PER_SECOND, dtype=bool)
control_bits[range(9,100,10)] = True
control_bits[0] = True

#-------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#-------------------------------------------------------------------------------

def get_date_from_timeseries(timeseries):
    """Decode the input waveform, which is assumed to be a 16384hz digitized
    IRIG-B signal using DCLS (DC Level Shift)."""
    # filter the timeseries to remove ringing at corners
    # filt = scf.gaussian_filter1d(timeseries, CONVOLUTION_SIGMA * SAMPLE_RATE)

    # check all test points
    bits_high = (timeseries[ALL_TEST_POINT_INDICES] >= HIGH_SIGNAL_THRESHOLD).astype(int)

    # represent control bits with the number 2
    bits = np.zeros(BITS_PER_SECOND, dtype=int)
    for i in range(bits_high.shape[0]):
        if   np.all(bits_high[i] == REP_0): bits[i] = 0
        elif np.all(bits_high[i] == REP_1): bits[i] = 1
        elif np.all(bits_high[i] == REP_C): bits[i] = 2
        else: raise ValueError("Bad bit: " + str(i))

    # are the control bits in the correct spots?
    if not np.all((bits == 2) == control_bits):
        raise ValueError("Control bits are not present where expected: \n"
                        + str((bits == 2) == control_bits))

    # find total seconds, minutes, hours, days, and years
    tot_s = bits.dot(seconds)
    tot_m = bits.dot(minutes)
    tot_h = bits.dot(hours)
    tot_d = bits.dot(days)
    tot_y = bits.dot(years) + 100*CURRENT_CENTURY

    # return datetime from this
    jan1 = datetime.datetime(tot_y, 1, 1, tot_h, tot_m, tot_s)
    return jan1 + datetime.timedelta(int(tot_d) - 1)

def print_formatted_date(converted_date):
    # finally, print the date
    print(converted_date.strftime('%a %b %d %X %Y'))

# and will cause a ValueError.
def read_1_second_from_stdin():
    # read in data from stdin; don't read more than a second worth of data
    timeseries = np.zeros(SAMPLE_RATE)
    line = ''
    i = 0
    while i < SAMPLE_RATE:
        line = sys.stdin.readline()
        if not line:
            if i == 0:
                raise EOFError('Hit EOF at end of a second.')
            else:
                raise ValueError('Hit EOF ' + str(i) + ' lines into second; '
                                 'provide integer number of seconds of data.')
        timeseries[i] = float(line)
        i += 1
    return timeseries

def main():
    while True:
        try:
            timeseries = read_1_second_from_stdin()
            print_formatted_date(get_date_from_timeseries(timeseries))
        except EOFError:
            return

# run this if we are running from command line
if __name__ == "__main__":
    main()
