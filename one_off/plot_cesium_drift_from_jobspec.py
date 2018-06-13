#!/usr/bin/env python
# (c) Stefan Countryman, 2018
# an ugly plotting script for the cesium clock
from gwpy.time import tconvert
import numpy as np
import matplotlib.pyplot as plt
from geco_gwpy_dump import Job

dat = Job.load().full_queries[0].read()


def make_plot(x_axis, y_axis, location):
    start_time = str(tconvert(int(min(x_axis))))
    end_time = str(tconvert(int(max(x_axis))))
    #converts y_axis time to microseconds from seconds
    MICROS_PER_SECOND = 1e6
    # find outliers
    outlierinds = np.nonzero(np.logical_or(y_axis > -2.8e-6, y_axis < -1e-5))
    outliertimes = x_axis[outlierinds]
    outliers = y_axis[outlierinds]
    print("Severe outliers found and removed (time: value):")
    for i, t in enumerate(outliertimes):
        print("{}: {}".format(t, outliers[i]))
    x = np.delete(x_axis, outlierinds)
    y_ax = np.delete(y_axis, outlierinds) * MICROS_PER_SECOND
    missing = np.nonzero(y_axis == 0)
    print("Missing values found and removed at times: {}".format(x[missing]))
    x = np.delete(x, missing)
    y_ax = np.delete(y_ax, missing)
    #creates array with slope and y-intercept of line of best fit
    lobf_array = np.polyfit(x, y_ax, 1)
    #dimensionless quantity that characterizes drift
    drift_coef = lobf_array[0]/MICROS_PER_SECOND
    y_axis_lobf = np.poly1d(lobf_array)(x)
    tmp = [lobf_array[0] * i for i in x]
    tmp += lobf_array[1]
    y_dif = y_ax - tmp
    print('making plots')
    fig = plt.figure(figsize=(13,18))
    plt.suptitle('Drift of cesium clock, from ' +
                 start_time + ' until ' + end_time +
                 location, fontsize=20)
    plt.subplots_adjust(top=0.88888888, bottom=0.1)
    ax1 = fig.add_subplot(211)
    ax1.set_title('Line of best fit versus offset')
    ax1.plot(x, y_ax, '#ff0000')
    ax1.plot(x, y_axis_lobf, '#617d8d')
    print(type(drift_coef))
    print(str(drift_coef))
    ax1.text(0.01, 0.05, 'Drift coefficient = ' + str(drift_coef),
             transform=ax1.transAxes, bbox=dict(facecolor='#99ccff',
             boxstyle='round', alpha=0.25))
    ax1.set_xlabel('GPS time')
    ax1.set_ylabel('Offset [$\mu$s]')
    ax2 = fig.add_subplot(212)
    ax2.plot(x, y_dif)
    ax2.set_xlabel('GPS time')
    ax2.set_title('Residual of the line of best fit')
    ax2.set_ylabel('Difference [$\mu$s]')
    print('drift coefficient:')
    print(drift_coef)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    fig.savefig('cesium_clock_drift_from_' + start_time.replace(' ','_') +
                '_until_' + end_time.replace(' ','_') + '.png', dpi=300)
 
if __name__ == "__main__":
    make_plot(dat.times.to("s").value, dat.value, "LLO")
