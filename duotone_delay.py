#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Stefan Countryman 2017, several functions translated from MATLAB code by
# Keita Kawabe (translated code attributed in dosctrings). Keita's code
# available at:
# svn.ligo.caltech.edu/svn/aligocalibration/trunk/Common/MatlabTools/timingi
# edited Yasmeen Asali 2019, new timing channel names updated
# Functions to measure DuoTone timing delay and make DuoTone related plots.

DESC="""A module (that can also be used as a script) for plotting delay
histograms in DuoTone signals as well as DuoTone overlay plots. Code is written
in python. Several functions are translated from Keita Kawabe's MATLAB code
for. Data is fetched from LIGO's NDS2 servers using gwpy.

A caveat to the user: only the commissioningFrameDuotoneStat plotting component
has been tested, so commissioningFrameDuotone will look horrible and need
improvements if it is to be used.

Keita's original MATLAB code available at:
svn.ligo.caltech.edu/svn/aligocalibration/trunk/Common/MatlabTools/timing
"""
EPILOG="""EXAMPLES:

"""
MINUTES = 5
SECONDS_PER_MINUTE = 60
IFOs = ['H1', 'L1']

# THE REST OF THE IMPORTS ARE AFTER THIS IF STATEMENT.
# Quits immediately on --help or -h flags to skip slow imports when you just
# want to read the help documentation.
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=DESC, epilog=EPILOG)
    parser.add_argument('-s','--stat', action='store_true', default=False,
                        help=("Make commissioningFrameDuotoneStat plots, i.e. "
                              "histograms of the deviation of the DuoTone "
                              "zero-crossing delay from the expected deviation "
                              "for each second in a {} minute time interval "
                              "surrounding the specified GPS time, as well as "
                              "a vertical line indicating the DuoTone delay "
                              "deviation at the specified GPS time. "
                              "Note: if running on pre January 2019 data, "
                              "manually uncomment the function with old channel names. "
                              "Based on Keita's MATLAB code.").format(MINUTES))
    parser.add_argument('-i','--ifo', choices=IFOs,
                        help=('Which IFO to include in the plot.'))
    parser.add_argument('-t','--gpstime', type=float,
                        help=('GPS time of the event.'))
    args = parser.parse_args()

# need print function for newline-free printing
import matplotlib
# Force matplotlib to not use any Xwindows backend. NECESSARY FOR HEADLESS.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Use gwpy to fetch data
import gwpy.timeseries
import numpy as np
import scipy.signal

# get a list of channels to plot and analyze
def chans(IFO):
    return  ['{}:CAL-PCALX_FPGA_DTONE_ADC_DQ'.format(IFO),
             '{}:CAL-PCALY_FPGA_DTONE_ADC_DQ'.format(IFO),
             '{}:OMC-FPGA_DTONE_IN1_DQ'.format(IFO),
             '{}:CAL-PCALX_DAC_DTONE_LOOPBACK_DQ'.format(IFO),
             '{}:CAL-PCALY_DAC_DTONE_LOOPBACK_DQ'.format(IFO)]
             #'{}:CAL-PCALX_FPGA_DTONE_DAC_DQ'.format(IFO), #new extra DAC channel
             #'{}:CAL-PCALY_FPGA_DTONE_DAC_DQ'.format(IFO)] #new extra DAC channel

'''
#uncomment this function to generate plots for the old timing channels (pre January 2019) 
def chans(IFO):
    return  ['{}:CAL-PCALX_FPGA_DTONE_IN1_DQ'.format(IFO),
             '{}:CAL-PCALY_FPGA_DTONE_IN1_DQ'.format(IFO),
             '{}:OMC-FPGA_DTONE_IN1_DQ'.format(IFO),
             '{}:CAL-PCALX_DAC_FILT_DTONE_IN1_DQ'.format(IFO),
             '{}:CAL-PCALY_DAC_FILT_DTONE_IN1_DQ'.format(IFO)]
'''

def duotoneDelay(duotone, f1, f2, t):
    """Directly translated from Keita's MATLAB function of the same name.
    Docstring copied from MATLAB function.
    Estimates the time delay of the duotone relative to the first sample
     (which is assumed to be the second boundary) by calculating the amplitude
     and phase of each sine wave separately by multiplying sine and cosine
     component of f1 and f2 and integrating over the entire duration:
    
    a1=sum(duotone*(sin(2*pi*f1*t) + 1i*cos(2*pi*f1*t)))*dt*2/duration;
    d1=-atan2(imag(a1), real(a1)) /2/pi/f1;
     (do the same for f2)
    delay = (d1+d2)/2;
    residual = duotone - abs(a1)*sin(2*pi*f1*(t-delay)) ...
                       - abs(a2)*sin(2*pi*f2*(t-delay));
   
    Positive delay means that the duotone signal origin is delayed from the
    first sample.
   
    duotone: Time series of duotone signal. 
             Make sure that the duration of measurement is exactly N seconds
              where N is a positive integer.
             Also make sure that the first sample is exactly on the second
              boundary.
    f1, f2:  First and second duotone frequency. This is NOT fit. In aLIGO
              timing system, these are always 960 and 961 Hz.
    t:       Time. 
             Note that you should not feed the time axis output by dtt.
             In dtt, time is calculated as dt*n with dt being single
              precision or something,  and the error accumulates."""
    dt = (t[-1] - t[0]) / (len(t) - 1)
    duration = dt*len(t)

    sin1 = np.sin(2 * np.pi * f1 * t)
    cos1 = np.cos(2 * np.pi * f1 * t)
    a1   = np.sum(duotone * (sin1 + 1j*cos1)) * dt *  2 / duration
    d1   = -np.arctan2(np.imag(a1), np.real(a1)) / 2 / np.pi / f1

    sin2 = np.sin(2 * np.pi * f2 * t)
    cos2 = np.cos(2 * np.pi * f2 * t)
    a2   = np.sum(duotone * (sin2 + 1j*cos2)) * dt *  2 / duration
    d2   = -np.arctan2(np.imag(a2), np.real(a2)) / 2 / np.pi / f2

    # this is the DELAY, positive delay means that the duotone in ADC is
    # delayed from the second boundary by this much.
    delay = (d1 + d2)/2

    residual = (  duotone 
                - np.abs(a1) * np.sin(2 * np.pi * f1 * (t - delay))
                - np.abs(a2) * np.sin(2 * np.pi * f2 * (t - delay)))

    return (delay, residual)

def commissioningFrameDuotone(IFO, sGPS, drawPlots=False):
    """Directly translated from Keita's MATLAB function of the same name.
    Docstring copied from MATLAB function.
    Measure the ADC timestamp delay relative to hardware duotone signal 
    generated by the timing system as well as round trip delay including
    AI and AA of pcal. 
    Measurement time is 1second.
    IFO: 'L1' or 'H1'.
    sGPS: start GPS time
    drawPlots: non-zero (true) for plotting, zero for no plot.
               Default=false
    Make sure to do Kerberos authentication before using this."""

    CHANS = chans(IFO)

    # use "fetch" to make sure we are using NDS2, since loading from frame
    # files does not always work
    bufs = gwpy.timeseries.TimeSeriesDict.fetch(CHANS, int(sGPS), int(sGPS)+1,
                                                verbose=False)

    delay = np.zeros(len(CHANS))
    plot_positions = [1, 3, 5, 2, 4]
    subplot_title_format = '{}\nRMS={:6f}, delay={:6f}µs'
    if drawPlots:
        plt.close()
    for i in range(len(CHANS)):
        sample_rate = bufs[CHANS[i]].sample_rate.value
        t = np.linspace(0, 1, len(bufs[CHANS[i]]), endpoint=False)
        x = bufs[CHANS[i]].value
        delay[i], residual = duotoneDelay(x, 960, 961, t)

        if drawPlots:
            plt.subplot(3, 2, plot_positions[i])
            plt.plot(t, x, 'b', t, residual, 'r')
            dtRMS  = np.sqrt(np.mean(np.square(dtone)))
            resRMS = np.sqrt(np.mean(np.square(residual)))
            plt.grid('on')
            title = subplot_title_format.format(CHANS[i],
                                                dtRMS,
                                                delay[i]*1e6)
            if i in (0,1):
                title = 'ADC timestamp delay WRT duotone\n' + title
            if i in (3,4):
                title = 'Loopback delay with analog AA and AI\n' + title
            plt.title(title, {'fontsize': 'small'})
            plt.xlabel('Time (sec)')
    if drawPlots:
        plt.suptitle('DuoTone channels at {} at GPS {}'.format(IFO, int(sGPS)))
        plt.tight_layout()
        plt.savefig('duotone_plots_{}_{}.png'.format(IFO, int(sGPS)))
    return tuple(delay)

def commissioningFrameDuotoneStat(IFO, eGPS):
    """Directly translated from Keita's MATLAB function of the same name.
    Docstring copied from MATLAB function.
    [pcalxDelay, pcalyDelay, omcDelay]=commissioningFrameDuotoneStat(IFO, eGPS)
   
    Measure the ADC timestamp delay relative to hardware duotone signal,
    as well as round trip delay including AI and AA of pcal before and after 
    5 minutes of the event GPS time eGPS. 
    
    IFO: 'L1' or 'H1'.
    eGPS: event GPS time
    Make sure to do Kerberos authentication before using this."""

    # duotone board delay relative to 1pps according to Zsuzsa
    omcBoardDelay = 6699e-9
    print('omcBoardDelay: {}'.format(omcBoardDelay))
    # variation between duotone board according to Zsuzsa.
    omcBoardDelayErr = 28e-9
    print('omcBoardDelayErr: {}'.format(omcBoardDelayErr))
    # 64k to 16k decimation, was hardcoded as 55.93e-6.
    decim4xDelay = iopDecimDelay(IFO, 960, eGPS)
    print('decim4xDelay: {}'.format(decim4xDelay))
    #same filter as decimation
    upsample4xDelay = decim4xDelay
    print('upsample4xDelay: {}'.format(upsample4xDelay))
    # 2 samples are stored upstream of DAC output.
    fifoDelay = 2./(2**16)
    print('fifoDelay: {}'.format(fifoDelay))
    # analog aa
    aaDelay = 39.82e-6
    print('aaDelay: {}'.format(aaDelay))
    # analog ai
    aiDelay = aaDelay
    print('aiDelay: {}'.format(aiDelay))
    # 1 cycle in user model processing
    userCycle = 1./(2**14)
    print('userCycle: {}'.format(userCycle))
    # 1 cycle from the output of upsample filter to fifo (i.e. iop processing). 
    iopCycle = 1./(2**16)
    print('iopCycle: {}'.format(iopCycle))
    # half cycle offset of DAC clock.
    dacClockNominalOfs = iopCycle/2.
    print('dacClockNominalOfs: {}'.format(dacClockNominalOfs))
    zeroOrderHold = iopCycle/2.
    print('zeroOrderHold: {}'.format(zeroOrderHold))

    # plus 4x decimation delay in the frontend.
    expectedAdcDelay  = omcBoardDelay + decim4xDelay
    print('expectedAdcDelay: {}'.format(expectedAdcDelay))
    expectedRoundTrip = (  userCycle
                         + upsample4xDelay
                         + iopCycle
                         + fifoDelay
                         + dacClockNominalOfs
                         + zeroOrderHold
                         + aiDelay
                         + aaDelay
                         + decim4xDelay)
    print('expectedRoundTrip: {}'.format(expectedRoundTrip))

    # look 5 minutes forwards and backward in time
    ts = np.array([range(-MINUTES * SECONDS_PER_MINUTE,
                          MINUTES * SECONDS_PER_MINUTE + 1)]).transpose()

    pxDelays       = np.zeros(len(ts))
    pxRtAiAaDelays = np.zeros(len(ts))
    pyDelays       = np.zeros(len(ts))
    pyRtAiAaDelays = np.zeros(len(ts))
    omcDelays      = np.zeros(len(ts))

    print('Fetching data. Progress:')
    NUM_STATUS_UPDATES = 10.
    for i in range(len(ts)):
        # print download status
        if (  int(i     * NUM_STATUS_UPDATES / len(ts))
            - int((i-1) * NUM_STATUS_UPDATES / len(ts)) == 1):
            print('{}% done.'.format(int(i * 100. / len(ts))))
        (pxDelays[i], pyDelays[i], omcDelays[i], pxRtAiAaDelays[i],
         pyRtAiAaDelays[i]) = commissioningFrameDuotone(IFO, ts[i]+eGPS, False)
    print('Done fetching data, plotting now.')

    pxRtAiAaDelays = pxRtAiAaDelays - pxDelays
    pyRtAiAaDelays = pyRtAiAaDelays - pyDelays
    idx = np.argwhere(ts == 0)[0][0]

    ROTATION_ANGLE = 20
    HEADROOM = 1.3

    plt.close()
    plt.figure(figsize=(8,10))

    # PLOT PCALX
    plt.subplot(3, 2, 1)
    # make a histogram of deviations in zero crossing delay from expected value
    n, bins, patches = plt.hist((pxDelays - expectedAdcDelay)*1e6)
    # plot a vertical line showing delay deviation at the time of the event
    plt.plot(np.array([1, 1])*(pxDelays[idx] - expectedAdcDelay)*1e6,
             [0, np.ceil(max(n)*HEADROOM)], 'r-')
    plt.title('ADC timestamp offset, GPS={}+/-5min'.format(eGPS),
              fontsize='small', y=1.05)
    plt.legend(('Event time', '{} pcalx'.format(IFO)), fontsize='small')
    plt.xlabel('Deviation from Expected Delay (microsec)', fontsize='small')
    plt.tick_params(labelsize='small')
    plt.xticks(rotation=ROTATION_ANGLE)
    print n, bins, patches

    # PLOT PCALY
    plt.subplot(3, 2, 3)
    # make a histogram of deviations in zero crossing delay from expected value
    n, bins, patches = plt.hist((pyDelays - expectedAdcDelay)*1e6)
    # plot a vertical line showing delay deviation at the time of the event
    plt.plot(np.array([1, 1])*(pyDelays[idx] - expectedAdcDelay)*1e6,
             [0, np.ceil(max(n)*HEADROOM)], 'r-')
    plt.title('ADC timestamp offset, GPS={}+/-5min'.format(eGPS),
              fontsize='small', y=1.05)
    plt.legend(('Event time', '{} pcaly'.format(IFO)), fontsize='small')
    plt.xlabel('Deviation from Expected Delay (microsec)', fontsize='small')
    plt.tick_params(labelsize='small')
    plt.xticks(rotation=ROTATION_ANGLE)
    print n, bins, patches

    # PLOT OMC
    plt.subplot(3, 2, 5)
    # make a histogram of deviations in zero crossing delay from expected value
    n, bins, patches = plt.hist((omcDelays - expectedAdcDelay)*1e6)
    # plot a vertical line showing delay deviation at the time of the event
    plt.plot(np.array([1, 1])*(omcDelays[idx] - expectedAdcDelay)*1e6,
             [0, np.ceil(max(n)*HEADROOM)], 'r-')
    plt.title('ADC timestamp offset, GPS={}+/-5min'.format(eGPS),
              fontsize='small', y=1.05)
    plt.legend(('Event time', '{} omc'.format(IFO)), fontsize='small')
    plt.xlabel('Deviation from Expected Delay (microsec)', fontsize='small')
    plt.tick_params(labelsize='small')
    plt.xticks(rotation=ROTATION_ANGLE)
    print n, bins, patches

    # PLOT PCALX DAC
    plt.subplot(3, 2, 2)
    # make a histogram of deviations in zero crossing delay from expected value
    n, bins, patches = plt.hist((pxRtAiAaDelays - expectedRoundTrip)*1e6)
    # plot a vertical line showing delay deviation at the time of the event
    plt.plot(np.array([1, 1])*(pxRtAiAaDelays[idx] - expectedRoundTrip)*1e6,
             [0, np.ceil(max(n)*HEADROOM)], 'r-')
    plt.title('DAC timestamp offset, GPS={}+/-5min'.format(eGPS),
              fontsize='small', y=1.05)
    plt.legend(('Event time', '{} pcalx'.format(IFO)), fontsize='small')
    plt.xlabel('Deviation from Expected Delay (microsec)', fontsize='small')
    plt.tick_params(labelsize='small')
    plt.xticks(rotation=ROTATION_ANGLE)
    print n, bins, patches

    # PLOT PCALY DAC
    plt.subplot(3, 2, 4)
    # make a histogram of deviations in zero crossing delay from expected value
    n, bins, patches = plt.hist((pyRtAiAaDelays - expectedRoundTrip)*1e6)
    # plot a vertical line showing delay deviation at the time of the event
    plt.plot(np.array([1, 1])*(pyRtAiAaDelays[idx] - expectedRoundTrip)*1e6,
             [0, np.ceil(max(n)*HEADROOM)], 'r-')
    plt.title('DAC timestamp offset, GPS={}+/-5min'.format(eGPS),
              fontsize='small', y=1.05)
    plt.legend(('Event time', '{} pcaly'.format(IFO)), fontsize='small')
    plt.xlabel('Deviation from Expected Delay (microsec)', fontsize='small')
    plt.tick_params(labelsize='small')
    plt.xticks(rotation=ROTATION_ANGLE)
    print n, bins, patches

    # layout and save figure
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('duotone_stat_plots_{}_{}.png'.format(IFO, int(eGPS)))

def iopDecimDelay(IFO, f, gpstime):
    """Directly translated from Keita's MATLAB function of the same name.
    Docstring copied from MATLAB function.
    function delay=iopDecimDelay(IFO, frequency, gpstime)
    Returns the delay (in seconds) of IOP 4x decimation filter
    i.e. from 64k to 16k
    for IFO ('L1' or 'H1') at frequency=f in seconds.
    Since decimation filter can be changed (and was changed after O1),
    this function has a hard coded table of the decimation filter VS epoch.
    If gpstime is omitted, the latest filter is used, otherwise a filter
    corresponding to the gpstime will be used.
    epoch boundary in the table is far from exact but is good enough to identify 
    the correct filter in any observing runs."""

    fs=2**16
    
    # Table of sos coefficients
    # First observing run O1, see e.g. RCG 2.9.7, src/fe/controller.c for
    # binilear filter coefficients for feCoeff4x.
    # https://redoubt.ligo-wa.caltech.edu/websvn/filedetails.php?repname=advLigoRTS&path=%2Ftags%2FadvLigoRTS-2.9.7%2Fsrc%2Ffe%2Fcontroller.c
    epoch = [{},{}]
    epoch[0]['g']   = 0.014805052402446 # gain
    epoch[0]['a11'] = np.array([[ 0.7166258547451800], [ 0.6838596423885499]])
    epoch[0]['a12'] = np.array([[-0.0683289874517300], [-0.2534855521841101]])
    epoch[0]['c1']  = np.array([[ 0.3031629575762000], [ 1.6838609161411500]])
    epoch[0]['c2']  = np.array([[ 0.5171469569032900], [ 1.7447155374502499]])
    # sos(1) filter is valid for sos(1).ts<=gpstime<sos(2).ts
    epoch[0]['ts']  = -np.inf
    
    # After O1, from May 05 2016
    # see e.g. RCG 3.0.0, src/fe/controller.c for bilinear filter coefficients
    # for feCoeff4x.
    # https://redoubt.ligo-wa.caltech.edu/websvn/filedetails.php?repname=advLigoRTS&path=%2Ftags%2FadvLigoRTS-3.0%2Fsrc%2Ffe%2Fcontroller.c
    epoch[1]['g']   = 0.054285975
    epoch[1]['a11'] = np.array([[0.3890221], [0.52191125]])
    epoch[1]['a12'] = np.array([[-0.17645085], [-0.37884382]])
    epoch[1]['c1']  = np.array([[-0.0417771600000001], [1.52190741336686]])
    epoch[1]['c2']  = np.array([[0.41775916], [1.69347541336686]])

    if IFO.upper() == 'L1':
        # this is Apr/12/2016 13:00:00 UTC, that's Tuesday 8AM local time at LLO
        epoch[1]['ts'] = 1144501217
    elif IFO.upper() == 'H1':
        # this is May/03/2016 15:00:00 UTC, that's Tuesday 8AM local time
        epoch[1]['ts'] = 1146322817
    else:
        raise ValueError('IFO identifier {} is not recognized.'.format(IFO))
        
    # Real work
    # Find the right epoch
    for i in range(len(epoch))[::-1]:
        if gpstime >= epoch[i]['ts']:
            epochid = i
            break
    
    # Make a state space model using sos, and obtain the frequency response.
    # For RCG coefficient definition for biquad IIR filter, see iir_filter_biquad.c
    # in src/include/drv/fm10Gen.c 
    # https://redoubt.ligo-wa.caltech.edu/websvn/filedetails.php?repname=advLigoRTS&path=%2Ftags%2FadvLigoRTS-3.0%2Fsrc%2Finclude%2Fdrv%2Ffm10Gen.c
    # For converting a11, a12, c1 and c2 to a1, a2, b1 and b2, see e.g. https://dcc.ligo.org/DocDB/0006/G0900928/001/G0900928-v1.pdf
    a1     = -1 - epoch[epochid]['a11']
    a2     = -1 - epoch[epochid]['a12'] - a1
    b1     =    + epoch[epochid]['c1']  + a1
    b2     =    + epoch[epochid]['c2']  + a1 + a2 - b1
    filler = np.ones(np.shape(a1))
    sosmtx = np.concatenate([filler, b1, b2, filler, a1, a2], axis=1)
    
    # convert second order states representation to zeroes poles gain; scipy
    # does not take a gain argument in sos2zpk, so just multiply the original
    # gain in with the gain returned by sos2zpk for the zpk2ss conversion.
    zz, pp, kk     = scipy.signal.sos2zpk(sosmtx)
    aa, bb, cc, dd = scipy.signal.zpk2ss(zz, pp, kk*epoch[epochid]['g'])
    ssm = scipy.signal.StateSpace(aa, bb, cc, dd, dt=1/fs)
    delay = -np.angle( ssm.freqresp( [2*np.pi*f/fs] )[1][0] ) / 2 / np.pi / f
    
    return delay

if __name__ == "__main__":
    # should we plot commissioningFrameDuotoneStat?
    if args.stat:
        # make sure IFO and gpstime are included
        if args.ifo is None or args.gpstime is None:
            print('ERROR: Must provide both IFO and gpstime of event.\n')
            print(DESC)
            exit(1)
        print('Making commissioningFrameDuotoneStat plots.')
        commissioningFrameDuotoneStat(args.ifo, args.gpstime)
