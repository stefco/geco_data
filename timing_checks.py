#!/usr/bin/env python
# (c) Stefan Countryman 2017

DESC="""Run Timing checks and save results to a LIGO viewable webpage."""

# run argparse before imports so that we don't waste the user's time
# with imports if they are just seeking --help; imports come after
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument('-g', '--graceid', required=True,
                        help=('The GraceDB ID for this event.'))
    parser.add_argument('-t', '--gpstime', type=int, required=True,
                        help=('The GPS time at which this event occured.'))
    parser.add_argument('-d', '--debug', action='store_true',
                        help=('Print debug information.'))
    args = parser.parse_args()
    if args.debug:
        print(format(args))
import glob
import sys
import os
import shutil
import datetime
import subprocess
import gwpy.time
import time

# location of geco_data scripts
geco_data_dir = os.path.dirname(os.path.realpath(__file__))

# how long to wait before trying to make these files?
SEC_BEFORE_IRIG = 60
SEC_BEFORE_DTONE = 60 * 12
SEC_BEFORE_OVERLAY = 60 * 18

# how long to wait before retrying?
SLEEP_TIME = 10

class NDS2AvailabilityException(IOError):
    """An exception indicating that you are trying to download data before it
    is likely to be available on NDS2."""

def try_decoding_irigb(gpstime, graceid, eventdir):
    """Try to make the decoded IRIG-B files for this event. Assumes the output
    event directory exists."""
    print('Trying to generate IRIG-B Decode...')
    # don't try making this before the data is up on NDS2
    now = datetime.datetime.now()
    if gwpy.time.tconvert(now).gpsSeconds - gpstime < SEC_BEFORE_IRIG:
        raise NDS2AvailabilityException()
    # if the files already exist, just return
    file_globs = ['*_H1_CAL-PCALX_IRIGB_OUT_DQ.png',
                  '*_H1_CAL-PCALY_IRIGB_OUT_DQ.png',
                  '*_L1_CAL-PCALX_IRIGB_OUT_DQ.png',
                  '*_L1_CAL-PCALY_IRIGB_OUT_DQ.png',
                  '{}-decoded-times.txt'.format(graceid)]
    # we expect 1 matching file per file pattern
    if all([len(glob.glob(g)) == 1 for g in file_globs]):
        print('IRIG-B Decode plots already exist! Skipping.')
        return True
    command = [os.path.join(geco_data_dir, 'irig-b-decode-commands.py'),
               '-t', str(gpstime), '-g', graceid]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    res, err = proc.communicate()
    if proc.returncode != 0:
        print('STDOUT: {}'.format(res))
        print('STDERR: {}'.format(err))
        raise Exception('Something went wrong generating IRIG-B decode plots.')
    print('Done with IRIG-B Decode.')
    return True

def try_plotting_duotone_delay(gpstime, eventdir):
    """Try to make the DuoTone histogram and delay plots for this event. Assumes
    the output event directory exists."""
    print('Trying to generate Duotone Delay Plots...')
    # don't try making this before the data is up on NDS2
    now = datetime.datetime.now()
    if gwpy.time.tconvert(now).gpsSeconds - gpstime < SEC_BEFORE_DTONE:
        raise NDS2AvailabilityException()
    # if the files already exist, just return
    file_globs = ['duotone_stat_plots_H1_*.png',
                  'duotone_stat_plots_L1_*.png']
    # we expect 1 matching file per file pattern
    if all([len(glob.glob(g)) == 1 for g in file_globs]):
        print('Duotone Delay plots already exist! Skipping.')
        return True
    command_h = [os.path.join(geco_data_dir, 'duotone_delay.py'),
                 '--stat', '--ifo', 'H1', '-t', str(gpstime)]
    command_l = [os.path.join(geco_data_dir, 'duotone_delay.py'),
                 '--stat', '--ifo', 'L1', '-t', str(gpstime)]
    proc_h = subprocess.Popen(command_h, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    proc_l = subprocess.Popen(command_l, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    res_h, err_h = proc_h.communicate()
    res_l, err_l = proc_l.communicate()
    if proc_h.returncode != 0:
        print('STDOUT: {}'.format(res_h))
        print('STDERR: {}'.format(err_h))
        raise Exception('Something went wrong generating H1 duotone delays.')
    if proc_l.returncode != 0:
        print('STDOUT: {}'.format(res_l))
        print('STDERR: {}'.format(err_l))
        raise Exception('Something went wrong generating L1 duotone delays.')
    print('Done with Duotone Delay Plots.')
    return True

def try_making_overlay_plots(gpstime, eventdir):
    """Try to make the DuoTone and IRIG-B overlay plots for this event. Assumes
    the output event directory exists."""
    print('Trying to generate Overlay Plots...')
    # don't try making this before the data is up on NDS2
    now = datetime.datetime.now()
    if gwpy.time.tconvert(now).gpsSeconds - gpstime < SEC_BEFORE_OVERLAY:
        raise NDS2AvailabilityException()
    # if the files already exist, just return
    file_globs = ['H1..CAL-PCALX_IRIGB_OUT_DQ-Overlay-*.png',
                  'H1..CAL-PCALY_IRIGB_OUT_DQ-Overlay-*.png',
                  'L1..CAL-PCALX_IRIGB_OUT_DQ-Overlay-*.png',
                  'L1..CAL-PCALY_IRIGB_OUT_DQ-Overlay-*.png',
                  'H1..CAL-PCALX_FPGA_DTONE_IN1_DQ-Overlay-*.png',
                  'H1..CAL-PCALY_FPGA_DTONE_IN1_DQ-Overlay-*.png',
                  'L1..CAL-PCALX_FPGA_DTONE_IN1_DQ-Overlay-*.png',
                  'L1..CAL-PCALY_FPGA_DTONE_IN1_DQ-Overlay-*.png']
    # we expect 1 matching file per file pattern
    if all([len(glob.glob(g)) == 1 for g in file_globs]):
        print('Overlay plots already exist! Skipping.')
        return True
    command = [os.path.join(geco_data_dir, 'geco_overlay_plots.py'),
               '-t', str(gpstime)]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    res, err = proc.communicate()
    if proc.returncode != 0:
        print('STDOUT: {}'.format(res))
        print('STDERR: {}'.format(err))
        raise Exception('Something went wrong generating overlay plots.')
    print('Done with Overlay Plots.')
    return True

def make_pdf_files(graceid, gpstime, eventdir):
    """Try to make the final output PDF file, i.e. the Timing Witness
    Document that will be uploaded to DCC."""
    print('Trying to make Timing Witness Document PDF...')
    command = [os.path.join(geco_data_dir, 'timing_witness_paper.py'),
               graceid, gpstime, eventdir]
    #proc = subprocess.Popen(command, stdout=subprocess.PIPE,
    #                        stderr=subprocess.PIPE)
    proc = subprocess.Popen(command)
    res, err = proc.communicate()
    if proc.returncode != 0:
        raise Exception('Something went wrong generating the final PDF file.')
    #    print('STDOUT: {}'.format(res))
    #    print('STDERR: {}'.format(err))
    print('Done with Timing Witness Document PDF.')
    return True

def main(gpstime, graceid, debug):
    """Generate missing files."""
    eventdir = os.path.expanduser('~/public_html/events/{}'.format(graceid))
    print('Starting at {}'.format(datetime.datetime.now().isoformat()))
    if not os.path.isdir(eventdir):
        os.makedirs(eventdir)
    os.chdir(eventdir)
    print('made eventdir and changed to it: {}'.format(eventdir))
    if debug:
        print('current directory: {}'.format(os.getcwd()))
    # make little convenience files containing the GPS time and GraceID
    for varname in ['graceid', 'gpstime']:
        fname = varname + '.txt'
        if not os.path.isfile(fname):
            with open(fname, 'w') as f:
                f.write(str(locals()[varname]))
    # try generating files
    done = False
    while not done:
        try:
            done = ( try_decoding_irigb(gpstime, graceid, eventdir) and
                     try_plotting_duotone_delay(gpstime, eventdir) and
                     try_making_overlay_plots(gpstime, eventdir) )
        except NDS2AvailabilityException:
            time.sleep(SLEEP_TIME)
    # try making the LATEX and PDF files
    make_pdf_files(graceid, str(gpstime), eventdir)

if __name__ == '__main__':
    main(args.gpstime, args.graceid, args.debug)
