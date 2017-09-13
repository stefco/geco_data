#!/usr/bin/env python
# (c) Stefan Countryman 2017

DESC="""Run Timing checks and save results to a LIGO viewable webpage. If
--gpstime or --graceid are not provided, read them from 'gpstime.txt' or
'graceid.txt', respectively."""
DEFAULT_EVENT_DIR_PREFIX = "~/public_html/events/"

# run argparse before imports so that we don't waste the user's time
# with imports if they are just seeking --help; imports come after
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument(
        '-g',
        '--graceid',
        default=None,
        help="""
             The GraceDB ID for this event. You can either specify this as a
             command line argument or store a file called "graceid.txt" in
             the current directory.
             """
    )
    parser.add_argument(
        '-t',
        '--gpstime',
        type=int,
        default=None,
        help="""
             The GPS time at which this event occured. You can either specify
             this as a command line argument or store a file called
             "gpstime.txt" in the current directory. If no GPS time is
             provided, the script will attempt to get the GPS time from
             GraceDB.
             """
    )
    parser.add_argument(
        '-n',
        '--dccnum',
        default=None,
        help="""
             The DCC number for the Timing Witness document. Not required,
             but without it, the output PDF will not cointain a DCC number
             and will only have a placeholder instead.
             """
    )
    parser.add_argument(
        '-p',
        '--eventdir-prefix',
        default=DEFAULT_EVENT_DIR_PREFIX,
        help="""
             The default place where event directories go. Defaults to: {}
             """.format(DEFAULT_EVENT_DIR_PREFIX)
    )
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help="""
             Print debug information.
             """
    )
    args = parser.parse_args()

    def read_arg_from_file_if_possible(args, key, required=False):
        """Read values from text files if args not provided. Error out and
        print help string if the arg is required but is not found anywhere."""
        if getattr(args, key) is None:
            try:
                with open(key + '.txt') as infile:
                    setattr(args, key, infile.read())
            except IOError:
                if required:
                    parser.print_help()
                    exit(1)

    # try to read in required arguments if they weren't provided
    read_arg_from_file_if_possible(args, 'graceid', required=True)
    read_arg_from_file_if_possible(args, 'gpstime')
    read_arg_from_file_if_possible(args, 'dccnum')

    # try to get the gps time from gracedb if it was not provided
    if args.gpstime is None:
        import ligo.gracedb.rest
        client = ligo.gracedb.rest.GraceDb()
        print('Reading GPS time from GraceDb...')
        event = client.event(args.graceid).json()
        args.gpstime = int(event['gpstime'])
        print('Got GPS time: {}'.format(args.gpstime))

    # make sure gpstime is an integer
    args.gpstime = int(args.gpstime)

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
GECO_DATA_DIR = os.path.dirname(os.path.realpath(__file__))

# how long to wait before trying to make these files?
SEC_BEFORE_IRIG = 60
SEC_BEFORE_DTONE = 60 * 12
SEC_BEFORE_OVERLAY = 60 * 18

# What file patterns do we expect each script to produce? If they exist,
# the script succeeded
GLOBS_FOR_IRIGB = [
    '*_H1_CAL-PCALX_IRIGB_OUT_DQ.png',
    '*_H1_CAL-PCALY_IRIGB_OUT_DQ.png',
    '*_L1_CAL-PCALX_IRIGB_OUT_DQ.png',
    '*_L1_CAL-PCALY_IRIGB_OUT_DQ.png',
    '*-decoded-times.txt'
]
GLOBS_FOR_DTONE = [
    'duotone_stat_plots_H1_*.png',
    'duotone_stat_plots_L1_*.png'
]
GLOBS_FOR_OVERLAY = [
    'H1..CAL-PCALX_IRIGB_OUT_DQ-Overlay-*.png',
    'H1..CAL-PCALY_IRIGB_OUT_DQ-Overlay-*.png',
    'L1..CAL-PCALX_IRIGB_OUT_DQ-Overlay-*.png',
    'L1..CAL-PCALY_IRIGB_OUT_DQ-Overlay-*.png',
    'H1..CAL-PCALX_FPGA_DTONE_IN1_DQ-Overlay-*.png',
    'H1..CAL-PCALY_FPGA_DTONE_IN1_DQ-Overlay-*.png',
    'L1..CAL-PCALX_FPGA_DTONE_IN1_DQ-Overlay-*.png',
    'L1..CAL-PCALY_FPGA_DTONE_IN1_DQ-Overlay-*.png'
]

# What file patterns does each script need? If they exist, the script can
# be made.
IN_GLOBS_FOR_PDF = GLOBS_FOR_IRIGB + GLOBS_FOR_DTONE + GLOBS_FOR_OVERLAY

# how long to wait before retrying?
SLEEP_TIME = 10

class NDS2AvailabilityException(IOError):
    """An exception indicating that you are trying to download data before it
    is likely to be available on NDS2."""


class JobDoneException(IOError):
    """An exception indicating that the output files already exist and that
    the job should not run again."""

class InputFilesUnavailableException(IOError):
    """An exception indicating that the input files required for this job
    are unavailable."""


class ProcRunner(object):
    """Run processes silently and asynchronously. Check for successful
    completion.
    """
    def __init__(self, cmd, desc=None, execpath="", muteout=False,
                 muteerr=False):
        self.cmd = cmd
        self.desc = desc
        self.execpath = execpath
        self.muteout = muteout
        self.muteerr = muteerr
        self.proc = None
    def run(self):
        """Run a system command silently in the background. This ProcRunner
        is returned, making it easy to chain commands."""
        import os
        import subprocess
        self.cmd[0] = os.path.join(self.execpath, self.cmd[0])
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return self
    def complete(self):
        """Wait for this process to complete. Return true if successful.
        If it failed, raise an exception including full stderr and stdout
        (unless either or both are muted)."""
        res, err = self.proc.communicate()
        if self.proc.returncode != 0:
            errdesc = ""
            if not self.muteout:
                errdesc += 'STDOUT:\n\n{}\n\n'.format(res)
            if not self.muteerr:
                errdesc += 'STDERR:\n\n{}\n\n'.format(err)
            if self.desc is None:
                errdesc += "Something went wrong."
            else:
                errdesc += "Something went wrong: {}".format(self.desc)
            raise Exception(errdesc)
        return True


class FileGenerator(object):
    """Check whether files have been generated by looking for matching blob
    expressions in the current directory. If they have not all been
    generated, rerun the generating command. Don't try to generate the files
    at all unless a minimum delay has been exceeded. All commands will be
    run in parallel.
    
    """
    def __init__(self, desc, file_globs, commands, execpath="", min_delay=0,
                 gpstime=None, graceid=None, verbose=True, muteout=False,
                 muteerr=False, input_globs=[], force_rerun=False):
        """Specify the command to run.

        ``desc``        Describes the process.
        ``file_globs``  Is a list of glob expressions for the required files.
                        if all files exist, the generator does not need to run.
        ``commands``    Is a list of commands that can be understood
                        by subprocess.Popen, i.e. each is a list of strings
                        constituting arguments.
        ``execpath``    Path prefix for the executable.
        ``min_delay``   Is the minimum amount of time to wait after the start
                        of the event before trying to make the files.
        ``gpstime``     Is the time of the event for which these files are
                        being made.
        ``graceid``     Is the graceid of the relevant event.
        ``verbose``     Print info to console.
        ``muteout``     Mute stderr for subprocesses.
        ``muteerr``     Mute stderr for subprocesses.
        ``input_globs`` Glob patterns representing filenames for data that
                        needs to exist for this job to run.
        ``force_rerun`` Rerun the job even if the output files exist.
        """
        self.desc = desc
        self.file_globs = file_globs
        self.commands = commands
        self.execpath = execpath
        self.min_delay = min_delay
        self.gpstime = gpstime
        self.graceid = graceid
        self.verbose = verbose
        self.muteout = muteout
        self.muteerr = muteerr
        self.input_globs = input_globs
        self.force_rerun = force_rerun
        self.runners = None
    def _globs_all_exist(self, file_globs):
        """Check whether each glob file exists and is unique."""
        import glob
        if all([len(glob.glob(g)) == 1 for g in file_globs]):
            return True
        else:
            return False
    def isdone(self):
        """Check if the files are generated. Optionally print an indication
        that the files are done (default: True)"""
        return self._globs_all_exist(self.file_globs)
    def infilesready(self):
        """Check if the input files needed for the job are available."""
        return self._globs_all_exist(self.input_globs)
    def ready(self):
        """Check if enough time has passed for us to try making these
        files."""
        import datetime
        import gwpy
        now = gwpy.time.tconvert(datetime.datetime.utcnow()).gpsSeconds
        return now - self.gpstime > self.min_delay
    def run(self):
        """Start running processes and set self.runners to be the list of
        ProcRunners currently running. Raises an NDS2AvailabilityException
        if the file is not yet ready to be made."""
        if not self.ready():
            raise NDS2AvailabilityException("Not ready: {}".format(self.desc))
        if self.isdone() and (not self.force_rerun):
            raise JobDoneException("Out files exist: {}".format(self.desc))
        if not self.infilesready():
            raise InputFilesUnavailableException(
                "Input files not ready: {}".format(self.desc)
            )
        self.runners = [
            ProcRunner(
                cmd = cmd,
                desc = self.desc,
                execpath = self.execpath,
                muteout = self.muteout,
                muteerr = self.muteerr
            ).run()
            for cmd in self.commands
        ]
    def wait(self):
        """Wait for all running processes to finish. If not processes have
        been started, raise a ValueError. Return ``True`` if all finished."""
        if self.runners is None:
            raise ValueError("No runners yet: {}".format(self.desc))
        finished = all([runner.complete() for runner in self.runners])
        if finished and self.verbose:
            print('Done with: {}'.format(self.desc))
        return finished


def start_as_data_becomes_available(file_generators, debug=False):
    """Start each generator as the data becomes available on NDS2. Returns
    once all of them are ready."""
    running_generators = []
    loops = 0
    while len(file_generators) > 0:
        if debug:
            print(loops)
            print("[File Generators:] {}".format(file_generators))
        loops += 1
        print('Time: {}'.format(datetime.datetime.now()))
        for fg in file_generators:
            if debug:
                print('On generator: {}'.format(fg.desc))
            try:
                if debug:
                    print('Trying to start: {}'.format(fg.desc))
                fg.run()
                if debug:
                    print('Generator started: {}'.format(fg.desc))
                running_generators.append(fg)
                file_generators.remove(fg)
            except NDS2AvailabilityException:
                if debug:
                    print('Generator not ready: {}'.format(fg.desc))
            except JobDoneException:
                if debug:
                    print('Files done, not running: {}'.format(fg.desc))
                file_generators.remove(fg)
            except InputFilesUnavailableException:
                if debug:
                    print('Infiles not ready, not running: {}'.format(fg.desc))
        if debug:
            print('Exiting loop & sleeping {}'.format(datetime.datetime.now()))
        time.sleep(SLEEP_TIME)
    return running_generators

def enter_event_directory(graceid, eventdirpre=DEFAULT_EVENT_DIR_PREFIX,
                          debug=False):
    """Define the directory for this event. Also change to that directory and
    return the path to that directory."""
    eventdir = os.path.expanduser(os.path.join(eventdirpre, graceid))
    print('Starting at {}'.format(datetime.datetime.utcnow().isoformat()))
    if not os.path.isdir(eventdir):
        os.makedirs(eventdir)
    os.chdir(eventdir)
    print('made eventdir and changed to it: {}'.format(eventdir))
    if debug:
        print('current directory: {}'.format(os.getcwd()))
        print('"./": {}'.format(os.path.realpath('.')))
    return eventdir

def main(gpstime, graceid, dccnum=None, eventdirpre=DEFAULT_EVENT_DIR_PREFIX,
         debug=False):
    """Generate missing files asyncronously as they become available."""
    enter_event_directory(graceid, eventdirpre=eventdirpre, debug=debug)

    # make little convenience files containing the GPS time and GraceID
    for varname in ['graceid', 'gpstime']:
        fname = varname + '.txt'
        if not os.path.isfile(fname):
            with open(fname, 'w') as f:
                f.write(str(locals()[varname]))

    # define the file generators and start them.
    print("Defining file generators and starting them up...")
    file_generators = start_as_data_becomes_available(
        [
            FileGenerator(
                desc = "IRIG-B decode checks and plots",
                file_globs = GLOBS_FOR_IRIGB,
                commands = [
                    ['irig-b-decode-commands.py', '-t', str(gpstime),
                    '-g', graceid]
                ],
                execpath = GECO_DATA_DIR,
                min_delay = SEC_BEFORE_IRIG,
                gpstime = gpstime,
                graceid = graceid
            ),
            FileGenerator(
                desc = "DuoTone Delay Plots",
                file_globs = GLOBS_FOR_DTONE,
                commands = [
                    ['duotone_delay.py', '--stat', '--ifo', 'H1',
                    '-t', str(gpstime)],
                    ['duotone_delay.py', '--stat', '--ifo', 'L1',
                    '-t', str(gpstime)]
                ],
                execpath = GECO_DATA_DIR,
                min_delay = SEC_BEFORE_DTONE,
                gpstime = gpstime,
                graceid = graceid
            ),
            FileGenerator(
                desc = "DuoTone/IRIG-B Overlay Plots",
                file_globs = GLOBS_FOR_OVERLAY,
                commands = [
                    ['geco_overlay_plots.py', '-t', str(gpstime)]
                ],
                execpath = GECO_DATA_DIR,
                min_delay = SEC_BEFORE_OVERLAY,
                gpstime = gpstime,
                graceid = graceid
            ),
            FileGenerator(
                desc = "Timing Witness PDF document",
                file_globs = [],
                input_globs = IN_GLOBS_FOR_PDF,
                commands = [
                    ['timing_witness_paper.py', graceid, str(gpstime), '.']
                ],
                execpath = GECO_DATA_DIR,
                muteout = True,
                muteerr = True,
                gpstime = gpstime,
                graceid = graceid,
                force_rerun = True
            )
        ],
        debug = debug
    )

    # wait for everything to finish. since we are waiting for all processes
    # to finish and since we don't need to free up any resources or anything
    # like that, it suffices to just wait on them in order.
    for fg in file_generators:
        fg.wait()


if __name__ == '__main__':
    main(
        gpstime = args.gpstime,
        graceid = args.graceid,
        dccnum = args.dccnum,
        eventdirpre = args.eventdir_prefix,
        debug = args.debug
    )
