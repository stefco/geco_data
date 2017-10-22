#!/usr/bin/env python
# (c) Stefan Countryman, 2017

DESC = """Find all frame files of a given frame type in a given time range on
a remote LIGO server and download them to the specified output directory,
skipping any files that have already been downloaded."""
DEFAULT_H_FRAMETYPES = ['H1_R']
DEFAULT_L_FRAMETYPES = ['L1_R']
DEFAULT_V_FRAMETYPES = []
DEFAULT_SERVER = 'ldas-pcdev2.ligo.caltech.edu'
DEFAULT_OUTDIR = '.'

# all other imports listed after argument parsing, allowing for fast help
# documentation printing.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        help="""
            The starting GPS time for this dump. Will be rounded down to the
            nearest even multiple of 64, since the frame files all start on
            multiples of 64 seconds GPS time.
        """
    )
    parser.add_argument(
        "-d",
        "--deltat",
        type=int,
        help="""
            The length of the time window in which to seek frame files,
            measured in seconds. For example, 86400 seconds is a whole day.
        """
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help="""
            The URL of the server on which to perform a ``gw_data_find`` query
            followed by a frame file download. DEFAULT: {}
            """.format(DEFAULT_SERVER)
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="""
            The default directory in which to save downloaded frame files.
            DEFAULT: {}
            """.format(DEFAULT_OUTDIR)
    )
    parser.add_argument(
        "-H",
        "--hanford-frametypes",
        nargs="*",
        default=DEFAULT_H_FRAMETYPES,
        help="""
            The frametypes to download for the Hanford detector (LHO). DEFAULT:
            {}
            """.format(DEFAULT_H_FRAMETYPES)
    )
    parser.add_argument(
        "-L",
        "--livingston-frametypes",
        nargs="*",
        default=DEFAULT_L_FRAMETYPES,
        help="""
            The frametypes to download for the Livingston detector (LLO).
            DEFAULT: {}
            """.format(DEFAULT_L_FRAMETYPES)
    )
    parser.add_argument(
        "-V",
        "--virgo-frametypes",
        nargs="*",
        default=DEFAULT_V_FRAMETYPES,
        help="""
            The frametypes to download for Virgo. DEFAULT:
            {}
            """.format(DEFAULT_V_FRAMETYPES)
    )
    args = parser.parse_args()

import numpy as np
import subprocess

class GWDataFindException(Exception):
    """An error thrown when ``gw_data_find`` on the remote server fails."""

class GWDataDownloadException(Exception):
    """An error thrown when ``gsiscp`` fails to download the desired data."""

class GWFrameQuery(object):
    """An object specifying the detector, frametype, frame start time,
    output filename, output directory, and server where remote data is stored
    for a GW frame.  Used to check if the frame file exists, and, if it
    doesn't, to find the file on a remote server and download it."""
    def __init__(self, detector, frametype, gpstime, outfile,
                 server=DEFAULT_SERVER, outdir=DEFAULT_OUTDIR):
        self.detector   = detector
        self.frametype  = frametype
        self.gpstime    = gpstime
        self.outfile    = outfile
        self.server     = server
        self.outdir     = outdir

    _FILENAME_FORMAT = '{}-{}-{}-64.gwf'

    _GW_DATA_FIND_QUERY_FMT = """gw_data_find \\
        --observatory {} \\
        --type {} \\
        --gps-start-time {} \\
        --gps-end-time {} \\
        --url-type file
    """

    @property
    def filename(self):
        return self._FILENAME_FORMAT.format(
            self.detector,
            self.frametype,
            self.gpstime
        )

    @property
    def fullpath(self):
        return os.path.join(self.outdir, self.filename)

    def exists(self):
        """Has this frame file already been downloaded?"""
        return os.path.isfile(self.fullpath)

    def remote_path(self):
        """Get the path to this frame file on the remote server."""
        query = self._GW_DATA_FIND_QUERY_FMT.format(
            self.detector,
            self.frametype,
            self.gpstime,
            self.gpstime
        )
        cmd = ['gsissh', self.server, query]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        res, err = proc.communicate()
        if proc.returncode != 0:
            raise GWDataFindException("Something went wrong: {}".format(err))
        return res.strip().replace('file://localhost', '')

    def download(self):
        """Download the file specified in ``self.remote_path()`` from the
        remote server."""
        remote_url = '{}:{}'.format(self.server, self.remote_path())
        cmd = ['gsiscp', remote_url, self.fullpath]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        res, err = proc.communicate()
        if proc.returncode != 0:
            raise GWDataDownloadException(
                "Something went wrong: {}".format(err)
            )

def get_times(start, deltat):
    """Get a list of start times for frame files based on in initial starting
    time, ``start``, and a specified length of time, ``deltat``. The initial
    starting time will be rounded down to the nearest multiple of 64, the
    default length of a frame file, since these are the customary starting
    times for LVC frame files. Similarly, the time window ``deltat`` will be
    rounded up to the nearest multiple of 64."""
    start  = int(np.floor(start // 64)) * 64
    deltat = int(np.ceil( start // 64)) * 64
    return range(start, start+deltat+64, 64)

def get_queries(

def main(args):
    times = get_times(args.start, args.deltat)

if __name__ == "__main__":
    main(args)
