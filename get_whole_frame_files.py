#!/usr/bin/env python
# (c) Stefan Countryman, 2017

DESC = """Find all frame files of a given frame type in a given time range on
a remote LIGO server and download them to the specified output directory,
skipping any files that have already been downloaded."""
DEFAULT_H_FRAMETYPES = ['H1_R']
DEFAULT_L_FRAMETYPES = ['L1_R']
DEFAULT_V_FRAMETYPES = []
DEFAULT_FRAME_LENGTH = 64
DEFAULT_SERVER = 'ldas-pcdev2.ligo.caltech.edu'
DEFAULT_OUTDIR = '.'

# all other imports listed after argument parsing, allowing for fast help
# documentation printing.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument(
        "-t",
        "--start",
        required=True,
        type=int,
        help="""
            The starting GPS time for this dump. Will be rounded down to the
            nearest even multiple of ``--length``, since the frame files all
            start on multiples of ``--length`` seconds GPS time.
        """
    )
    parser.add_argument(
        "-l",
        "--length",
        default=DEFAULT_FRAME_LENGTH,
        help="""
            The duration of each frame file. The script is dumb and will only
            try to find files assuming this length. DEFAULT: {}
        """.format(DEFAULT_FRAME_LENGTH)
    )
    parser.add_argument(
        "-d",
        "--deltat",
        required=True,
        type=int,
        help="""
            The length of the time window in which to seek frame files,
            measured in seconds. For example, 86400 seconds is a whole day.
        """
    )
    parser.add_argument(
        "-s",
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
import collections
import subprocess
import datetime
import os

class GWDataFindException(Exception):
    """An error thrown when ``gw_data_find`` on the remote server fails."""

class GWDataDownloadException(Exception):
    """An error thrown when ``gsiscp`` fails to download the desired data."""

class GWRemoteSha256Exception(Exception):
    """An error thrown when ``sha256sum`` fails on the remote server."""

class GWLocalSha256Exception(Exception):
    """An error thrown when ``sha256sum`` fails locally."""

class RemoteFileInfo(object):
    """A container holding data about a remote frame file (based on its
    filename as returned by ``gw_data_find``) along with convenience methods
    for generating a proper local file name.
    
    It is assumed that URLs returned by ``gw_data_find`` look like:
    
    file://localhost/hdfs/frames/O2/hoft_C02/H1/H-H1_HOFT_C02-11869/H-H1_HOFT_C02-1186959360-4096.gwf

    """
    def __init__(self, gw_data_find_response):
        """Initialize using the ``gw_data_find`` file URL string. Strips
        surrounding whitespace from the remote filename."""
        self.gw_data_find_response = gw_data_find_response.strip()
    FILE_URL_PREFIX = "file://localhost"
    @property
    def fullpath(self):
        """Return the full path on the remote server (removing the
        "file://localhost" prefix from the response string)."""
        return self.gw_data_find_response.replace(FILE_URL_PREFIX, '')
    @property
    def filename(self):
        """Get the remote filename without the containing directory."""
        return os.path.basename(self.fullpath)
    @property
    def gps_start_time(self):
        """Get the GPS start time of this frame file."""
        return self.filename.split('.')[0].split('-')[2]
    @property
    def frame_duration(self):
        """Get the duration in seconds of this frame file."""
        return self.filename.split('.')[0].split('-')[3]

class GWFrameQuery(object):
    """An object specifying the detector, frametype, frame start time, output
    directory, and server where remote data is stored for a GW frame.  Used to
    check if the frame file exists, and, if it doesn't, to find the file on a
    remote server and download it."""
    def __init__(self, detector, frametype, gpstime,
                 framelength=DEFAULT_FRAME_LENGTH, server=DEFAULT_SERVER,
                 outdir=DEFAULT_OUTDIR):
        self.detector    = detector
        self.frametype   = frametype
        self.gpstime     = gpstime
        self.framelength = framelength
        self.server      = server
        self.outdir      = outdir

    _FILENAME_FORMAT = '{}-{}-{}-{}.gwf'

    _GW_DATA_FIND_QUERY_FMT = """gw_data_find \\
        --observatory {} \\
        --type {} \\
        --gps-start-time {} \\
        --gps-end-time {} \\
        --url-type file
    """

    _SHA256_SUM_FMT = "sha256sum '{}'"

    @property
    def estimated_filename(self):
        """What the filename *should* be assuming that the frame length of
        each frame file on the *remote server* is the same as the
        ``framelength`` specified in this query object."""
        return self._FILENAME_FORMAT.format(
            self.detector,
            self.frametype,
            self.gpstime,
            self.framelength
        )

    @property
    def estimated_fullpath(self):
        """Again, what the full path *should* be. See
        ``estimated_filename``."""
        return os.path.join(self.outdir, self.estimated_filename)

    def estimated_fullpath_exists(self):
        """Has this frame file already been downloaded? Do a naive check for
        what we *think* the filename is, though this might actually be
        different if the frames on the remote server have different frame
        lengths than what we estimated for this query in ``framelength``."""
        return os.path.isfile(self.estimated_fullpath)

    def local_filename_from_remote(self, remote_url):
        """Get the local filename based on the filename of the remote URL. This
        might not be the expected filename, so we need to check."""
        remote_file_info = RemoteFileInfo(remote_url)
        return self._FILENAME_FORMAT.format(
            self.detector,
            self.frametype,
            remote_file_info.gps_start_time,
            remote_file_info.frame_duration
        )

    def local_fullpath_from_remote(self, remote_url):
        """Get the local full path based on the filename of the remote URL.
        This might not be the expected full path, so we need to check."""
        return os.path.join(
            self.outdir,
            self.local_filename_from_remote(remote_url)
        )

    LOCAL_RIDER_TYPES = [
        'remote_sha256',
        'local_sha256',
        'query_repr',
        'error_msg',
        'remote_url'
    ]
    RIDER_FORMAT = '.{}.{}.txt'
    LocalRiders = collections.namedtuple('LocalRiders', LOCAL_RIDER_TYPES)
    LocalRiders.__doc__ = """
        A container for rider file paths specifying metadata about our
        downloads including remote URL and remote/local sha256 sums as well as
        information on GWFrameQuery instance used and any error messages
        generated while downloading data.
    """

    def local_rider_fullpaths_from_remote(self, remote_url):
        """Get a ``LocalRiders`` namedtuple specifying the file paths to
        rider files for this query. These files contain metadata about the
        remote download, specifically, the remote and local sha256 sums, the
        file query used to generate the file, and the remote_url originally
        returned by gw_data_find."""
        actual_local_filename = self.local_filename_from_remote(remote_url)
        rider_filenames = [
            self.RIDER_FORMAT.format(
                actual_local_filename,
                rider_type
            ) for rider_type in self.LOCAL_RIDER_TYPES
        ]
        return self.LocalRiders(*rider_filenames)

    def local_fullpath_from_remote_exists(self, remote_url):
        """Check whether the local file corresponding to the remote_url exists.
        The remote_url might have an unexpected filename due to e.g. differing
        frame durations, so we need to check this."""
        return os.path.isfile(self.local_fullpath_from_remote(remote_url))

    def remote_url(self):
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

    def remote_sha256(self, remote_url=None):
        """Get the sha256 sum for the file specified in ``self.remote_url()``
        and write it to a rider file (if the rider file does not already
        exist). Optionally override the ``remote_url`` argument, for example if
        the remote URL (as returned by ``gw_data_find``) has already been
        fetched."""
        # if no remote URL specified, find it automatically
        if remote_url == None:
            remote_url = self.remote_url()
        remote_fullpath = RemoteFileInfo(remote_url).fullpath
        sha256cmd = self._SHA256_SUM_FMT.format(remote_fullpath)
        cmd = ['gsissh', self.server, sha256cmd]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        res, err = proc.communicate()
        if proc.returncode != 0:
            errtime = datetime.datetime.utcnow().isoformat()
            errfmt = "REMOTE_SHA256 ERROR at {}. STDERR: \n{}\n"
            errmsg = errfmt.format(errtime, err)
            raise GWRemoteSha256Exception(errmsg)
        return res.split()[0]

    def local_sha256(self, remote_url=None):
        """Get the sha256 sum for the *local* file specified by
        ``self.remote_url()`` and write it to a rider file (if the rider file
        does not already exist). Optionally override the ``remote_url``
        argument, for example if the remote URL (as returned by
        ``gw_data_find``) has already been fetched."""
        # if no remote URL specified, find it automatically
        if remote_url == None:
            remote_url = self.remote_url()
        fullpath = self.local_fullpath_from_remote(remote_url)
        cmd = ['sha256sum', fullpath]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        res, err = proc.communicate()
        if proc.returncode != 0:
            errtime = datetime.datetime.utcnow().isoformat()
            errfmt = "LOCAL_SHA256 ERROR at {}. STDERR: \n{}\n"
            errmsg = errfmt.format(errtime, err)
            raise GWLocalSha256Exception(errmsg)
        return res.split()[0]

    def download(self):
        """Download the file specified in ``self.remote_url()`` from the
        remote server. The remote file might actually have a different filename
        than what is expected, particularly if the user has incorrectly
        guessed the frame duration, so an extra check is made to see if the
        local filename differs. Also write the ``self.remote_url()`` to a rider
        file for future reference."""
        remote_url = self.remote_url()
        # only download the file if it does not exist locally.
        if not self.local_fullpath_from_remote_exists(remote_url):
            download_url = '{}:{}'.format(self.server, remote_url)
            riders = self.local_rider_fullpaths_from_remote(remote_url)
            # record the remote url for debugging and record-keeping
            with open(riders.remote_url, 'w') as f:
                f.write(remote_url)
            # record a representation of this query
            with open(riders.query_repr, 'w') as f:
                f.write(repr(self))
            cmd = [
                'gsiscp',
                download_url,
                self.local_fullpath_from_remote(remote_url)
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            res, err = proc.communicate()
            if proc.returncode == 0:
                # get the remote sha256 sum
                try:
                    remote_sha256 = self.remote_sha256(remote_url)
                    with open(riders.remote_sha256, 'w') as f:
                        f.write(remote_sha256)
                except GWRemoteSha256Exception as e:
                    with open(riders.error_msg, 'a') as f:
                        f.write(e.args[0])
                    raise e
                # get the local sha256 sum
                try:
                    local_sha256 = self.local_sha256(remote_url)
                    with open(riders.local_sha256, 'w') as f:
                        f.write(local_sha256)
                except GWLocalSha256Exception as e:
                    with open(riders.error_msg, 'a') as f:
                        f.write(e.args[0])
                    raise e
            else:
                errtime = datetime.datetime.utcnow().isoformat()
                errfmt = "DOWNLOAD ERROR at {}. STDERR: \n{}\n"
                errmsg = errfmt.format(errtime, err)
                with open(riders.error_msg, 'a') as f:
                    f.write(errmsg)
                raise GWDataDownloadException(errmsg)

    def __repr__(self):
        fmt="{}('{}', '{}', '{}', framelength='{}', server='{}', outdir='{}')"
        return fmt.format(
            type(self).__name__,
            self.detector,
            self.frametype,
            self.gpstime,
            self.framelength,
            self.server,
            self.outdir
        )

def get_times(start, deltat, frlength):
    """Get a list of start times for frame files based on in initial starting
    time, ``start``, and a specified length of time, ``deltat``. The initial
    starting time will be rounded down to the nearest multiple of
    ``frlength``, the default length of a frame file, since these are
    the customary starting times for LVC frame files. Similarly, the time
    window ``deltat`` will be rounded up to the nearest multiple of
    ``frlength``."""
    start  = ( 
        int(np.floor(start // frlength)) * frlength
    )
    deltat = (
        int(np.ceil(deltat // frlength)) * frlength
    )
    return range(
        start,
        start + deltat + frlength,
        frlength
    )

def get_queries(
        start,
        deltat,
        length=DEFAULT_FRAME_LENGTH,
        server=DEFAULT_SERVER,
        outdir=DEFAULT_OUTDIR,
        h_frametypes=DEFAULT_H_FRAMETYPES,
        l_frametypes=DEFAULT_L_FRAMETYPES,
        v_frametypes=DEFAULT_V_FRAMETYPES
):
    """Get all GWFrameQuery objects in the given time window for each
    combination of detector and frametype provided.

    Args:

        ``start``       The GPS time at which the time window starts.
        ``deltat``      The width of the time window in seconds.
        ``server``      The server on which to look for frame files.
        ``outdir``      The output directory where downloaded files should be
                        saved.

    Additionally, one can specify the frame file types to be downloaded for
    each detector:

        ``h_frametypes``    Frametypes for LIGO Hanford.
        ``l_frametypes``    Frametypes for LIGO Livingston.
        ``v_frametypes``    Frametypes for Virgo.
    """
    queries = list()
    # add queries for each detector/frametype combination
    detector_frametypes_dict = {
        "H": h_frametypes,
        "L": l_frametypes,
        "V": v_frametypes
    }
    for detector in detector_frametypes_dict.keys():
        for frametype in detector_frametypes_dict[detector]:
            queries += [GWFrameQuery(detector, frametype, t, length=length,
                                     server=server, outdir=outdir)
                        for t in get_times(start, deltat, length)]
    return queries

def main(args):
    queries = get_queries(
        args.start,
        args.deltat,
        args.length,
        server=args.server,
        outdir=args.outdir,
        h_frametypes=args.hanford_frametypes,
        l_frametypes=args.livingston_frametypes,
        v_frametypes=args.virgo_frametypes
    )
    for query in queries:
        if not query.estimated_fullpath_exists():
            query.download()

if __name__ == "__main__":
    main(args)
