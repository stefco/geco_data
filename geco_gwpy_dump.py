#! /usr/bin/env python
# (c) Stefan Countryman, Jan 2017

# allowed file extensions for GWPy writing to file, documented at:
# https://gwpy.github.io/docs/v0.1/timeseries/index.html#gwpy.timeseries.TimeSeries.write
NUM_THREADS = 6     # number of parallel download threads
VERBOSE_GWPY = True
ALLOWED_EXTENSIONS = ["csv", "framecpp", "hdf", "hdf5", "txt"]
DEFAULT_EXTENSION = ['hdf5']
# by default, download full data, i.e. no trend extension
DEFAULT_FLAGS = ["H1:DMT-ANALYSIS_READY:1", "L1:DMT-ANALYSIS_READY:1"]
DEFAULT_TRENDS = ['']
# make a dictionary of conversion factors between seconds and other time units
# returned by ``Plotter.t_units``
SEC_PER = {
    "ns": 1e-9,
    "s": 1.,
    "minutes": 60.,
    "days": 86400.
}
# download in 5 minute chunks by default
DEFAULT_MAX_CHUNK = SEC_PER['minutes'] * 5
DEFAULT_PAD = -1.
INDEX_MISSING_FMT = ('{} index not found for segment {} of {}, time {}\n'
                     'Setting {} index to {}.')
USAGE="""
Save channel data in a sane, interruptible, parallelizable way.

Usage (note that multiple option flags are NOT supported):

Start downloading data specified in jobspec.json:

    geco_gwpy_dump

Check incremental progress of download:

    geco_gwpy_dump -p

List final output filenames and whether they exist or not:

    geco_gwpy_dump -o

Archive the output files to a single archive (fails if dump not finished):

    geco_gwpy_dump -a

Unarchive the output files from an existing archive file (fails if archive is
missing or if any expected output files are not in the archive):

    geco_gwpy_dump -u

Unarchive the the jobspec contained in "archive.tar.gz" and safely extract the
output files corresponding to that jobspec from the same archive, running
multiple consistency checks along the way (note: you can skip specifying the
archive filename as long as there is exactly one file in the current directory
with a .tar.gz extension; in this case, that file is assumed to be the correct
archive). Will fail if any of the archived files already exist (including the
jobspec, which will always be called jobspec.json). Will also fail if the
archive filename does not correspond to the canonical output filename (if you
are using custom filenames, you should probably use the ``-X`` option below):

    geco_gwpy_dump -x archive.tar.gz

Like ``-x``, extract the jobspec and output files from the given tarfile, but
don't bother checking the archive filename for consistency with the jobspec
output canonical archive filename. Use this if you have given custom
descriptive names to your archive files.

    geco_gwpy_dump -X archive.tar.gz

Print the filename of the archive for this jobspec and quit (works whether the
archive file exists or not, since this filename is based purely on the
jobspec):

    geco_gwpy_dump -f

Look for a file in the current directory called "jobspec.json", which is
a dictionary containing "start", "end", "channels", and "trends" key-value
pairs. The "start" and "end" values must merely be readable by
gwpy.timeseries.TimeSeries.get. The "channels" value is an array containing
strings of valid ligo epics channels. The "trends" value is an array
containing trend suffixes to use. For full data, use [""]. For all
available minute trends, use

    [
        ".mean,m-trend",
        ".max,m-trend",
        ".min,m-trend",
        ".n,m-trend",
        ".rms,m-trend"
    ]

etc.

If an argument is given, that argument will be interpreted as the jobspec
filepath.

By default, data is downloaded in 5-minute-long chunks (except for the starting
and trailing timespans, which might be shorter). The data spans are
contiguous with no overlap.

If some data cannot be fetched from the server, values of -1 will be used to
pad the final concatenated output files.

Default file output is in {} format. For now, only one file extension can be
specified. All possible file formats:

{}

If full data (rather than minute trends) is required, then the specified trend
should be an empty string. This is the default behavior when no trends are
provided.

""".format(DEFAULT_EXTENSION, ALLOWED_EXTENSIONS) + """
An example jobspec.json file downloading all possible trend extensions for the
minute trends:

{
    "start": "Mon Oct 31 16:00:00 2016",
    "end": "Fri Mar 31 16:00:00 2017",
    "channels": [
        "H1:SYS-TIMING_C_MA_A_PORT_2_SLAVE_CFC_TIMEDIFF_2",
        "H1:SYS-TIMING_C_MA_A_PORT_2_SLAVE_CFC_TIMEDIFF_3",
        "H1:SYS-TIMING_X_FO_A_PORT_9_SLAVE_CFC_TIMEDIFF_3",
        "H1:SYS-TIMING_Y_FO_A_PORT_9_SLAVE_CFC_TIMEDIFF_3",
        "L1:SYS-TIMING_X_FO_A_PORT_9_SLAVE_CFC_TIMEDIFF_2",
        "L1:SYS-TIMING_Y_FO_A_PORT_9_SLAVE_CFC_TIMEDIFF_2",
        "L1:SYS-TIMING_C_MA_A_PORT_2_SLAVE_CFC_TIMEDIFF_1",
        "L1:SYS-TIMING_C_MA_A_PORT_2_SLAVE_CFC_TIMEDIFF_2"
    ],
    "trends": [
        ".mean,m-trend",
        ".rms,m-trend",
        ".max,m-trend",
        ".min,m-trend",
        ".rms,m-trend"
    ],
    "max_chunk_length": 3600
}
"""
# terminal color codes for pretty printing
_GREEN = '\033[92m'
_RED   = '\033[91m'
_CLEAR = '\033[0m'

import sys
# don't import the rest if someone just wants help
if __name__ == '__main__':
    check_progress = False
    list_outfiles = False
    archive_outfiles = False
    unarchive_outfiles = False
    unarchive_job = False
    print_archive_filename = False
    check_archive_filename = True
    if len(sys.argv) != 1 and sys.argv[1] in ['-h', '--help']:
        print(USAGE)
        exit()
    if '-X' in sys.argv:
        check_archive_filename = False
        x_opt_ind = sys.argv.index('-X')
        sys.argv[x_opt_ind] = '-x'
    if '-x' in sys.argv:
        sys.argv.remove('-x')
        unarchive_job = True
    if '-p' in sys.argv:
        sys.argv.remove('-p')
        check_progress = True
    if '-o' in sys.argv:
        sys.argv.remove('-o')
        list_outfiles = True
    if '-a' in sys.argv:
        sys.argv.remove('-a')
        archive_outfiles = True
    if '-u' in sys.argv:
        sys.argv.remove('-u')
        unarchive_outfiles = True
    if '-f' in sys.argv:
        sys.argv.remove('-f')
        print_archive_filename = True
# slow import; only import if we are going to use it.
if not (__name__ == '__main__'
        and (check_progress or list_outfiles)):
    import gwpy.timeseries
    import gwpy.segments
import gwpy.time
import numpy as np
import json
import functools
import hashlib
import tarfile
import tempfile
import glob
import multiprocessing
import math
import os
import logging
import shutil
import datetime

class NDS2Exception(IOError):
    """An error thrown in association with some sort of failure to download
    data."""

def indices_to_intervals(inds):
    """Takes a list of indices and turns it into a list of start/stop indices
    (useful for splitting an array into contiguous subintervals). If there are
    N contiguous subintervals in the list if input indices, returns a
    numpy.ndarray with length 2*N of the format:

    [start1,end1,...,startN,endN]

    >>> indices_to_intervals([0,1,2,3,9,10,11])
    np.array([0,3,9,11])
    """
    # if there are no input indices, just return an empty array
    if len(inds) == 0:
        return np.array([], dtype=int)

    # make sure the input is an np.ndarray
    input_inds = np.array(inds)

    # look for indices in contiguous chunks. find the the indices
    # of the edges of those chunks within the list of non-filler indices.
    change_inds = np.argwhere(input_inds[1:] != input_inds[:-1] + 1).flatten()

    # use list comprehension to flatten the list of ends/starts; these are
    # still indices into the list of nonfiller indices rather than indices into
    # the full timeseries itself.
    inner_inds = [ ind for subint in [ [i,i+1] for i in change_inds ]
                       for ind in subint ]
    all_inds = np.concatenate([[0], inner_inds, [-1]]).astype(int)
    intervals = input_inds[all_inds]
    return intervals

class Query(object):
    """A channel and timespan for a single NDS query and save operation."""
    def __init__(self, start, end, channel, ext):
        self.start      = start
        self.end        = end
        self.channel    = str(channel)
        self.ext        = ext
    @property
    def sanitized_channel(self):
        """get the channel name as used in filenames, i.e. with commas (,) and
        colons (:) replaced in order to fit filename conventions."""
        return sanitize_for_filename(self.channel)
    @property
    def fname(self):
        """get a filename for this particular timespan and channel"""
        return "{}__{}__{}.{}".format(self.start, self.end,
                                      self.sanitized_channel, self.ext)
    @property
    def fname_err(self):
        """get the filename for any error output produced when failing to download
        this channel to this file for this particular timespan."""
        return self.fname + ".ERROR"
    def file_exists(self):
        """see if this file exists."""
        return os.path.isfile(self.fname)
    def query_failed(self):
        """check if this query failed by seeing if an fname_err file exists."""
        return os.path.isfile(self.fname_err)
    def get(self, **kwargs):
        """Fetch the timeseries corresponding to this Query from NDS2 or from
        frame files using GWpy."""
        return gwpy.timeseries.TimeSeries.get(self.channel, self.start,
                                              self.end, pad=DEFAULT_PAD,
                                              verbose=VERBOSE_GWPY,
                                              **kwargs)
    def fetch(self, **kwargs):
        """Get the timeseries corresponding to this Query explicitly from NDS2.
        There is no option to pad missing values using this method."""
        return gwpy.timeseries.TimeSeries.fetch(self.channel, self.start,
                                                self.end, verbose=VERBOSE_GWPY,
                                                **kwargs)
    def read(self, **kwargs):
        """Read this timeseries from file using GWpy. If the file is not
        present, an IOError is raised, UNLESS an unsuccessful attempt has been
        made to download the file, in which case it raises an
        NDS2Exception (a custom error type)."""
        try:
            return gwpy.timeseries.TimeSeries.read(self.fname, **kwargs)
        except IOError as e:
            if not self.query_failed():
                msg = ('tried concatenating data, but a download attempt seems '
                       'not to have been made for this query: {} See IOError '
                       'message: {}').format(self, e)
                logging.error(msg)
                raise IOError(('Aborting concatenation. Neither an error log '
                               'file nor a saved timeseries file were found '
                               'for this query: {}').format(self))
            else:
                logging.warn(('While reading, encountered failed query: '
                              '{}. Padding...').format(self))
                raise NDS2Exception(('This query seems to have failed '
                                     'downloading: {}').format(self))
    @property
    def missing_gps_times(self, pad=DEFAULT_PAD):
        """Get a list of missing times for this query. These values are
        floats."""
        t = self.read()
        missing_ind = np.nonzero(t == -1.)[0]
        return t.times[missing_ind].value
    def _get_missing_m_trend(self, pad='DEFAULT_PAD', **kwargs):
        """Get a single second of missing data."""
        logging.debug('Fetching missing m-trend: {}'.format(self))
        missing_buf = self.fetch() # explicitly fetch from NDS2
        trend = self.channel.split('.')[1].split(',')[0]
        # make m-trend value for this minute based on trend extension
        if len(np.nonzero(missing_buf == -1)[0]) != 0:
            # this won't actually check for anything at the moment because
            # gwpy.timeseries.TimeSeries.fetch() does not have a padding option
            # yet
            logging.warn('Still missing data in {}'.format(self))
        elif trend == 'mean':
            buf_trend = missing_buf.mean()
        elif trend == 'min':
            buf_trend = missing_buf.min()
        elif trend == 'max':
            buf_trend = missing_buf.max()
        elif trend == 'rms':
            buf_trend = missing_buf.rms(60)[0]
        elif trend == 'n':
            buf_trend = missing_buf.sum()
        else:
            raise ValueError('Unrecognized trend type: {}'.format(trend))
        return buf_trend
    def fill_in_missing_m_trend(self, pad='DEFAULT_PAD', **kwargs):
        """Missing m-trend data can often be filled in with s-trend data in
        cases where the m-trend fails to generate for some reason. This function
        takes a saved, completed query, loads the completely downloaded
        timeseries from disk, identifies missing values, fetches the s-trend
        for the missing minutes, generates m-trend values, and then saves the
        filled-in timeseries to disk."""
        buf = self.read()
        chan = buf.channel.name.split('.')
        # if this query is not a minute trend (m-trend), don't bother with
        # this; it won't work. just return. check this by noting that
        # the trend type will be specified for m-trends, but the ',m-trend'
        # suffix is implicit and will be left out (unlike for 's-trend', which
        # will include the ',s-trend' suffix in the channel name)
        if (len(chan) == 1) or (',' in chan[1]):
            return
        chan, trend = chan
        missing_times = [int(x) for x in self.missing_gps_times]
        # rename original file so that we don't overwrite it
        now = datetime.datetime.now().isoformat()
        backup_fname = 'with-missing-{}-{}'.format(now, self.fname)
        shutil.copyfile(self.fname, backup_fname)
        # download the s-trend 1 minute at a time
        for t in missing_times:
            full_trend = ','.join([trend, 's-trend'])
            squery = type(self)(t, t+60, '.'.join([chan, full_trend]), self.ext)
            buf_trend = squery._get_missing_m_trend(pad=pad, **kwargs)
            # replace missing value in loaded trend data
            missing_ind = np.argwhere(buf.times.value == t)[0][0]
            buf[missing_ind] = buf_trend
            # write to file, overwriting old file
            if os.path.isfile(self.fname):
                os.remove(self.fname)
            buf.write(self.fname)
    def read_and_split_on_missing(self, pad=DEFAULT_PAD, invert=False,
                                  **kwargs):
        """Read this timeseries from file using .read(), then find missing
        values (identified by the `pad' argument, i.e. the value used to pad
        missing space in the timeseries). Returns a list of contiguous
        timeseries that are a subset of this query's full time interval
        with all missing subintervals removed."""
        t = self.read(**kwargs)
        # find indices that are not just filler
        intervals = indices_to_intervals(np.nonzero(t != pad)[0])
        timeseries = []
        for i in range(len(intervals) // 2):
            timeseries.append(t[intervals[2*i]:intervals[2*i+1]+1])
        return timeseries
    @property
    def trend(self):
        """Get the trend extension for this query by splitting the channel
        name on the period. Will give results like ``'.mean,m-trend'``.
        Returns a blank string if there is no trend specified, i.e. full
        data."""
        if len(self.channel.split('.')) == 1:
            return ''
        else:
            return self.channel.split('.')[1]
    @property
    def channel_sans_trend(self):
        """Get the channel name with any trend extension, e.g.
        ``'.mean,m-trend'``, removed; this is the channel name as it should
        appear in a ``Job`` specification."""
        return self.channel.split('.')[0]
    def read_and_split_into_segments(self, dq_flag_segments):
        """Read this timeseries from file using ``.read()`` and split it into
        a list of subintervals that overlap with the provided
        ``dq_flag_segments``. ``dq_flag_segments`` must be an instance of
        ``gwpy.segments.DataQualityFlag``. Assumes m-trend for now."""
        # make sure this query is an m-trend; the code assumes this.
        if not 'm-trend' in self.trend:
            msg = 'Can only read and split m-trends by dq_flag.'
            logging.error(msg)
            raise ValueError(msg)
        # read in the timeseries
        t = self.read()
        n_segs = len(dq_flag_segments.active)
        t_subintervals = []
        for i_seg, seg in enumerate(dq_flag_segments.active):
            # this next bit seems to be necessary due to a bug; IIRC, one time
            # value might appear as text data rather than numerical data,
            # forcing this stupid kludgy conversion.
            start = gwpy.time.to_gps(seg.start).gpsSeconds
            end = gwpy.time.to_gps(seg.end).gpsSeconds
            # the start index for this segment might be outside the full timeseries
            try:
                i_start = np.argwhere(t.times.value==(start // 60 * 60))[0][0]
            except IndexError:
                i_start = 0
                msg = INDEX_MISSING_FMT.format('Start', i_seg, n_segs,
                                               start, 'start', i_start)
                logging.info(msg)
            # the end index for this segment might be outside the full timeseries
            try:
                i_end = np.argwhere(t.times.value==(end // 60 * 60 + 60))[0][0]
            except IndexError:
                # just pick the index of the last value in t.times
                i_end   = len(t.times) - 1
                msg = INDEX_MISSING_FMT.format('End', i_seg, n_segs,
                                               end, 'end', i_end)
                logging.info(msg)
            t_subintervals.append(t[i_start:i_end+1])
        return t_subintervals
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __str__(self):
        fmt = "start: {}, end: {}, channel: {}, ext: {}"
        return fmt.format(self.start, self.end, self.channel, self.ext)
    def __repr__(self):
        fmt = type(self).__name__ + '(start={}, end={}, channel={}, ext={})'
        return fmt.format(repr(self.start), repr(self.end),
                          repr(self.channel), repr(self.ext))
    # must be a staticmethod so that we can use multiprocessing on it
    @staticmethod
    def _download_data_if_missing(query, getmethod='get'):
        """download missing data if necessary. the query contains start, end,
        channel name, and file extension information in the following format:
            [ [start, end], channel, ext ]
        Specify whether ``fetch`` or ``get`` from gwpy should be used by
        passing the ``getmethod`` kwargument."""
        # only download the data if the file doesn't already exist
        logging.debug(("running query: {}, \nchecking "
                       "if file exists: {}").format(repr(query), query.fname))
        if not query.file_exists():
            logging.debug("{} not found, running query.".format(repr(query)))
            try:
                if getmethod == 'get':
                    data = query.get()
                elif getmethod == 'fetch':
                    data = query.fetch()
                else:
                    raise ValueError("``getmethod`` must be 'get' or 'fetch'.")
                logging.info("query succeeded: {} saving to file".format(query))
                data.write(query.fname)
            except RuntimeError as e:
                logging.warn(("Error while downloading {} from {} to {}: "
                              "{}").format(query.channel, query.start,
                                           query.end, e))
                with open(query.fname_err, 'w') as f:
                    f.write('Download failed: {}'.format(e))
    def download_data_if_missing(self, getmethod='get'):
        """download missing data if necessary. the query contains start, end,
        channel name, and file extension information in the following format:
            [ [start, end], channel, ext ]"""
        _download_data_if_missing(self, getmethod=getmethod)

def _download_data_if_missing(query, getmethod='get'):
    """Must define this at Global level to allow for multiprocessing"""
    Query._download_data_if_missing(query, getmethod=getmethod)

class Job(object):
    """A description of a data downloading job. Contains information on which
    time ranges should be downloaded, which channels and statistical trends
    should be downloaded, and which data quality (dq) flags should be
    downloaded. Provides methods for safely downloading required data in chunks
    and filling in missing values."""
    def __init__(self, start, end, channels, exts=DEFAULT_EXTENSION,
                 dq_flags=DEFAULT_FLAGS, trends=DEFAULT_TRENDS,
                 max_chunk_length=DEFAULT_MAX_CHUNK, filename=None):
        """Start and end times can be specified as either integer GPS times or
        as human-readable time strings that are parsable by gwpy.time.to_gps.
        max_chunk_length is measured in seconds and must be a multiple of 60."""
        if not set(exts).issubset(ALLOWED_EXTENSIONS):
            raise ValueError(('Must pick saved data file extension from: '
                              '{}').format(ALLOWED_EXTENSIONS))
        if not len(exts) == 1:
            raise ValueError(('For now, can only specify a single file '
                              'extension for downloaded data; instead, got: '
                              '{}').format(exts))
        if not max_chunk_length % 60 == 0:
            raise ValueError(('max_chunk_length must be a multiple of 60; got'
                              '{} instead.').format(max_chunk_length))
        self.start              = gwpy.time.to_gps(start).gpsSeconds
        self.end                = gwpy.time.to_gps(end).gpsSeconds
        self.channels           = [ str(c) for c in channels ]
        self.exts               = exts
        self.trends             = [ str(t) for t in trends ]
        self.dq_flags           = dq_flags
        self.max_chunk_length   = max_chunk_length
        self.filename           = filename
        # if minute-trends are being downloaded, expand the interval so
        # that start and end times are divisible by 60.
        if any(['m-trend' in c for c in self.channels_with_trends]):
            self.start = (self.start // 60) * 60
            self.end   = int(math.ceil(self.end / 60.)) * 60
        else:
            self.start = self.start
            self.end   = self.end
    @classmethod
    def from_dict(cls, d):
        """Instantiate a Job from a dictionary. Optional keyword arguments
        do not need to be present in the dictionary. Filename information is
        not included."""
        # there are some optional parameters that we will only pass to the
        # __init__ method if they are included in the JSON.
        kwargs = {}
        for optional_key in ['dq_flags', 'exts', 'trends', 'max_chunk_length']:
            if optional_key in d:
                kwargs[optional_key] = d[optional_key]
        # start and end cannot be unicode strings because GWpy complains
        for key in ['start', 'end']:
            if isinstance(d[key], unicode):
                d[key] = str(d[key])
        return cls(d['start'], d['end'], d['channels'], **kwargs)
    @classmethod
    def load(cls, jobspecfile='jobspec.json'):
        """load this job from a job specification file, assumed to be formatted
        in JSON."""
        with open(jobspecfile, "r") as f:
            job = cls.from_dict(json.load(f))
            job.filename = jobspecfile
            return job
    def to_dict(self):
        """Return a dict representing this job. Filename information is not
        included."""
        return { 'start':               self.start,
                 'end':                 self.end,
                 'channels':            self.channels,
                 'exts':                self.exts,
                 'dq_flags':            self.dq_flags,
                 'trends':              self.trends,
                 'max_chunk_length':    self.max_chunk_length }
    def save(self, jobspecfile):
        """Write this job specification to a JSON file named
        ``jobspecfile``."""
        with open(jobspecfile, 'w') as f:
            json.dump(self.to_dict(), f, sort_keys=True, indent=2)
    def overwrite(self):
        """Write this job specification to the same JSON file that it was
        loaded from (or whatever the current value of the job's ``filename``
        property is). Will fail if no ``filename`` property is present for the
        job.
        
        WARNING: no backup will be made and no confirmation will be requested
        before overwriting the old file.
        """
        self.save(self.filename)
    @property
    def job_sha(self):
        """Get the sha256 checksum of this job (as represented in its canonical
        JSON format with sorted keys and no indentation)."""
        return hashlib.sha256(json.dumps(self.to_dict(),
                                         sort_keys=True)).hexdigest()
    @property
    def duration(self):
        """Get the duration of this job in seconds."""
        return self.end - self.start
    @property
    def subspans(self):
        """split the time interval into subintervals that are each up to the
        ``max_chunk_length`` in duration and return that list of subintervals.
        returns a list of [start, stop] pairs."""
        mchunk = self.max_chunk_length
        # do we start and end cleanly at the start of a new chunk (in gps time)?
        # measured in number of time chunks since GPS time 0.
        end_first_chunk = int(math.ceil(self.start / float(mchunk)))
        start_last_chunk = int(self.end // mchunk)
        # if this is all happening in the same chunk, no splitting needed
        if start_last_chunk + 1 == end_first_chunk:
            return [[self.start, self.end]]
        spans = [ [ i*mchunk, (i+1)*mchunk ]
                  for i in range(end_first_chunk, start_last_chunk) ]
        # include the parts of the timespan outside of the full chunks
        if self.start != end_first_chunk * mchunk:
            spans.insert(0, [self.start, end_first_chunk * mchunk])
        if self.end != start_last_chunk * mchunk:
            spans.append([start_last_chunk * mchunk, self.end])
        return spans
    @property
    def channels_with_trends(self):
        """get all combinations of channels and trend extensions in this job.
        returns a list of channel and trend pairs, each of the form
        [ channel, trend ]."""
        chans = [ c + t
                    for c in self.channels
                    for t in self.trends   ]
        logging.debug('all channels: {}'.format(chans))
        return chans
    @property
    def queries(self):
        """Return a list of Queries that are necessary to execute this job."""
        return [ Query(start = span[0], end = span[1], channel = chan,
                       ext = ext)
                    for chan in self.channels_with_trends
                    for span in self.subspans 
                    for ext  in self.exts ]
    @property
    def full_queries(self):
        """Return a list of Queries corresponding to each channel/trend
        combination. The full time interval for this job is used for each Query;
        it is not split into smaller subintervals, so this list of Queries is
        probably useless for fetching remote data."""
        return [ Query(start = j.start, end = j.end,
                       channel = j.channels_with_trends[0], ext = j.exts[0])
                     for j in self.joblets ]
    @property
    def joblets(self):
        """Return a bunch of jobs with a single channel, trend, and extension
        for each which, when combined, are equivalent to the total job.."""
        return [ type(self)(self.start, self.end, [chan], exts = [ext], 
                            dq_flags = self.dq_flags, trends = [trend],
                            max_chunk_length = self.max_chunk_length)
                    for chan in self.channels
                    for ext in self.exts
                    for trend in self.trends ]
    @property
    def start_iso(self):
        """An ISO timestring of the start time of this job."""
        return gwpy.time.tconvert(self.start).isoformat()
    @property
    def end_iso(self):
        """An ISO timestring of the end time of this job."""
        return gwpy.time.tconvert(self.end).isoformat()
    def run_queries(self, getmethod='get'):
        """Try to download all data, i.e. run all queries. Can use multiple
        processes to try to improve I/O performance, though by default, only
        runs in a single process."""
        _run_queries(self, multiproc=False, getmethod=getmethod)
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __repr__(self):
        fmt = (type(self).__name__
               + '(start={}, end={}, channels={}, exts={}, dq_flags={}, '
               +  'trends={}, max_chunk_length={})')
        return fmt.format(repr(self.start), repr(self.end), repr(self.channels),
                          repr(self.exts), repr(self.dq_flags),
                          repr(self.trends), repr(self.max_chunk_length))
    @property
    def output_filenames(self):
        """Get the filenames for all final output files created by this job
        (after concatenation of timeseries)."""
        return [ q.fname for q in self.full_queries ]
    def concatenate_files(self):
        """Once all data has been downloaded for a job, concatenate that data
        based on the extension specified for the job."""
        for joblet in self.joblets:
            full_query = Query(start = joblet.start, end = joblet.end,
                               channel = joblet.channels_with_trends[0], 
                               ext = joblet.exts[0])
            if full_query.file_exists():
                logging.info(('This joblet has already been concatenated, '
                              'skipping: {}').format(joblet))
            else:
                logging.debug(('concatenating timeseries for '
                               '{}').format(full_query.channel))
                # load everything into memory... will fail for large jobs.
                starting_index = 0
                queries = joblet.queries
                data_initialized = False
                # if the first timespan was not available, simply try the next.
                while not data_initialized:
                    try:
                        query = queries[starting_index]
                        data = query.read().copy()
                        data_initialized = True
                    except NDS2Exception:
                        starting_index += 1
                for query in queries[starting_index + 1:]:
                    try:
                        data.append(query.read().copy(), gap='pad',
                                    pad=DEFAULT_PAD)
                    except NDS2Exception:
                        pass
                if not full_query.file_exists():
                    data.write(full_query.fname)
                logging.debug('done concatenating: {}'.format(full_query))
    def fill_in_missing_m_trend(self):
        """Iterate through channel and trend extension combinations and fill in
        missing data due to malformed minute trends. This should ONLY be run
        after all data has been downloaded using the conventional approach.
        See Query.fill_in_missing_m_trend() for a full description of what this
        entails."""
        for q in self.full_queries:
            logging.info('Filling in missing m-trend values for {}'.format(q))
            q.fill_in_missing_m_trend()
    @property
    def is_finished(self):
        return all([q.file_exists() for q in self.full_queries])
    def current_progress(self):
        """Print out current progress of this download."""
        print('{}Checking progress on job{}: {}'.format(_GREEN, _CLEAR,
                                                        self.to_dict()))
        queries = self.queries
        n_tot = len(queries)
        print(_RED + 'NOTE that below values only show incremental progress,')
        print('not finished files! If you have the finished files already,')
        print('then you probably don\'t need all of the partial downloads.')
        print('You can check whether the final outputs of this job have been')
        print('downloaded by using the -o flag.' + _CLEAR)
        print('{}Total downloads needed:{} {}'.format(_GREEN, _CLEAR, n_tot))
        successful = filter(lambda q: q.file_exists(), queries)
        successful_percentage = len(successful) * 100. / n_tot
        print('{}Successful downloads{}: {}'.format(_GREEN, _CLEAR,
                                                    len(successful)))
        not_done = filter(lambda q: not q.file_exists(), queries)
        failed = filter(lambda q: q.query_failed(), not_done)
        failed_percentage = len(failed) * 100. / n_tot
        print('{}Failed downloads{}: {}'.format(_GREEN, _CLEAR,
                                                len(failed)))
        failed_times = set([(q.start, q.end) for q in failed])
        print('{}Failed timespans{}:'.format(_GREEN, _CLEAR))
        for f in failed_times:
            print('    {}'.format(f))
        in_progress = filter(lambda q: not q.query_failed(), not_done)
        in_progress_percentage = len(in_progress) * 100. / n_tot
        print('{}In progress downloads{}: {}'.format(_GREEN, _CLEAR,
                                                     len(in_progress)))
        summary_fmt = '{}SUMMARY{}:\n{}% done\n{}% failed\n{}% remains'
        print(summary_fmt.format(_GREEN, _CLEAR, successful_percentage,
                                 failed_percentage, in_progress_percentage))
    def list_outfiles(self):
        """List output filenames (i.e. the files that should be produced once
        all data in the jobspec are downloaded and concatenated) and whether
        they exist or not in a human-readable format."""
        does_exist = '[{} EXISTS {}] '.format(_GREEN, _CLEAR)
        does_not_exist = '[{} MISSING {}]'.format(_RED, _CLEAR)
        for f in self.output_filenames:
            if os.path.isfile(f):
                exists = does_exist
            else:
                exists = does_not_exist
            print('{} -> {}'.format(exists, f))
    @property
    def output_filenames_sha(self):
        """Get the sha256 sum of the output filenames. Used for handily
        labeling collections of output files for this jobspec."""
        return hashlib.sha256('\n'.join(self.output_filenames)).hexdigest()
    @property
    def output_archive_filename(self):
        """Get a filename for a .tar.gz archive that the output of this jobspec
        will be stored in. This is based on the output filenames via a hash
        sum and should therefore with high probability be unique for a given
        jobspec."""
        return "jobarchive_{}.tar.gz".format(self.output_filenames_sha)
    def output_archive(self):
        """Archive output files into a single file whose name is uniquely based
        on the contents of the jobspec for easy transport and later retrieval.
        Also archive the job specification in use as a JSON file in the archive
        named "jobspec.json" (irrespective of the jobspec filename that the
        jobspec was loaded from).
        
        If the jobspec was loaded from a file, this
        file is copied verbatim to the archive. If this jobspec has no
        corresponding file, then it will be dumped to a temporary file that
        will be copied to the archive.
        
        Will fail if any of the job's output files are missing."""
        if not all([os.path.isfile(f) for f in self.output_filenames]):
            raise IOError( 'GWpy dump job has missing output files. Aborting.')
        with tarfile.open(self.output_archive_filename, "w:gz") as archive:
            for output_file in self.output_filenames:
                archive.add(output_file)
            if self.filename is None:
                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    pass
                self.save(temp.name)
                archive.add(temp.name, arcname='jobspec.json')
                temp.unlink(temp.name)
            else:
                archive.add(self.filename, arcname='jobspec.json')
    def output_unarchive(self, archive_filename=None):
        """Unarchive the output files for this job. Looks for an archive file
        whose name is uniquely based on the output files of this job and
        extracts the dumped output files from it to the current directory for
        immediate use. You can specify a custom archive filename if you know
        that the archived does not have the canonical filename.
        
        Will fail if the file does not exist or if one of the expected output
        file names is not available within the archive.  Will also fail if the
        output file already exists in order to prevent accidental data deletion
        and potential subsequent subtle errors."""
        if archive_filename is None:
            archive_filename = self.output_archive_filename
        with tarfile.open(archive_filename, "r:gz") as archive:
            for output_file in self.output_filenames:
                if os.path.exists(output_file):
                    raise IOError('GWpy dump output file exists, aborting.')
                archive.extract(output_file)
    @classmethod
    def job_unarchive(cls, archive_filename, check_archive_filename=True):
        """Unarchive a jobspec file as well as its entire collection of output
        files from an output file archive. The jobspec will be parsed and
        loaded as part of the process, ensuring that it is a valid jobspec, and
        its contents will be used to finish unarchiving its output files,
        ensuring that they are consistent with the jobspec. The jobspec will be
        extracted as "jobspec.json" with no regard to the filename it was given
        when saved; it will not overwrite an existing file with the same name.
        
        Will fail if:
        
        - a file already exists with the name jobspec.json
        - the jobspec is invalid
        - any of the output filenames are incorrectly named or missing in the
          archive"""
        if os.path.exists('jobspec.json'):
            raise IOError('jobspec.json exists, aborting job unarchiving.')
        with tarfile.open(archive_filename, "r:gz") as archive:
            archive.extract('jobspec.json')
        job = cls.load('jobspec.json')
        if check_archive_filename:
            archive_filename = job.output_archive_filename
        job.output_unarchive(archive_filename)
    @property
    def segment_filename(self):
        """The filename of HDF5 file that holds the segments specified in this
        job (and any other job with the same start and end)."""
        return "{}-{}-segments.hdf5".format(self.start, self.end)
    def fetch_dq_segments(self):
        """Download data quality segments into a gwpy.DataQualityDict using
        that class's ``query`` method for the full timespan of this job."""
        return gwpy.segments.DataQualityDict.query(self.dq_flags, self.start,
                                                   self.end)
    def read_dq_segments(self):
        """Read the segments for this job from an HDF5 file, throwing an
        IOError if not all DataQualityFlags are present in the saved file."""
        segs = gwpy.segments.DataQualityDict.read(self.segment_filename)
        if set(self.dq_flags).issubset(segs.keys()):
            # remove extraneous dq_flags
            extraneous_keys = set(segs.keys()) - set(self.dq_flags)
            for extraneous_key in extraneous_keys:
                segs.pop(extraneous_key)
            return segs
        else:
            raise IOError('Not all DataQualityFlags present for this job.')
    def get_dq_segments(self):
        """Download data quality segments and save them to an HDF5 formatted
        file. If that file already exists and has all required segments,
        just load from the file. Returns a DataQualityDict containing
        all DataQualityFlags specified for this job over the entire time
        interval specified by this job."""
        # if a segments file already exists for this time interval, just
        # fetch the missing dq_flags and add them in.
        if os.path.isfile(self.segment_filename):
            segs = gwpy.segments.DataQualityDict.read(self.segment_filename)
            missing_keys = set(self.dq_flags) - set(segs.keys())
            # if there are missing keys, query server
            if not len(missing_keys) == 0:
                missing = gwpy.segments.DataQualityDict.query(missing_keys,
                                                              self.start,
                                                              self.end)
                # add missing values into the previously saved DataQualityDict
                for key in missing:
                    segs[key] = missing[key]
                # save the combined DataQualityDict, renaming the old copy with
                # '.orig' appended
                os.rename(self.segment_filename,
                          '{}.orig'.format(self.segment_filename))
                segs.write(self.segment_filename)
        # otherwise, just get the segments and save them.
        else:
            segs = self.fetch_dq_segments()
            segs.write(self.segment_filename)
        # remove extraneous keys that are not included in this ``Job``'s
        # dq_flags
        extraneous_keys = set(segs.keys()) - set(self.dq_flags)
        for extraneous_key in extraneous_keys:
            segs.pop(extraneous_key)
        return segs

def _run_queries(job, multiproc=False, getmethod='get'):
    """Try to download all data, i.e. run all queries. Can use multiple
    processes to try to improve I/O performance, though by default, only
    runs in a single process. Must define this at the global level to allow
    for multiprocessing."""
    if multiproc:
        mapf = multiprocessing.Pool(processes=NUM_THREADS).map
    else:
        mapf = map
    kwargs = {"getmethod": getmethod}
    mapf(functools.partial(_download_data_if_missing, **kwargs), job.queries)
    logging.info('done downloading data.')

def sanitize_for_filename(string):
    """Take some string and return a sanitized filename with offensive
    characters (colons and commas) replaced with innocuous characters.
    This is not a unique encoding; it is just meant to be minimally offensive
    to the eye and simple to understand. It is intended for use with EPICS
    channels and DQ flag names, which only colons and commas as non-standard
    characters for filenames."""
    return string.replace(':', '..').replace(',', '--')

if __name__ == '__main__':
    # if we are unarchiving an entire job and it's output, then there is no
    # jobspec file already in existence; we need to extract it from the jobspec
    # file, which is provided as the first arg after ``-x``, or which is
    # implicitly the only *.tar.gz file in the directory.
    if unarchive_job:
        if len(sys.argv) == 1:
            tarfiles = glob.glob('*.tar.gz')
            if len(tarfiles) != 1:
                raise ValueError(('Must be exactly 1 .tar.gz file in '
                                  'directory; otherwise, specify manually.'))
            archive_filename = tarfiles[0]
        elif len(sys.argv) == 2:
            archive_filename = sys.argv[1]
        else:
            raise ValueError(('Must specify exactly on archive filename or '
                              'have exactly 1 .tar.gz file in directory.'))
        Job.job_unarchive(archive_filename, check_archive_filename)
        print('Job unarchived successfully!')
        exit(0)
    # specify the job specification file to load
    if len(sys.argv) == 1:
        jobspecfile = 'jobspec.json'
    else:
        jobspecfile = sys.argv[1]
    # set up logging
    logging.basicConfig(filename='{}.log'.format(jobspecfile),
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    job = Job.load(jobspecfile)
    # see if we are supposed to do something besides download the data
    # (argparse is used at the start of the script to set these variables based
    # on command line arguments passed in)
    if check_progress:
        job.current_progress()
    if list_outfiles:
        job.list_outfiles()
    if archive_outfiles:
        job.output_archive()
        print('Done, archived filename:')
        print(job.output_archive_filename)
    if unarchive_outfiles:
        job.output_unarchive()
    if print_archive_filename:
        print(job.output_archive_filename)
    if (check_progress or list_outfiles or archive_outfiles or
            unarchive_outfiles or print_archive_filename):
        exit(0)
    logging.debug('job after gps conversion: {}'.format(job.to_dict()))
    logging.debug('all spans: {}'.format(job.subspans))
    logging.debug('all queries: {}'.format(job.queries))
    _run_queries(job, multiproc=True)
    logging.debug('finished downloading data. concatenating files...')
    job.concatenate_files()
    logging.debug('finished concatenating files. filling in missing values...')
    job.fill_in_missing_m_trend()
    logging.debug('finished files. DONE.')
