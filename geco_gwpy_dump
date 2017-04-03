#! /usr/bin/env python
# (c) Stefan Countryman, Jan 2017

# allowed file extensions for GWPy writing to file, documented at:
# https://gwpy.github.io/docs/v0.1/timeseries/index.html#gwpy.timeseries.TimeSeries.write
NUM_THREADS = 6     # number of parallel download threads
VERBOSE_GWPY = True
ALLOWED_EXTENSIONS = ["csv", "framecpp", "hdf", "hdf5", "txt"]
DEFAULT_EXTENSION = ['txt']
# by default, download full data, i.e. no trend extension
DEFAULT_TRENDS = ['']
SEC_PER_DAY = 86400
DEFAULT_MAX_CHUNK = SEC_PER_DAY
USAGE="""
Save channel data in a sane, interruptible, parallelizable way.

Usage:

    geco_gwpy_dump

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

By default, data is downloaded in day-long chunks (except for the starting and
trailing timespans, which might be shorter). Full day timespans start and end
on gps days, so their gps times are divisible by 86400. The data spans are
contiguous with no overlap.

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

import sys
# don't import the rest if someone just wants help
if __name__ == '__main__':
    if len(sys.argv) != 1 and sys.argv[1] in ['-h', '--help']:
        print(USAGE)
        exit()
import gwpy.timeseries
import gwpy.time
import json
import multiprocessing
import math
import os
import logging
import shutil

class Query(object):
    """A channel and timespan for a single NDS query and save operation."""
    def __init__(self, start, end, channel, ext):
        self.start      = start
        self.end        = end
        self.channel    = channel
        self.ext        = ext
    @property
    def sanitized_channel(self):
        """get the channel name as used in filenames, i.e. with commas (,) and
        colons (:) replaced in order to fit filename conventions."""
        return self.channel.replace(':', '..').replace(',', '--')
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
    def fetch(self, **kwargs):
        """Fetch the timeseries corresponding to this Query from NDS2 using
        GWpy."""
        return gwpy.timeseries.TimeSeries.fetch(self.channel, self.start,
                                                self.end, verbose=VERBOSE_GWPY,
                                                **kwargs)
    def read(self, **kwargs):
        """Read this timeseries from file using GWpy."""
        return gwpy.timeseries.TimeSeries.read(self.fname, **kwargs)
    def __str__(self):
        fmt = "start: {}, end: {}, channel: {}, ext: {}"
        return fmt.format(self.start, self.end, self.channel, self.ext)
    def __repr__(self):
        fmt = str(type(self)) + '(start={}, end={}, channel={}, ext={})'
        return fmt.format(repr(self.start), repr(self.end),
                          repr(self.channel), repr(self.ext))
    # must be a staticmethod so that we can use multiprocessing on it
    @staticmethod
    def _download_data_if_missing(query):
        """download missing data if necessary. the query contains start, end,
        channel name, and file extension information in the following format:
            [ [start, end], channel, ext ]"""
        # only download the data if the file doesn't already exist
        logging.debug(("running query: {}, \nchecking "
                       "if file exists: {}").format(repr(query), query.fname))
        if not query.file_exists():
            logging.debug("{} not found, running query.".format(repr(query)))
            try:
                data = query.fetch()
                logging.info("query succeeded: {} saving to file".format(query))
                data.write(query.fname)
            except RuntimeError as e:
                logging.warn(("Error while downloading {} from {} to {}: "
                              "{}").format(query.channel, query.start,
                                           query.end, e))
                with open(query.fname_err, 'w') as f:
                    f.write('Download failed: {}'.format(e))
    def download_data_if_missing(self):
        """download missing data if necessary. the query contains start, end,
        channel name, and file extension information in the following format:
            [ [start, end], channel, ext ]"""
        _download_data_if_missing(self)

def _download_data_if_missing(query):
    """Must define this at Global level to allow for multiprocessing"""
    Query._download_data_if_missing(query)

class Job(object):
    """A description of a data downloading job."""
    def __init__(self, start, end, channels, exts=DEFAULT_EXTENSION,
                 trends=DEFAULT_TRENDS, max_chunk_length=DEFAULT_MAX_CHUNK):
        """Start and end times can be specified as either integer GPS times or
        as human-readable time strings that are parsable by gwpy.time.to_gps.
        max_chunk_length is measured in seconds and must be a multiple of 60."""
        if not set(exts).issubset(ALLOWED_EXTENSIONS):
            raise ValueError(('Must pick saved data file extension from: '
                              '{}').format(ALLOWED_EXTENSIONS))
        if not len(exts) == 1:
            raise ValueError(('For now, can only specify a single file '
                              'extension for downloaded data; instead, got: '
                              '{}').format(ext))
        if not max_chunk_length % 60 == 0:
            raise ValueError(('max_chunk_length must be a multiple of 60; got'
                              '{} instead.').format(max_chunk_length))
        self.start              = gwpy.time.to_gps(start).gpsSeconds
        self.end                = gwpy.time.to_gps(end).gpsSeconds
        self.channels           = channels
        self.exts               = exts
        self.trends             = trends
        self.max_chunk_length   = max_chunk_length
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
        do not need to be present in the dictionary."""
        # there are some optional parameters that we will only pass to the
        # __init__ method if they are included in the JSON.
        kwargs = {}
        for optional_key in ['exts', 'trends', 'max_chunk_length']:
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
            return cls.from_dict(json.load(f))
    def to_dict(self):
        """Return a dict representing this job."""
        return { 'start':               self.start,
                 'end':                 self.end,
                 'channels':            self.channels,
                 'exts':                self.exts,
                 'trends':              self.trends,
                 'max_chunk_length':    self.max_chunk_length }
    def save(self, jobspecfile):
        """Write this job specification to a JSON file."""
        with open(jobspecfile, 'w') as f:
            json.dump(self.to_dict(), f)
    @property
    def subspans(self):
        """split the time interval into subintervals that are each a day long,
        return that list of subintervals. returns a list of [start, stop]
        pairs."""
        total_span = [(self.start // 60) * 60,
                      int(math.ceil(self.end / 60.)) * 60]
        # do we start and end cleanly at the start of new days (in gps time)?
        first_gps_day = int(math.ceil(self.start
                                      / float(self.max_chunk_length)))
        last_gps_day = self.end // self.max_chunk_length
        spans = [ [ i * self.max_chunk_length, (i+1) * self.max_chunk_length ]
                  for i in range(first_gps_day, last_gps_day) ]
        # include the parts of the timespan outside of the full days
        if total_span[0] != first_gps_day * self.max_chunk_length:
            spans.insert(0, [total_span[0],
                         first_gps_day * self.max_chunk_length])
        if total_span[1] != last_gps_day * self.max_chunk_length:
            spans.append([last_gps_day * self.max_chunk_length, total_span[1]])
        return spans
    @property
    def channels_with_trends(self):
        """get all combinations of channels and trend extensions in this job."""
        chans = [ c + t
                    for c in self.channels
                    for t in self.trends   ]
        logging.debug('all channels: {}'.format(chans))
        return chans
    @property
    def queries(self):
        """Return a list of Queries that are necessary to execute this job."""
        return [ Query(span[0], span[1], chan, ext)
                    for chan in self.channels_with_trends
                    for span in self.subspans 
                    for ext  in self.exts ]
    @property
    def joblets(self):
        """Return a bunch of jobs with a single channel, trend, and extension
        for each which, when combined, are equivalent to the total job.."""
        return [ type(self)(self.start, self.end, [chan], [ext], [trend],
                               self.max_chunk_length)
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
    def run_queries(self):
        """Try to download all data, i.e. run all queries. Can use multiple
        processes to try to improve I/O performance, though by default, only
        runs in a single process."""
        _run_queries(self, multiproc=False)
    def concatenate_files(self):
        """Once all data has been downloaded for a job, concatenate that data
        based on the extension specified for the job."""
        for joblet in self.joblets:
            full_query = Query(joblet.start, joblet.end,
                               joblet.channels_with_trends[0], joblet.exts[0])
            if not full_query.file_exists():
                logging.debug(('combining timeseries for '
                               '{}').format(full_query.channel))
                # load everything into memory... will fail for large jobs.
                data = joblet.queries[0].read().copy()
                for query in joblet.queries[1:]:
                    data.append(query.read().copy())
                data.write(full_query.fname)
                logging.debug('done concatenating {}.'.format(full_query))

def _run_queries(job, multiproc=False):
    """Try to download all data, i.e. run all queries. Can use multiple
    processes to try to improve I/O performance, though by default, only
    runs in a single process. Must define this at the global level to allow
    for multiprocessing."""
    if multiproc:
        mapf = multiprocessing.Pool(processes=NUM_THREADS).map
    else:
        mapf = map
    mapf(_download_data_if_missing, job.queries)
    logging.info('done downloading data.')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        jobspecfile = 'jobspec.json'
    else:
        jobspecfile = sys.argv[1]
    logging.basicConfig(filename='{}.log'.format(jobspecfile),
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    job = Job.load(jobspecfile)
    logging.debug('job after gps conversion: {}'.format(job.to_dict()))
    logging.debug('all spans: {}'.format(job.subspans))
    logging.debug('all queries: {}'.format(job.queries))
    _run_queries(job, multiproc=True)
    job.concatenate_files()
