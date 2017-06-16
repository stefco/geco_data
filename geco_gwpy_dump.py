#! /usr/bin/env python
# (c) Stefan Countryman, Jan 2017

# allowed file extensions for GWPy writing to file, documented at:
# https://gwpy.github.io/docs/v0.1/timeseries/index.html#gwpy.timeseries.TimeSeries.write
NUM_THREADS = 6     # number of parallel download threads
VERBOSE_GWPY = True
ALLOWED_EXTENSIONS = ["csv", "framecpp", "hdf", "hdf5", "txt"]
DEFAULT_EXTENSION = ['txt']
# by default, download full data, i.e. no trend extension
DEFAULT_FLAGS = ["H1:DMT-ANALYSIS_READY:1", "L1:DMT-ANALYSIS_READY:1"]
DEFAULT_TRENDS = ['']
SEC_PER_DAY = 86400
DEFAULT_MAX_CHUNK = SEC_PER_DAY
DEFAULT_PAD = -1.
INDEX_MISSING_FMT = ('{} index not found for segment {} of {}, time {}\n'
                     'Setting {} index to {}.')
USAGE="""
Save channel data in a sane, interruptible, parallelizable way.

Usage:

Start downloading data specified in jobspec.json:

    geco_gwpy_dump

Check incremental progress of download:

    geco_gwpy_dump -p

List final output filenames and whether they exist or not:

    geco_gwpy_dump -o

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
_GREEN = '\033[92m'
_RED   = '\033[91m'
_CLEAR = '\033[0m'

import sys
# don't import the rest if someone just wants help
if __name__ == '__main__':
    check_progress = False
    list_outfiles = False
    if len(sys.argv) != 1 and sys.argv[1] in ['-h', '--help']:
        print(USAGE)
        exit()
    if '-p' in sys.argv:
        sys.argv.remove('-p')
        check_progress = True
    if '-o' in sys.argv:
        sys.argv.remove('-o')
        list_outfiles = True
# slow import; only import if we are going to use it.
if not (__name__ == '__main__'
        and (check_progress or list_outfiles)):
    import gwpy.timeseries
    import gwpy.segments
import gwpy.time
import numpy as np
import json
import multiprocessing
import math
import os
import logging
import shutil
import datetime

class NDS2Exception(IOError):
    """An error thrown in association with some sort of failure to download
    data."""

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
        if (len(chan) == 1) or (',' in chan[1]):
            raise ValueError('Tried running with non m-trend')
        chan, trend = chan # for m-trend, 'm-trend' implicit in trend extension
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
        exist = np.nonzero(t != pad)[0]

        # those indices are usually in contiguous chunks. find the the indices
        # of the edges of those chunks within the list of non-filler indices.
        change_inds = np.argwhere(exist[1:] != exist[:-1] + 1).flatten()

        # basically just flatten the list of ends/starts; these are still
        # indices into the list of nonfiller indices rather than indices into
        # the full timeseries itself.
        inner_inds = [ ind for subint in [ [i,i+1] for i in change_inds ]
                           for ind in subint ]
        all_inds = np.concatenate([[0], inner_inds, [-1]])
        intervals = exist[all_inds]
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
                data = query.get()
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
    """A description of a data downloading job. Contains information on which
    time ranges should be downloaded, which channels and statistical trends
    should be downloaded, and which data quality (dq) flags should be
    downloaded. Provides methods for safely downloading required data in chunks
    and filling in missing values."""
    def __init__(self, start, end, channels, exts=DEFAULT_EXTENSION,
                 dq_flags=DEFAULT_FLAGS, trends=DEFAULT_TRENDS,
                 max_chunk_length=DEFAULT_MAX_CHUNK):
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
        self.channels           = [ str(c) for c in channels ]
        self.exts               = exts
        self.trends             = [ str(t) for t in trends ]
        self.dq_flags           = dq_flags
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
            return cls.from_dict(json.load(f))
    def to_dict(self):
        """Return a dict representing this job."""
        return { 'start':               self.start,
                 'end':                 self.end,
                 'channels':            self.channels,
                 'exts':                self.exts,
                 'dq_flags':            self.dq_flags,
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
    def run_queries(self):
        """Try to download all data, i.e. run all queries. Can use multiple
        processes to try to improve I/O performance, though by default, only
        runs in a single process."""
        _run_queries(self, multiproc=False)
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
                    except NDS2Exception as e:
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
        does_exist = '[{} EXISTS {}] '.format(_GREEN, _CLEAR)
        does_not_exist = '[{} MISSING {}]'.format(_RED, _CLEAR)
        for f in job.output_filenames:
            if os.path.isfile(f):
                exists = does_exist
            else:
                exists = does_not_exist
            print('{} -> {}'.format(exists, f))
    if check_progress or list_outfiles:
        exit(0)
    logging.debug('job after gps conversion: {}'.format(job.to_dict()))
    logging.debug('all spans: {}'.format(job.subspans))
    logging.debug('all queries: {}'.format(job.queries))
    _run_queries(job, multiproc=True)
    job.concatenate_files()
