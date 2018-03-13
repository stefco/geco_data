#!/usr/bin/env python
# (c) Stefan Countryman, 2017
DESC="""Read table rows from an input tab-separated-value file and extract
corresponding events from the specified database file in 1. the coinc.xml
format expected by BAYESTAR, and 2. the skymap_info.json format expected by
the GWHEN pipeline. Save these output files in per-trigger directories named
``${gps_time}_${far}`` within a containing events directory.
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument(
        '-t',
        '--inputtable',
        help="""
            A tab-separated-value table containing GSTLAL triggers, one per
            row.  Must have columns for ``FAR``, ``end_time``, and
            ``end_time_ns``.
        """
    )
    parser.add_argument(
        '-d',
        '--inputdb',
        help="""
            A sqlite database file of the form output by the gstlal pipeline.
            Must contain events described in the table provided for the
            ``inputtable`` argument.
        """
    )
    parser.add_argument(
        '-o',
        '--outdir',
        help="""
            A directory to hold the per-trigger event directories. The
            per-trigger event directories will thus have paths of the form:
            ``${outdir}/${gps_time}_${far}``.
        """
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help="""
            Log warning messages to ``stderr``.
        """
    )
    args = parser.parse_args()
    verbose = args.verbose
else:
    verbose = False
import sys
import os
import json
import numpy as np
import astropy.time
from subprocess import Popen, PIPE

def complain(msg):
    """Write a message to stderr if the ``verbose`` flag has been specified."""
    if verbose:
        sys.stderr.write("{}\n".format(msg))

def tsv_to_dict(infile):
    """input file must be a .tsv format file."""
    with open(infile) as f:
        header = f.readline().decode("utf-8-sig").strip().split('\t')
        table = dict()
        for col in header:
            table[col] = list()
        for rowtxt in f.readlines():
            row = rowtxt.decode("utf-8-sig").strip().split('\t')
            for i, cell in enumerate(row):
                table[header[i]].append(cell)
    return table

def calc_gps_times(trigdict):
    """Calculate GPS times from the trigger dict returned by
    ``tsv_to_dict``. Save them in the ``gpstimes`` key of the dict."""
    gpstimes = (
           np.array(trigdict['end_time'], dtype=float)
        + (np.array(trigdict['end_time_ns'], dtype=float) * 1e-9)
    )
    trigdict['gpstimes'] = list(gpstimes)
    return trigdict

def calc_skymap_info(trigdict):
    """Calculate the ``skymap_info.json`` dicts for each trigger, which will
    later be dumpted to JSON files. Save them to the ``skyamp_info`` key of the
    dict."""
    skymap_info_list = list()
    for i in range(len(trigdict['FAR'])):
        skymap_info_list.append({
            'alert_type': 'gstlal-offline',
            'event_time_iso': astropy.time.Time(
                trigdict['gpstimes'][i],
                format='gps',
                scale='utc'
            ).isot,
            'far': trigdict['FAR'][i],
            'pipeline': 'GSTLAL',
            'skymap_filename': 'bayestar.fits.gz'
        })
    trigdict['skymap_info'] = skymap_info_list
    return trigdict

def extract_coinc(gpstime, inputdb, trigdir):
    """Extract a single ``coinc.xml`` file to an output directory. Returns
    ``True`` if successful, ``False`` otherwise."""
    inputdb_fullpath = os.path.realpath(inputdb)
    cmd = [
        'cody-gstlal_inspiral_coinc_extractor',
        '--fap-thresh',
        '1.0',
        '--gps-times',
        str(int(gpstime)),
        inputdb_fullpath
    ]
    complain(cmd)
    proc = Popen(cmd, cwd=trigdir, stdout=PIPE, stderr=PIPE)
    res, err = proc.communicate()
    if proc.returncode == 0:
        complain("Successfully extracted coinc xml.")
        return True
    else:
        fmt = ("Could not extract:\ngpstime:{}\ninputdb full path:{}"
               "\ntrigdir:{}\nstderr:{}\nstdout:{}\n")
        msg = fmt.format(gpstime, inputdb_fullpath, trigdir, err, res)
        complain(msg)
        return False

def extract_all(trigdict, inputdb, outdir):
    """Extract ``coinc.xml`` and ``skymap_info.json`` files for each trigger
    specified in ``trigdict`` from the sqlite file ``inputdb`` and save them in
    per-trigger directories contained in ``outdir`` following the name pattern
    ``${outdir}/${gps_time}_${far}``."""
    for i in range(len(trigdict['gpstimes'])):
        gpstime = trigdict['gpstimes'][i]
        far = trigdict['FAR'][i]
        fmt = 'gstlal-offline-{:0=10d}_{:+.3e}'
        trigdirname = fmt.format(int(gpstime), float(far))
        trigdir = os.path.join(outdir, trigdirname)
        if not os.path.isdir(trigdir):
            os.mkdir(trigdir)
        extract_coinc(gpstime, inputdb, trigdir)
        jsonpath = os.path.join(trigdir, 'skymap_info.json')
        with open(jsonpath, 'w') as f:
            json.dump(trigdict['skymap_info'][i], f, indent=2)

def main():
    trigdict = tsv_to_dict(args.inputtable)
    calc_gps_times(trigdict)
    calc_skymap_info(trigdict)
    extract_all(trigdict, args.inputdb, args.outdir)

if __name__ == "__main__":
    main()
