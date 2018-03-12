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
    args = parser.parse_args()
import json
import numpy as np
import astropy.time
from subprocess import Popen

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
    # TODO implement
    raise Exception('not implemented')

def extract_all_coincs(trigdict, inputdb, outdir):
    # TODO implement
    raise Exception('not implemented')

def main():
    trigdict = tsv_to_dict(args.inputtable)
    calc_gps_times(trigdict)
    calc_skymap_info(trigdict)
    # TODO finish implementing
