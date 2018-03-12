#!/usr/bin/env python
# (c) Stefan Countryman, 2017
import argparse
DESC="""Read table rows from an input tab-separated-value file and extract
corresponding events from the specified database file in 1. the coinc.xml
format expected by BAYESTAR, and 2. the skymap_info.json format expected by
the GWHEN pipeline. Save these output files in per-trigger directories named
``${gps_time}_${far}_${p_value}`` within a containing events directory.
"""

parser = argparse.ArgumentParser(description=DESC)
parser.add_argument(
    '-t',
    '--inputtable',
    help="""
        A tab-separated-value table containing GSTLAL triggers, one per row.
        Must have columns for ``FAR``, ``end_time``, and ``end_time_ns``.
    """
)
parser.add_argument(
    '-d',
    '--inputdb',
    help="""
        A sqlite database file of the form output by the gstlal pipeline. Must
        contain events described in the table provided for the ``inputtable``
        argument.
    """
)
parser.add_argument(
    '-o',
    '--outdir',
    help="""
        A directory to hold the per-trigger event directories. The per-trigger
        event directories will thus have paths of the form:
        ``${outdir}/${gps_time}_${far}_${p_value}``.
    """
)
args = parser.parse_args()
