#!/usr/bin/env python

import sqlite3
import re

DEFAULT_TREND_TYPES = ['min','max','mean','rms','n']

conn = sqlite3.connect(':memory:')
c = conn.cursor()

def create_trend_table(trend_type):
    c.execute('CREATE TABLE ' + trend_type + ' (t REAL PRIMARY KEY, v REAL)')

def insert(trend_type, line):
    c.execute('INSERT INTO ' + trend_type + ' VALUES (?, ?)',
              tuple(re.split('[ \t]*', line)))

def join_on_times():
    """Return a cursor (which can be iterated through) and which will return
    rows of data for times where all specified trend types have been
    provided."""
    query = 'SELECT i.t, i.v'
    for t in TREND_TYPES[1:]:
        query += ', ' + t + '.v'
    query += ' FROM ' + TREND_TYPES[0] + ' i'
    for t in TREND_TYPES[1:]:
        query += ' JOIN ' + t + ' ON i.t = ' + t + '.t'
    return c.execute(query)

def read_from_file(infile):
    """Read data in from an input file."""
    trend_type = None
    for line in infile:
        if line in TREND_TYPES:
            trend_type = line
        elif (not trend_type is None) and len(re.split('[ \t]*', line)) == 2:
            insert(trend_type, line)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trend_types',
                        help=('Use this flag to specify which trends you will '
                            'will be feeding in. This can be used to specify '
                            'multiple trends seperated by commas. DEFAULT: '
                            + ','.join(DEFAULT_TREND_TYPES)),
                        default=','.join(DEFAULT_TREND_TYPES))
    args = parser.parse_args()
    return args

def main():
    import sys
    # get arguments from command line
    args = parse_arguments()
    TREND_TYPES = args.trend_types.split(',')
    if '' in TREND_TYPES:
        raise ValueError('Cannot provide an empty string as a trend type.')

    # create tables
    for t in TREND_TYPES:
        create_trend_table(t)

    # read from stdin
    read_from_filename(sys.stdin)

    # print results, starting with a header row describing each column
    print('\t'.join(TREND_TYPES))
    for row in join_on_times():
        print('\t'.join([str(x) for x in i]))

if __name__ == "__main__":
    main()
