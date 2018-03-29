#!/usr/bin/env python

INFILES = ['O1GRBlist.csv', 'O2_GRB_o2abcFermiSwift.csv']

def csv_to_json(infile):
    """input file must be a .csv format file. dumpt to json and return dict."""
    import json
    outfile = infile[:-3] + 'json'
    with open(infile) as f:
        header = f.readline().decode("utf-8-sig").strip().split(',')
        table = dict()
        for col in header:
            table[col] = list()
        for rowtxt in f.readlines():
            row = rowtxt.decode("utf-8-sig").strip().split(',')
            for i, cell in enumerate(row):
                table[header[i]].append(cell)
    process_table(table)
    with open(outfile, 'wb') as of:
        json.dump(table, of, indent=2, allow_nan=True)
    return table

def process_table(table):
    """process and clean data, putting it into appropriate types. mutates the
    values in the table."""
    # convert these columns to floats
    float_cols = [
        'Trigger_time(GPS)',
        'T90_error',
        'T90',
        'redshift',
        'total_flux/fluence',
        'significance_for _subthreshold',
        'RA',
        'Dec',
        'position_error'
    ]
    # check these for "None" values
    none_cols = [
        'GraceID'
    ]
    for coltitle in float_cols:
        col = table[coltitle]
        for i, val in enumerate(col):
            try:
                col[i] = float(val)
            except ValueError as e:
                col[i] = None
    for coltitle in none_cols:
        col = table[coltitle]
        for i, val in enumerate(col):
            if val.lower() == "none" or val.lower() == "n/a":
                col[i] = None

for infile in INFILES:
    csv_to_json(infile)
