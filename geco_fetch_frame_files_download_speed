#!/bin/bash
# (c) Stefan Countryman, 2018

outfile=dl_delay.png
usage(){
    echo Find GWF files in this directory and make a histogram showing time
    echo delay between successive files.
    echo
    echo Histogram is saved to "'${outfile}'".
}

if [ "$1"z = -hz ]; then
    usage
    exit
fi

# how many standard deviations away before its an outlier and we ignore it?
SIGMAX=4

find -name '*.gwf' -printf '%T@\n' | sort -n | python -c "
import sys
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load the modification times
mtimes = numpy.loadtxt(sys.stdin)
dtimes = mtimes[1:] - mtimes[:-1]

# remove outliers
non_outlier_inds = numpy.argwhere(dtimes < dtimes.mean() + ${SIGMAX}*dtimes.std())
outlier_inds = numpy.argwhere(dtimes >= dtimes.mean() + ${SIGMAX}*dtimes.std())
clean_dtimes = dtimes[non_outlier_inds]

# list outliers
msg = ''
for i in outlier_inds:
    msg += str(dtimes[i]) + ', '

# make a histogram
fig = plt.figure()
n, bins, patches = plt.hist(clean_dtimes, 30)
plt.text(0.6,0.6,'Outliers > \$${SIGMAX}\sigma\$\nremoved:\n'+msg, transform=fig.transFigure)
plt.xlabel('Delay between successive file downloads [\$s\$]')
plt.ylabel('Count')
plt.title('Download speeds for GWF files in\n$(pwd)')
plt.savefig('${outfile}')
"
