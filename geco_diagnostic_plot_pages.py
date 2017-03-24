#!/usr/bin/env python
# (c) Stefan Countryman 2017

import matplotlib
matplotlib.use('Agg')
import gwpy.timeseries

GPS = 1172486067
MAX_SIMULTANEOUS_CHANS = 5
HTML_TOP = """
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Plots around {}</title>
        <style type="text/css">
            img, p {{
                max-width: 400px;
            }}
            div {{
                display: inline
            }}
        </style>
    </head>
    <body>

""".format(GPS)
HTML_BOTTOM = """

    </body>
</html>
"""

def channel_fname(GPS, chan):
    return '{}_{}.png'.format(GPS, chan.replace(':', '..'))

def image_link_html(GPS, chan):
    """return an HTML string for an image element for this plot"""
    return ('<div>\n'
            '    <p>{}</p>\n'
            '    <br>\n'
            '    <img src="./{}">\n'
            '</div>\n').format(chan, channel_fname(GPS, chan))

# read channel list
with open('diag_channels.txt', 'r') as f:
    chans = list(set(f.read().split('\n')) - {''})
    print('channels to plot: {}'.format(chans))

# make a preview html page
with open('index.html', 'w') as f:
    f.write(HTML_TOP)
    f.write('\n')
    for chan in chans:
        f.write(image_link_html(GPS, chan))
    f.write(HTML_BOTTOM)

"""
# fetch and plot channels
for i in range(0, len(chans), MAX_SIMULTANEOUS_CHANS):
    print('plotting {} through {} of {}'.format(i, i+MAX_SIMULTANEOUS_CHANS,
                                                len(chans)))
    chans_sublist = chans[i:i+MAX_SIMULTANEOUS_CHANS]
    try:
        bufs = gwpy.timeseries.TimeSeriesDict.fetch(chans_sublist, GPS-5, GPS+5,
                                                    verbose=True)
    except RuntimeError:
        print('Could not fetch all at once:\n{}'.format(chans_sublist))
        bufs = {}
        for chan in chans_sublist:
            try:
                buf = gwpy.timeseries.TimeSeries.fetch(chan, GPS-5, GPS+5,
                                                       verbose=True)
                bufs[chan] = buf
            except RuntimeError:
                print('Bad channel: {}'.format(chan))
    for chan in bufs:
        buf = bufs[chan]
        buf.plot().savefig(channel_fname(GPS, chan))
"""
