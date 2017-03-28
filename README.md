# GECo Data

Scripts for manipulating timing data stored in GW frame files.

## Usage

So far, using these should just involve placing them in a folder in your path
and then calling the commands by name. Most of them take arguments involving
start/end times and possibly channel names.

To see detailed usage instructions, run the command followed by an `-h` flag.

## Examples

### Checks and Plots for EVNT Log and Timing Witness Paper

After an event, you will need to generate a bunch of timing plots. Generate the
IRIG-B decoded times and plots with:

```bash
irig-b-decode-commands.py -t GPSTIME -g GRACEID
```

Make DuoTone statistics plots showing the zero-crossing at the time of the
event as well as in the surrounding 10 minutes for each site:

```bash
duotone_delay.py --stat --ifo H1 -t GPSTIME
duotone_delay.py --stat --ifo L1 -t GPSTIME
```

Make overlay plots for IRIG-B and DuoTone as well as zero-crossing plots
for the DuoTone in the +/-15 minutes surrounding the event:

```bash
geco_overlay_plots.py -t GPSTIME
```

### Dumping data

Here is how I have been using these scripts to e.g. dump all files
for October 2015.

First, dump the slow channel data (using the `-l` flag to specify the
slow-channel list) from the entire month of October to the
`~/oct` directory (each channel will get its own directory within `~/oct`;
if `-p` is not specified, then `~` will be used by default):

```bash
geco_dump -s 'Oct 1 2015' -e 'Nov 1 2015' -p ~/oct -l slow
```

You could also just specify a couple of channels you want to
dump if you don't need all of the timing slow channels:

```bash
geco_dump -s 'Oct 1 2015' -e 'Nov 1 2015' -p ~/oct \
    "H1:SYS-TIMING_C_MA_A_PORT_2_SLAVE_CFC_TIMEDIFF_1" \
    "H1:SYS-TIMING_X_FO_A_PORT_9_SLAVE_CFC_TIMEDIFF_1"
```

If you have already dumped data from some of these frame files, they
will not be re-dumped. The time spent dumping should not depend on the
number of slow channels, since each channel dump gets its own separate
process.

While the dump is in progress, you can monitor how many frame files
out of the total have been dumped so far by adding the `-P` flag
to your previous query:

```bash
geco_dump -s 'Oct 1 2015' -e 'Nov 1 2015' -p ~/oct -l slow -P
```

If you need to print a comma-separated timeseries to `stdout`
(instead of saving each 64-second interval produced by
`framecpp_dump_channel` to its own file without editing, which is the
default behavior), you can specify the `-T` flag (for "timeseries"):

```bash
geco_dump -s 'Oct 1 2015' -e 'Nov 1 2015' -T "H1:SYS-TIMING_C_MA_A_PORT_2_SLAVE_CFC_TIMEDIFF_1"
```

The output of this command is a comma-separated timeseries suitable for
immediate reading into any plotting script, or piping into another utility.
Note that you can only do this with one channel at a time, for obvious
reasons.
If you have already dumped this time range to files, you can get the
same comma-separated timeseries read straight from text files for a big
speed-up using the `-R` flag rather than the `-T` flag. This is, of
course, also useful if you are running the script off of datagrid, and
you don't have any frame files handy:

```bash
geco_dump -s 'Oct 1 2015' -e 'Nov 1 2015' -R -p ~/oct "H1:SYS-TIMING_C_MA_A_PORT_2_SLAVE_CFC_TIMEDIFF_1"
```

Note that you have to tell the script where the files were saved originally;
in this case, assuming we already ran the first example command, they would
be saved to the `~/oct` directory.

Finally, once you have dumped the files, you can change to the `~/oct`
directory and zip them all up with:

```
geco_zip
```

## Tips

After dumping all that data. you should probably zip up your output directories
to make it easier to transfer the data elsewhere.  Compressing the data will
speed up file transfers, but more importantly, representing all of the files in
your output directories as a single zipped archive will make file transfer much
faster (downloading tens of thousands of small files will take many, many times
longer than downloading a single large file, even if the total amount of data
is nominally the same). You can do this like so:

```bash
zip -r output_file_name.zip input_directory
```

You can then do something clever like, say, uploading your data to a remote
server using `scp` or something like the pure-`bash`
[Dropbox-Uploader](https://github.com/andreafabrizi/Dropbox-Uploader) script.
