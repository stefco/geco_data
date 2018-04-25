#!/bin/sh
#
# Download one day worth of LIGO data
#
#SBATCH --account=geco
#SBATCH --job-name=LIGOFrameDownload
#SBATCH -c 1                # number of CPU cores to use
#SBATCH --time=72:00:00     # run for 3 days
#SBATCH --mem-per-cpu=2gb
#SBATCH --mail-type=ALL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=stefan.countryman@gmail.com     # Where to send mail

# Some notes on start/stop times for runs. From JRPC Committee page:
# https://wiki.ligo.org/LSC/JRPComm/ObsRun2
# Page also has recommendations on DQ segments and frame types.
# O2 start:                 Wed Nov 30 16:00:00 UTC 2016
#   in GPS:                 1164556817
# O2 winter break start:    Thu Dec 22 23:00:00 UTC 2016
#   in GPS:                 1166482817
# O2 winter break end:      Wed Jan 04 16:00:00 UTC 2017
#   in GPS:                 1167580818
# O2 commissioning start:   Mon May 08 00:00:00 UTC 2017
#   in GPS:                 1178236818
# LLO resumed:              Fri May 26 06:00:00 UTC 2017
#   in GPS:                 1179813618
# LHO resumed:              Thu Jun 08 18:40:06 UTC 2017
#   in GPS:                 1180982424
# O2 end:                   Fri Aug 25 22:00:00 UTC 2017
#   in GPS:                 1187733618
#
# O1 Dates:
# From https://wiki.ligo.org/LSC/JRPComm/ObsRun1
# ER8B? start   Sep 12th 0:00 UTC (GPS 1126051217) 
# Switch from O1 data to post O1 Tues, Jan 19 11:07:59 AM CT (GPS 1137258496) 

#--[ USER INPUT ]--

# Set the start and end times for this dump in ISO Format
STARTMONTH=10
STARTYEAR=2015
ENDMONTH=11
ENDYEAR=2015

# how many seconds of data in each frame?
frame_length=4096

# what server to download from?
server=ldas-pcdev2.ligo.caltech.edu

#--[ DERIVED QUANTITIES ]--

# Calculate GPS start time and duration from UTC start/end dates
UTCSTART="${STARTYEAR}-${STARTMONTH}-01T00:00:00"
UTCEND="${ENDYEAR}-${ENDMONTH}-01T00:00:00"

# Must have the tconvert script installed for this to work
start=$(tconvert ${STARTYEAR}-${STARTMONTH}-01)
end=$(tconvert ${ENDYEAR}-${ENDMONTH}-01)
deltat=$((end - start))

# name output directory after start/end gps times
dirname="hoft-${STARTYEAR}-${STARTMONTH}-to-${ENDYEAR}-${ENDMONTH}"
outdir=/rigel/geco/users/stc2117/"${dirname}"
mkdir -p "${outdir}"
echo OUTDIR: "${outdir}"

# not a great hack; don't authenticate if we're doing local stuff
auth_needed=0
for var in "$@"; do
    if [ "$var"z = -pz ] || [ "$var"z = --progressz ]; then
        auth_needed=1
    fi
done
if [ $auth_needed -eq 0 ]; then
    echo "authenticating."
    pass="$(cat /rigel/home/stc2117/ligopass.txt)"
    hacked-ligo-proxy-init "stefan.countryman:${pass}"
else
    echo "not authenticating."
fi

geco_fetch_frame_files.py "$@" \
    --start                 "${start}" \
    --deltat                "${deltat}" \
    --outdir                "${outdir}" \
    --length                "${frame_length}" \
    --server                "${server}" \
    --hanford-frametypes    H1_HOFT_C02 \
    --livingston-frametypes L1_HOFT_C02
exitcode=$?

# mark end time
printf 'JOB EXITING.\n'
date
exit ${exitcode}
