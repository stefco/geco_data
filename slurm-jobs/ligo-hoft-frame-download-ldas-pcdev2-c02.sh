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

#--[ USER INPUT ]--

# Set the start and end times for this dump in ISO Format
STARTMONTH=03
STARTYEAR=2017
ENDMONTH=04
ENDYEAR=2017

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

# not a great hack
pass="$(cat /rigel/home/stc2117/ligopass.txt)"
hacked-ligo-proxy-init "stefan.countryman:${pass}"

get_whole_frame_files.py \
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
