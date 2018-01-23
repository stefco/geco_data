#!/bin/sh
#
# Download one day worth of LIGO data
#
#SBATCH --account=geco
#SBATCH --job-name=LIGOhoftFrameDownloadUWMPcdev1C02
#SBATCH -c 1                 # number of CPU cores to use
#SBATCH --time=72:00:00      # run for 3 days
#SBATCH --mem-per-cpu=2gb
#SBATCH --mail-type=ALL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=stefan.countryman@gmail.com     # Where to send mail

# BNS merger date
start=1185580818    # `lalapps_tconvert Aug 1 2017`

# output directory
outdir=/rigel/geco/users/stc2117/test-hoft-fix-2

# how many seconds of data in each frame?
frame_length=4096

# how many seconds of data to fetch?
deltat=2592000  # 30 days.

# what server to download from?
server=pcdev1.cgca.uwm.edu

# not a great hack
pass="$(cat /rigel/home/stc2117/ligopass.txt)"
/rigel/home/stc2117/bin/hacked-ligo-proxy-init "stefan.countryman:${pass}"

get_whole_frame_files.py \
    --start                 "${start}" \
    --deltat                "${deltat}" \
    --outdir                "${outdir}" \
    --length                "${frame_length}" \
    --server                "${server}" \
    --hanford-frametypes    H1_HOFT_C02 \
    --livingston-frametypes L1_HOFT_C02

# mark end time
printf 'JOB EXITING.\n'
date
