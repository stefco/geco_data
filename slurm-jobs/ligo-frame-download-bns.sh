#!/bin/sh
#
# Download one day worth of LIGO data
#
#SBATCH --account=geco
#SBATCH --job-name=LIGOFrameDownload
#SBATCH -c 1                 # number of CPU cores to use
#SBATCH --time=24:00:00    # run for a day
#SBATCH --mem-per-cpu=2gb

# BNS merger date
start=1187022080

# not a great hack
pass="$(cat /rigel/home/stc2117/ligopass.txt)"
/rigel/home/stc2117/bin/hacked-ligo-proxy-init "stefan.countryman:${pass}"

cd /rigel/home/stc2117/frames
get_whole_frame_files.py \
    --start                 "${start}" \
    --deltat                86400 \
    --hanford-frametypes    H1_HOFT_C00 H1_R \
    --livingston-frametypes L1_HOFT_C00 L1_R
