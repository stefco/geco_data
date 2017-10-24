#!/bin/sh
#
# Download one day worth of LIGO data
#
#SBATCH --account=geco
#SBATCH --job-name=LIGOFrameDownload
#SBATCH -c 1                 # number of CPU cores to use
#SBATCH --time=2:00:00:00    # two days
#SBATCH --mem-per-cpu=2gb

cd /rigel/home/stc2117/frames
get_whole_frame_files.py -t 1186704000 -d 86400
