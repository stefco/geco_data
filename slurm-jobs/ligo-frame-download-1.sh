#!/bin/sh
#
# Download one day worth of LIGO data
#
#SBATCH --account=geco
#SBATCH --job-name=LIGOFrameDownload1
#SBATCH -c 1                 # number of CPU cores to use
#SBATCH --time=24:00:00      # run for a day
#SBATCH --mem-per-cpu=2gb

# not a great hack
pass=$(cat /rigel/home/stc2117/ligopass.txt)
/rigel/home/stc2117/bin/hacked-ligo-proxy-init stefan.countryman:${pass}

cd /rigel/home/stc2117/frames
get_whole_frame_files.py -t 1186704000 -d 86400 -s ldas-pcdev1.ligo.caltech.edu

# mark end time
printf 'JOB EXITING.\n'
date
