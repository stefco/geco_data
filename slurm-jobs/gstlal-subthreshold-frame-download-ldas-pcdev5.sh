#!/bin/sh
#
# Download raw frames surrounding O1/O2 GSTLAL subthreshold triggers.
#
#SBATCH --account=geco
#SBATCH --job-name=FrDL5
#SBATCH -c 1                # number of CPU cores to use
#SBATCH --time=72:00:00     # run for 3 days
#SBATCH --mem-per-cpu=2gb
#SBATCH --mail-type=ALL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=stefan.countryman@gmail.com     # Where to send mail

#--[ USER INPUT ]--

# how many seconds of data in each frame?
frame_length=64

# what server to download from?
server=ldas-pcdev5.ligo.caltech.edu

# name output directory after start/end gps times
dirname=gstlal-subthreshold-raw-3

# where is the list of start/stop times?
times=/rigel/home/stc2117/dev/geco_data/slurm-jobs/static/raw-frame-times-v3-3.txt

#--[ DERIVED QUANTITIES ]--

outdir=/rigel/geco/users/shared/frames/gstlal_offline_subthreshold/"${dirname}"
mkdir -p "${outdir}"
echo OUTDIR: "${outdir}"

# not a great hack; don't authenticate if we're doing local stuff
auth_needed=0
for var in "$@"; do
    if [ "$var"z = -pz ] || [ "$var"z = --progressz ] \
        || [ "$var"z = -hz ] || [ "$var"z = --helpz ]; then
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

geco_fetch_frame_files.py <"${times}" "$@" \
    --times \
    --outdir                "${outdir}" \
    --length                "${frame_length}" \
    --server                "${server}" \
    --hanford-frametypes    H1_R \
    --livingston-frametypes L1_R
    # --start                 "${start}" \
    # --deltat                "${deltat}" \
exitcode=$?

# mark end time
printf 'JOB EXITING.\n'
date
exit ${exitcode}
