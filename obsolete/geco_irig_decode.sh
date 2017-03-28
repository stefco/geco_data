#!/bin/bash
# (c) Stefan Countryman 4/15/2016

# take the first second worth of an IRIG-B timeseries from stdin and reads what
# time it represents.

set -o errexit
set -o nounset
set -o noclobber

usage () {
    cat <<USAGE
USAGE:

    <source_of_timeseries> | geco_irig_decode

    geco_irig_decode takes the first second worth of an IRIG-B timeseries from
    stdin and decodes the time it represents, then prints that time to stdout.
    You can pipe in as much data as you want, but it will by default only
    decode the first second worth of data (you can override this with the -A
    flag). This helps guarantee its speed, even if the user is foolishly
    pumping gallons of data into the script.

    It will also check to make sure that the signal was not *too* erratic, that
    all control characters are in the right places within the signal, and that
    no illegal waveforms popped up. The script is implemented in pure bash and
    is adaptable. It is best used with geco_dump as the data source for the
    sake of convenience.

OPTIONS:

    -A                    decode as many seconds of input as are received.

    -h                    print this help dialog.

USAGE
}

decode_all=false
while getopts ":Ah" opt; do
    case ${opt} in
        A)  decode_all=true;;
        h)  usage && exit;;
    esac
done
shift $((OPTIND-1))

# this function converts from day-of-the-year to the date. unfortunately, it
# will only work in GNU date; the POSIX solution requires a sort of binary
# search through dates; an example is here:
# http://superuser.com/questions/231692/how-to-convert-from-day-of-year-and-year-to-a-date-yyyymmdd
# will probably implement that one in the future: TODO
day_to_date () { date -d "$1-01-01 +$2 days -1 day" "+%b %d"; }

# bitrate of the irig-b signal
bitrate=16384

# what is considered a high vs. low signal; set it to 3500 so that
# ringing doesn't get read as too high
threshold=3500

# where within each character to test
t1=20
t2=60
t3=110
t4=150

# how much to add using dc, since dc will just add the difference between test points
a1=${t1}
a2=$(( t2 - t1 ))
a3=$(( t3 - t2 ))
a4=$(( t4 - t3 ))

# how each character type is represented; these are the only acceptable line values
rep_0="1000"
rep_1="1100"
rep_c="1110"

# each line specifies the unit followed by a multiple. if, e.g., line s1
# equals 1, add 1 to the number of seconds; if 0, add none. if line s4
# equals 1, add 4 to the number of seconds; if 0, add none. etc. each
# line specifies which of the characters in each 100 character second-
# long IRIG-B second we are referring to. THESE ARE INDEXED STARTING FROM
# ZERO, since they will be accessing values in an array.

# seconds
s1=1
s2=2
s4=3
s8=4
s10=6
s20=7
s40=8

# minutes
m1=10
m2=11
m4=12
m8=13
m10=15
m20=16
m40=17

# hours
h1=20
h2=21
h4=22
h8=23
h10=25
h20=26

# days
d1=30
d2=31
d4=32
d8=33
d10=35
d20=36
d40=37
d80=38
d100=40
d200=41

# years (just the last two digits)
y1=50
y2=51
y4=52
y8=53
y10=55
y20=56
y40=57
y80=58

# read a timeseries from standard input and print the decoded time
get_time () {

    # make a temporary file to hold the raw output of the read
    tempfile=$(mktemp)

    # i think it is pretty clear what the next line does
    sed ${bitrate}q \
        | sed -n "$(echo {0..99}" ${bitrate}*100/${a1}+p${a2}+p${a3}+p${a4}+p" | dc | sed 's/$/p/')" \
        | sed 's_$_ '"${threshold}"'/p_' \
        | dc \
        | sed 'N;N;N;s/'$'\\\n''//g;s/'${rep_0}'/0/;s/'${rep_1}'/1/;s/'${rep_c}'/c/;/..../q1' \
        >| "${tempfile}"

    # check whether control signals are on first line and lines that are
    # multiples of ten
    sed -n '1p;10p;20p;30p;40p;50p;60p;70p;80p;90p;100p' "${tempfile}" \
        | sed -n '/c/!q1'

    # decode; read each line into an array
    # from: http://mywiki.wooledge.org/BashFAQ/005?highlight=%28readarray%29#Loading_lines_from_a_file_or_stream
    while IFS= read -r next_char; do
        chars+=("$next_char")
    done < "${tempfile}"
    [[ $next_char ]] && lines+=("$next_char")

    local s_ones=$((chars[s1] + 2*chars[s2] + 4*chars[s4] + 8*chars[s8]))
    local s_tens=$((chars[s10] + 2*chars[s20] + 4*chars[s40]))

    local m_ones=$((chars[m1] + 2*chars[m2] + 4*chars[m4] + 8*chars[m8]))
    local m_tens=$((chars[m10] + 2*chars[m20] + 4*chars[m40]))

    local h_ones=$((chars[h1] + 2*chars[h2] + 4*chars[h4] + 8*chars[h8]))
    local h_tens=$((chars[h10] + 2*chars[h20]))

    local d_ones=$((chars[d1] + 2*chars[d2] + 4*chars[d4] + 8*chars[d8]))
    local d_tens=$((chars[d10] + 2*chars[d20] + 4*chars[d40] + 8*chars[d80]))
    local d_hund=$((chars[d100] + 2*chars[d200]))

    local y_ones=$((chars[y1] + 2*chars[y2] + 4*chars[y4] + 8*chars[y8]))
    local y_tens=$((chars[y10] + 2*chars[y20] + 4*chars[y40] + 8*chars[y80]))

    local year=20${y_tens}${y_ones}
    local time_of_day=${h_tens}${h_ones}:${m_tens}${m_ones}:${s_tens}${s_ones}
    local day_of_year=${d_hund}${d_tens}${d_ones}
    local month_and_day=$(day_to_date ${year} ${day_of_year})

    echo ${month_and_day} ${time_of_day} ${year}
    rm "${tempfile}"
}

#
# Reads N lines from input, keeping further lines in the input.  If input has
# fewer than N lines, throw an error and exit the script; we expect an even
# multiple of N lines, corresponding to an even multiple of the bitrate. If no
# lines are read, simply return 1.
#
# Arguments:
#   $1: number N of lines to read.
#
# Return code:
#   0 if exactly N line were read OR if no lines were read
#   1 if input is empty.
#   2 if input has fewer than N lines, and exit the script.
#
# edited from source:
# http://stackoverflow.com/questions/8314499/read-n-lines-at-a-time-using-bash
function readlines () {
    local N="$1"
    local line
    local lines_handled=0

    # Read N lines
    for i in $(seq 1 $N); do
        # Try reading a single line
        read line
        if [ $? -eq 0 ]
        then
            # Output line
            echo $line
            let lines_handled+=1
        else
            if [ "${lines_handled}" -eq 0 ]; then
                return 1
            else
                echo "Only ${lines_handled} lines handled, expected ${N} lines!" >&2
                echo "Error while processing second # ${i} from stdin." >&2
                exit 2
            fi
        fi
    done
}

# the main loop
if ${decode_all}; then
    while one_second_of_data=$(readlines ${bitrate}); do
        echo "${one_second_of_data}" | get_time
    done
else
    get_time
fi
