#!/bin/bash
# (c) Stefan Countryman, 2017

usage(){
    echo "Find corrupt frame files downloaded by get_whole_frame_files.py in"
    echo "the current directory. Prints nothing if all downloaded GWF files"
    echo "are intact. Works by comparing the local and remote sha256 sums."
    echo "Good for detecting corruption due to file system problems or,"
    echo "more likely, due to interrupted downloads."
}

if [ "$1"z = -hz ]; then
    usage
    exit
fi

for f in *.gwf; do
    if ! cmp .${f}.{remote,local}_sha256.txt 2>/dev/null; then
        echo corrupt file: $f
    fi
done
