#!/bin/sh
INPUT_DIR=$1
OUTPUT_DIR=$2

if [ -d $OUTPUT_DIR ]; then rm -rf $OUTPUT_DIR; fi

cp -r $INPUT_DIR $OUTPUT_DIR

for split in $OUTPUT_DIR/*; do
    for emotion in $split/*; do
        for frame in $emotion/*; do
            name=$(head /dev/urandom | tr -dc a-z0-9 | head -c 8)
            mv "$frame" "$(dirname "$frame")/${name}$(basename "$frame")"
        done
    done
done