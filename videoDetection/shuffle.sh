#!/bin/sh

for frame in "$1"/*; do
    name=$(head /dev/urandom | tr -dc a-z0-9 | head -c 8)
    base=${frame%.*}
    ext=${frame#$base.}
    cp "$frame" "$2"/"$name"."$ext"
done