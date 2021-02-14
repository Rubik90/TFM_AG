INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [ -d $OUTPUT_DIR ]; then rm -Rf $OUTPUT_DIR; fi

mkdir $OUTPUT_DIR

cp -r $INPUT_DIR $OUTPUT_DIR

find $OUTPUT_DIR -size  0 -print -delete


