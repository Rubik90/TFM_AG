for dir in $(find $1 -type d); do
    echo "${dir}: $(find ${dir} -maxdepth 1 -type f | wc -l)"
done
