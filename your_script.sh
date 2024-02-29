#!/bin/bash

# Check if the number of arguments is correct
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <file> <keyword1> [<keyword2> ...]"
    exit 1
fi

# Assign input parameters to variables
file="$1"
shift
keywords=("$@")

# Loop through each keyword and search for it in the file
for keyword in "${keywords[@]}"; do
    echo "Searching for '$keyword' in $file..."
    grep "$keyword" "$file" | grep "error" >  output.txt
    if [ $? -eq 0 ]; then
        echo "Found '$keyword' in $file"
    else
        echo "Keyword '$keyword' not found in $file"
    fi
done
