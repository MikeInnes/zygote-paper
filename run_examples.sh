#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Ensure that all examples can be run directly
for subdir in ${DIR}/*_*_*; do
    script=$(echo ${subdir}/*.jl)
    if [[ -f ${script} ]]; then
        julia --project=. --color=yes "${script}"
    fi
done
