#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Ensure that all examples can be run directly
for subdir in ${DIR}/*_*_*; do
    for f in ${subdir}/run_*.jl; do
        julia --project=. --color=yes "${script}"
    done
done
