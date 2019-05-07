#!/bin/bash

# Ensure that all examples can be run directly
for subdir in *_*_*; do
    julia --project=. --color=yes ${subdir}/*.jl
done
