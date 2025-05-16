#!/bin/sh
cmake -S . -B build  # Press "c" to configure and "g" to generate
cmake --build build -j 2  # Run make with 8 threads