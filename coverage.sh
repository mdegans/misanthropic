#!/bin/bash
set -e

# CD to this script's directory
cd "$(dirname "$0")"

# This script is used to generate coverage report for the project. If you have
# VsCode, use the "Coverage Gutters" extension to view the coverage report. Just
# install it and click "Watch" on your bottom status bar.
cargo llvm-cov --all-features --output-path lcov.info
cargo llvm-cov --all-features