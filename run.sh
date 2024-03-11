#!/bin/bash

# Set LD_LIBRARY_PATH to an empty string and run the Julia script
LD_LIBRARY_PATH="" julia --threads 6 Batsher/runSherpa2.jl

python Batsher/plots.py
# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "Julia script executed successfully, now running Python script."
    # Run the Python script
    python Batsher/plots.py
else
    echo "Julia script failed to execute."
fi

echo "All done."
