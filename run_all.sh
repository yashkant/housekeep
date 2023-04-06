#!/bin/bash

experiment="baseline_phasic_oracle"

for file in /srv/flash1/gchhablani3/housekeep/logs/"$experiment"/configs/*.yaml
do
    name=$(basename "$file" .yaml)
    echo "Running command for $name..."
    CSR_PATH="csr_raw/$name/$experiment"
    export CSR_PATH=$CSR_PATH
    ./run_cli.sh "$experiment" "$name" 100
done