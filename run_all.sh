#!/bin/bash

experiment="baseline_phasic_oracle"
# experiment="baseline_oracle_oracle"

# List of names
# names=("pomaria_2_int" "merom_1_int")
# names=("pomaria_0_int" "pomaria_1_int")
names=("pomaria_2_int" "wainscott_0_int" "merom_0_int" "pomaria_0_int")
# Iterate through the list of names
for file in "${names[@]}";
# for file in /srv/flash1/gchhablani3/housekeep/logs/"$experiment"/configs/*.yaml
# for file in $(ls -r /srv/flash1/gchhablani3/housekeep/logs/"$experiment"/configs/*.yaml)
do
    name=$(basename "$file" .yaml)
    echo "Checking if folder exists for $name..."
    if [ -d "csr_raw/$name/$experiment" ]
    then
        echo "Folder already exists for $name, skipping..."
    else
        echo "Running command for $name..."
        CSR_PATH="csr_raw/$name/$experiment"
        export CSR_PATH=$CSR_PATH
        ./run_cli.sh "$experiment" "$name" 50
    fi
done
