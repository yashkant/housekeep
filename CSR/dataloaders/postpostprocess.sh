#! /bin/bash

source /srv/rail-lab/flash5/kvr6/csrremote.sh
conda activate csr
python postpostprocess_index.py
