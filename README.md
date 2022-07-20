# Housekeep

This repository contains the implementation of our ECCV 2022 paper [Housekeep: Tidying Virtual Households using Commonsense Reasoning](https://arxiv.org/pdf/2205.10712.pdf)

![teaser](images/teaser_wout_ab.png)

## Dependencies Installation

- Setup Python `conda create -n habitat python=3.7 cmake=3.14.0`
- Install habitat-sim `conda install habitat-sim==0.2.1 withbullet headless -c conda-forge -c aihabitat`
- Clone this repo and get all submodules for this repo: `git submodule init`, `git submodule update`

#### Habitat Lab
- Go into Habitat Lab directory `cd habitat-lab`
- Install requirements `pip install -r requirements.txt`
- Install development version of Habitat Lab `python setup.py develop --all`

#### Additional Installs
- Install A* path planner `cd cos_eor/explore/astar_pycpp && make && cd -`
- Install additional requirements with `pip install -r requirements.txt` in the base directory

#### Data
Find the instructions for setting up our dataset [here](data/README.md)

## Running the baseline
- Generate config files for running the experiments `./generate_configs.sh <exp_name> <explore_type> <rank_type>`
`<explore_type>` is the type of exploration module to use: `phasic` or `oracle`
`<rank_type>` is the ranking function to use: `LangModel` or `Oracle` <br><br>
This will create configuration files for all scenes inside `./logs/<exp_name>/configs`. Edit the configuration files to change other properties.

- Run hierarchical policy on a scene (eg. `ihlen_1_int`)
`./run_cli.sh <exp_name> <scene_name> <num_episodes>`
The experiment logs will be written inside `logs/<exp_name>` directory.

## Citing

If you find our work useful for your research, please consider citing:

```
@misc{kant2022housekeep,
            title={Housekeep: Tidying Virtual Households using Commonsense Reasoning},
            author={Yash Kant and Arun Ramachandran and Sriram Yenamandra and Igor Gilitschenski and Dhruv Batra and Andrew Szot and Harsh Agrawal},
            year={2022},
            eprint={2205.10712},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
}
```
