# `Multifaceted confidence in exploratory choice.` *(preprint)*
## *Solopchuk O., Dayan P.*, 2024

The code reproduces all analysis and figures of the [preprint on biorxiv](https://www.biorxiv.org/content/10.1101/2024.05.23.595493v1)

## Abstract

Our choices are typically accompanied by a feeling of confidence - an internal estimate that they are correct. Correctness, however, depends on our goals. For example, exploration-exploitation problems entail a tension between short- and long-term goals: finding out about the value of one option could mean foregoing another option that is apparently more rewarding. Here, we hypothesised that after making an exploratory choice that involves sacrificing an immediate gain, subjects will be confident that they chose a better option for long-term rewards, but not confident that it was a better option for immediate reward. We asked 250 subjects across 2 experiments to perform a varying-horizon two-arm bandits task, in which we asked them to rate their confidence that their choice would lead to more immediate, or more total reward. Confirming previous studies, we found a significant increase in exploration with increasing trial horizon, but, contrary to our predictions, we found no difference between confidence in immediate or total reward. This dissociation is further evidence for a separation in the mechanisms involved in choices and confidence judgements.

## Installation

In order to install the code, either clone the repository, or download the zip file. Then do the following:
```sh
# change into directory
cd confidence_exploration

# create and activate a virtual environment
python3 -m venv .env
source .env/bin/activate # for mac or linux, or
.env\Scripts\activate # for windows

# install the required packages
pip install -r requirements.txt
```

## Usage

Running the analysis, and generating the figures is done by running `analysis_script.py`:
```sh
python analysis_script.py
```

## Description

### Python code
The code for generating the figures is `analysis_script.py`, with some utility functions in `supporting_functions.py`. 

### Data
The repository contains `data_exp1.npy` and `data_exp2.npy` preprocessed data files in the `data` directory, as well as the raw data in `data.zip`. 

It also contains (in `parameter_fits`) the fitted parameter values for choice and confidence models. These are automatically recreated by the script if removed.