# Subgroup Separability

Repository is anonymised for review.

## Abstract

We investigate performance disparities in deep classifiers. We find that the ability of classifiers to separate individuals into subgroups varies substantially across medical imaging modalities and protected characteristics; crucially, we show that this property is predictive of algorithmic bias. Through theoretical analysis and extensive empirical evaluation, we find a relationship between subgroup separability, subgroup disparities, and performance degradation when models are trained on biased data. Our findings shed new light on the question of how models become biased, which provides important insights for the development of fair medical imaging AI.

## Data

The preprocessing notebooks explain how to download the data, preprocess it, and generate the split csv files we have in the project. You will have to perform some of these steps manually, but we provide the final csv files specifying the image paths and metadata.

## Usage

All experiments can be run through the main file. Run `python main.py -h` to see the available configuration.

Two preconfigured sweeps are provided in the `sweeps/` directory. Run both of these to reproduce all the experiments in the paper. See `sweeps/README.md` for more information.

Once you've run the sweeps, we provide a notebook to analyze the results and reproduce our figures.

## Installation

We inlcude `pyproject.toml` and `poetry.lock` files to install the project using Poetry, or you can simply use pip with the provided `requirements.txt`. For users familiar with Docker, we provide a Dockerfile and a devcontainer configuration for VSCode.
