# Incentive Simulator

A tool for testing new reward functions by simulating their impact on miner incentive distributions.

![Miner Incentives Animation](static/miner_incentives.gif)

This tool is geared towards simulating different reward functions on SN34. The goal is to generalize this repository to be easiliy usable by other subnet owners to facilitate arbitrary validation mechanism simulations. 

Currently, `simulation.ipynb`...
- Loads and processes historical validator data from Weights & Biases (W&B) or local cache
- Parallelizes the cross product of validators x reward functions across available CPUs
- Has functionality to compute and plot rewards, scores, weights and incentives at different timesteps
