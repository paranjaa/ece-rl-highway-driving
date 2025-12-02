# ece-1508-rl-highway-driving
Repository for the Reinforcement Learning Course Project 

## Benchmarking trained models
Trained models are listed under `./models/`. The base `highway-env` config is listed under `config.json`. 

To benchmark a model, activate your virtual environment, install the requirements from `requirements.txt` and then run `benchmark.py`.
```text
>>> python benchmark.py --help
usage: benchmark.py [-h] [--use_density_2] [-f FILEPATH] {BASELINE,DQN,PPO,DDQN}

Tests each trained model on various vehicle densities and extracts performance metrics.

positional arguments:
  {BASELINE,DQN,PPO,DDQN}
                        The model type to use as the RL agent for benchmarking.

options:
  -h, --help            show this help message and exit
  --use_density_2       Whether to use a variant of the model that was trained with vehicle desnity 2. Note that this is not the vehicle density used for the benchmarking.
  -f FILEPATH, --filepath FILEPATH
                        Specify a filepath to a trained model of the appropriate type. Omitting this will default to the latest trained model for that model type.
```
