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

## Example usage

Sample input for running `benchmark.py` on DDQN with trained density 2:
```shell
python benchmark.py DDQN --use_density_2
```

Sample output for running `benchmark.py` on DDQN with trained density 2:
```text
>>> python benchmark.py DDQN --use_density_2
Model: DDQN (loading from models/DDQN_density2\policy_net.pth)
======================================================================
Highway Model Benchmark Summary (50 eps/config, n_env=4)
Model type: MODEL_TYPE.DDQN
======================================================================
density=1.0, duration=40s
  Avg Reward     :   50.987
  Collision Rate :     0.00%
  Avg Speed      :   20.454 m/s
  Avg Distance   :  818.172 m
  RMS Accel      :    0.926 m/s^2
  RMS Jerk      :    2.901 m/s^3
  Action distribution (total actions = 6240):
      [0] LANE_LEFT :   2.31% (144 actions)
      [1] IDLE      :   8.97% (560 actions)
      [2] LANE_RIGHT:   3.32% (207 actions)
      [3] FASTER    :   3.43% (214 actions)
      [4] SLOWER    :  81.97% (5115 actions)
------------------------------------------------------------
density=1.5, duration=40s
  Avg Reward     :   49.734
  Collision Rate :    10.00%
  Avg Speed      :   21.437 m/s
  Avg Distance   :  813.165 m
  RMS Accel      :    2.471 m/s^2
  RMS Jerk      :    9.710 m/s^3
  Action distribution (total actions = 5760):
      [0] LANE_LEFT :   9.60% (553 actions)
      [1] IDLE      :  28.59% (1647 actions)
      [2] LANE_RIGHT:   8.45% (487 actions)
      [3] FASTER    :  14.70% (847 actions)
      [4] SLOWER    :  38.65% (2226 actions)
------------------------------------------------------------
density=2.0, duration=40s
  Avg Reward     :   45.389
  Collision Rate :    12.00%
  Avg Speed      :   20.631 m/s
  Avg Distance   :  778.407 m
  RMS Accel      :    1.632 m/s^2
  RMS Jerk      :    5.927 m/s^3
  Action distribution (total actions = 5836):
      [0] LANE_LEFT :  11.67% (681 actions)
      [1] IDLE      :  41.64% (2430 actions)
      [2] LANE_RIGHT:  10.86% (634 actions)
      [3] FASTER    :   6.31% (368 actions)
      [4] SLOWER    :  29.52% (1723 actions)
------------------------------------------------------------
density=1.0, duration=20s
  Avg Reward     :   25.055
  Collision Rate :     2.00%
  Avg Speed      :   20.503 m/s
  Avg Distance   :  413.890 m
  RMS Accel      :    1.051 m/s^2
  RMS Jerk      :    3.248 m/s^3
  Action distribution (total actions = 3172):
      [0] LANE_LEFT :   0.22% (7 actions)
      [1] IDLE      :   9.84% (312 actions)
      [2] LANE_RIGHT:   3.22% (102 actions)
      [3] FASTER    :   3.56% (113 actions)
      [4] SLOWER    :  83.17% (2638 actions)
------------------------------------------------------------
density=1.5, duration=20s
  Avg Reward     :   24.577
  Collision Rate :     6.00%
  Avg Speed      :   20.973 m/s
  Avg Distance   :  407.265 m
  RMS Accel      :    1.859 m/s^2
  RMS Jerk      :    7.362 m/s^3
  Action distribution (total actions = 2972):
      [0] LANE_LEFT :   8.45% (251 actions)
      [1] IDLE      :  36.51% (1085 actions)
      [2] LANE_RIGHT:   8.58% (255 actions)
      [3] FASTER    :   9.42% (280 actions)
      [4] SLOWER    :  37.05% (1101 actions)
------------------------------------------------------------
density=2.0, duration=20s
  Avg Reward     :   22.676
  Collision Rate :     8.00%
  Avg Speed      :   20.466 m/s
  Avg Distance   :  405.506 m
  RMS Accel      :    1.271 m/s^2
  RMS Jerk      :    4.133 m/s^3
  Action distribution (total actions = 3068):
      [0] LANE_LEFT :  15.61% (479 actions)
      [1] IDLE      :  36.31% (1114 actions)
      [2] LANE_RIGHT:  13.36% (410 actions)
      [3] FASTER    :   3.85% (118 actions)
      [4] SLOWER    :  30.87% (947 actions)
------------------------------------------------------------
```