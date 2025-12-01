"""
benchmark.py

Evaluate the trained SB3 DQN policy across multiple highway-env configs.
The model is loaded from a checkpoint (you have to specify the directory).
For each config we run 50 evaluation episodes using a 4-env SubprocVecEnv
without rendering, and report:
    Average distance covered over episodes (in m)
    Average episode reward
    Collision rate (% of episodes that ended in a crash)
    Average ego speed (m/s) averaged over each episode
    TODO: Minimum TTC (time to collision): The minimum time to collision (in s) to the nearest vehicle assuming all vehicles continue at constant velocity
    RMS Acceleration : Root Mean Square Acceleration (m/s^2)
    TODO:RMS Jerk 
"""

import copy
import json
import os
from typing import Dict, List, Tuple

from enum import Enum

import gymnasium as gym
import highway_env  # noqa: F401 (register env)
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import RuleBasedAgent
from baseline.baseline import RuleBasedAgent

# ─────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────
BASE_CONFIG_PATH = "config.json"

BASE_MODEL_PATH = "models/"

class MODEL_TYPE(Enum):
    BASELINE = "BASELINE"
    DQN = "DQN"
    DDQN = "DDQN"
    PPO = "PPO_v2"

N_ENVS = 4
EPISODES_PER_CONFIG = 50

ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_base_config() -> Dict:
    with open(BASE_CONFIG_PATH, "r") as f:
        return json.load(f)


def make_vec_env(run_config: Dict) -> SubprocVecEnv:
    def make_env(rank: int):
        def _init():
            cfg = copy.deepcopy(run_config)
            cfg["offscreen_rendering"] = True
            cfg["real_time_rendering"] = False
            env = gym.make("highway-fast-v0", render_mode=None, config=cfg)
            return env

        return _init

    return SubprocVecEnv([make_env(i) for i in range(N_ENVS)])


def maybe_get_speed(info: Dict, obs_chunk) -> float:
    """
    Safely extract ego speed from the info dict (preferred, un-normalized).
    Fallback to 0.0 if not available.
    """
    if isinstance(info, dict) and "speed" in info:
        return float(info["speed"])
    
    # If everything fails, return 0.0
    return 0.0


def evaluate_config(model, run_config: Dict, label: str) -> Tuple[float, float, float, float, float, np.ndarray, int]:
    env = make_vec_env(run_config)

    # Determine dt (time step duration)
    # Default policy_frequency is 1 Hz if not specified.
    # We should use the same logic dt = 1 / policy_frequency
    policy_freq = run_config.get("policy_frequency", 1)
    dt = 1.0 / policy_freq

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, _info = reset_out
    else:
        obs = reset_out

    obs = np.array(obs)
    n_actions = int(env.action_space.n)
    action_counts = np.zeros(n_actions, dtype=np.int64)

    # Initialize metric lists
    ep_returns: List[float] = []
    ep_avg_speeds: List[float] = []
    ep_collisions: List[bool] = []
    ep_distances: List[float] = []
    ep_rms_accels: List[float] = [] 
    ep_rms_jerks : List[float] = []

    # Initialize env state trackers
    curr_return = np.zeros(N_ENVS, dtype=np.float32)
    curr_steps = np.zeros(N_ENVS, dtype=np.int32)
    curr_speed_sum = np.zeros(N_ENVS, dtype=np.float32)
    curr_dist_sum = np.zeros(N_ENVS, dtype=np.float32)
    curr_collision = np.zeros(N_ENVS, dtype=bool)
    
    # Acceleration trackers
    curr_sq_accel_sum = np.zeros(N_ENVS, dtype=np.float32) # Sum of accel^2
    curr_accel_steps = np.zeros(N_ENVS, dtype=np.int32)    # Count of accel steps
    prev_speeds = np.zeros(N_ENVS, dtype=np.float32)       # To calculate dv
    
    #set up trackers for calculating jerk (da/dt)
    # Sum of jerk^2
    curr_sq_jerk_sum = np.zeros(N_ENVS, dtype=np.float32)  
    # number of of jerk steps
    curr_jerk_steps = np.zeros(N_ENVS, dtype=np.int32)     
    # also prev acceleration
    prev_accels = np.zeros(N_ENVS, dtype=np.float32)       



    # We need to know if an env just reset to ignore the speed jump from End->Start
    just_reset = np.ones(N_ENVS, dtype=bool) 

    completed_eps = 0

    while completed_eps < EPISODES_PER_CONFIG:
        # ─────────────────────────────────────────────────────────────
        # Model Prediction Logic
        # ─────────────────────────────────────────────────────────────
        if isinstance(model, RuleBasedAgent):
            # RuleBasedAgent expects single observation, not vectorized
            actions = []
            for i in range(N_ENVS):
                action = model.select_action(obs[i])
                actions.append(action)
            actions = np.array(actions)

        else:
            actions, _ = model.predict(obs, deterministic=True)

        for action in np.asarray(actions).reshape(-1):
            action_counts[int(action)] += 1
        step_out = env.step(actions)
        if len(step_out) == 4:
            next_obs, rewards, dones, infos = step_out
        else:
            next_obs, rewards, dones, _, infos = step_out  # gymnasium compatibility

        rewards = np.array(rewards)
        dones = np.array(dones)
        next_obs = np.array(next_obs)

        curr_return += rewards
        curr_steps += 1

        # infos is a list of dicts per env
        for idx, info in enumerate(infos):
            ego_obs = next_obs[idx]
            speed = maybe_get_speed(info, ego_obs)
            
            curr_speed_sum[idx] += speed
            curr_dist_sum[idx] += speed * dt
            
            # Acceleration Calculation
            if not just_reset[idx]:
                dv = speed - prev_speeds[idx]
                accel = dv / dt
                curr_sq_accel_sum[idx] += accel ** 2
                curr_accel_steps[idx] += 1

                #also Jerk Calculation
                #but only if there are 2 accel values (curr_jerk is 0 initially)
                if curr_accel_steps[idx] > 1:
                    #jerk = da/dt = a_t - a_(t-1) / d_t
                    jerk = (accel - prev_accels[idx]) / dt
                    curr_sq_jerk_sum[idx] += jerk ** 2
                    curr_jerk_steps[idx] += 1
                prev_accels[idx] = accel

            else:
                # First step after reset, cannot calculate accel properly
                just_reset[idx] = False
            
            prev_speeds[idx] = speed # Update for next step

            if isinstance(info, dict) and info.get("crashed", False):
                curr_collision[idx] = True

        for idx, done in enumerate(dones):
            if done:
                avg_speed = curr_speed_sum[idx] / max(curr_steps[idx], 1)
                ep_returns.append(float(curr_return[idx]))
                ep_avg_speeds.append(float(avg_speed))
                ep_collisions.append(bool(curr_collision[idx]))
                ep_distances.append(float(curr_dist_sum[idx]))

                if curr_accel_steps[idx] > 0:
                    rms = np.sqrt(curr_sq_accel_sum[idx] / curr_accel_steps[idx])

                else:
                    rms = 0.0

                if curr_jerk_steps[idx] > 0:
                    rms_jerk = np.sqrt(curr_sq_jerk_sum[idx] / curr_jerk_steps[idx])
                else:
                    rms_jerk = 0.0

                ep_rms_accels.append(float(rms)) 

                ep_rms_jerks.append(float(rms_jerk))

                # Reset trackers
                curr_return[idx] = 0.0
                curr_steps[idx] = 0
                curr_speed_sum[idx] = 0.0
                curr_dist_sum[idx] = 0.0
                curr_sq_accel_sum[idx] = 0.0 
                curr_accel_steps[idx] = 0    
                curr_collision[idx] = False

                #also reset the ones for jerk
                curr_sq_jerk_sum[idx] = 0.0
                curr_jerk_steps[idx] = 0
                prev_accels[idx] = 0.0
                
                just_reset[idx] = True
                prev_speeds[idx] = 0.0 

                completed_eps += 1
                if completed_eps >= EPISODES_PER_CONFIG:
                    break

        obs = next_obs

    env.close()

    avg_return = float(np.mean(ep_returns))
    collision_rate = float(np.mean(ep_collisions) * 100.0)
    avg_speed = float(np.mean(ep_avg_speeds))
    avg_distance = float(np.mean(ep_distances)) 
    avg_rms_accel = float(np.mean(ep_rms_accels)) 
    #adding a counter in for jerk
    avg_rms_jerk = float(np.mean(ep_rms_jerks)) 
    total_actions = action_counts.sum()
    
    #forgot to pass the counter in for jerk as well
    return avg_return, collision_rate, avg_speed, avg_distance, avg_rms_accel, avg_rms_jerk, action_counts, total_actions


def build_configs(base_cfg: Dict) -> List[Tuple[str, Dict]]:
    configs = []
    densities = [1.0, 1.5, 2.0]

    # Duration 40 first (as base), then duration 20 variants
    for density in densities:
        cfg = copy.deepcopy(base_cfg)
        cfg["vehicles_density"] = density
        cfg["duration"] = base_cfg.get("duration", 40)
        cfg["real_time_rendering"] = False
        label = f"density={density:.1f}, duration={cfg['duration']}s"
        configs.append((label, cfg))

    for density in densities:
        cfg = copy.deepcopy(base_cfg)
        cfg["vehicles_density"] = density
        cfg["duration"] = 20
        cfg["real_time_rendering"] = False
        label = f"density={density:.1f}, duration=20s"
        configs.append((label, cfg))

    return configs


def get_latest_model(model_type, configs, override=None):
    if model_type == MODEL_TYPE.BASELINE:
        print("Model: RuleBasedAgent")
        # Initialize RuleBasedAgent
        model = RuleBasedAgent(target_speed=23.0)

        # RuleBasedAgent requires un-normalized observations (meters)
        # We must modify the configs to set "normalize": False
        for _, cfg in configs:
            if "observation" in cfg:
                cfg["observation"]["normalize"] = False
            else:
                # If observation key is missing (unlikely based on your file), ensure it exists
                cfg["observation"] = {"normalize": False}

    elif model_type == MODEL_TYPE.DQN:
        model_path = get_model_path(model_type, override)
        print(f"Model: DQN (loading from {model_path})")
        model = DQN.load(model_path, device="cuda")

    elif model_type == MODEL_TYPE.PPO:
        model_path = get_model_path(model_type, override)
        print(f"Model: PPO (loading from {model_path})")
        model = PPO.load(model_path)

    elif model_type == MODEL_TYPE.DDQN:
        print("DDQN is currently unsupported.")
        exit(-1)
        # model_path = get_model_path(model_type)
        # print(f"Model: DDQN (loading from {model_path})")
        # model =
    else:
        print("Unknown model type")
        exit(-1)

    return model, configs


def get_model_path(model_type, override=None):
    if override:
        return override

    model_load_path = os.path.join(BASE_MODEL_PATH, model_type.value)
    try:
        files = [os.path.join(model_load_path, file) for file in os.listdir(model_load_path) if os.path.isfile(os.path.join(model_load_path, file))]

        latest_file = max(files, key=os.path.getmtime)
    except:
        # Fallback on checkpoints
        try:
            old_load_path = model_load_path
            model_load_path = os.path.join(model_load_path, "checkpoints")
            files = [os.path.join(model_load_path, file) for file in os.listdir(model_load_path) if os.path.isfile(os.path.join(model_load_path, file))]

            latest_file = max(files, key=os.path.getmtime)
        except:
            raise ValueError(f"No model found at {old_load_path} or {model_load_path}")

    return latest_file


if __name__ == "__main__":
    base_config = load_base_config()
    configs = build_configs(base_config)

    # ─────────────────────────────────────────────────────────────
    # Model Loading & Config Adjustments
    # ─────────────────────────────────────────────────────────────

    ############## SET THE MODEL TYPE ##############
    model_type = MODEL_TYPE.PPO
    ############## ------------------ ##############

    model, configs = get_latest_model(model_type, configs)

    # # You can specify a specific filename as an override if needs be
    # model, configs = get_latest_model(model_type, configs, override="models/PPO_v2/vd_1_5.zip)

    # ─────────────────────────────────────────────────────────────
    # Run Evaluation
    # ─────────────────────────────────────────────────────────────
    summary = []
    for label, cfg in configs:
        metrics = evaluate_config(model, cfg, label)
        summary.append((label, *metrics))

    print("=" * 70)
    print("Highway Model Benchmark Summary (50 eps/config, n_env=4)")
    print(f"Model type: {model_type}")
    print("=" * 70)
    for label, avg_ret, coll_rate, avg_speed, avg_dist, avg_rms_accel, avg_rms_jerk, action_counts, total_actions in summary:
        print(f"{label}")
        print(f"  Avg Reward     : {avg_ret:8.3f}")
        print(f"  Collision Rate : {coll_rate:8.2f}%")
        print(f"  Avg Speed      : {avg_speed:8.3f} m/s")
        print(f"  Avg Distance   : {avg_dist:8.3f} m") 
        print(f"  RMS Accel      : {avg_rms_accel:8.3f} m/s^2")
        print(f"  RMS Jerk      : {avg_rms_jerk:8.3f} m/s^3")
        print(f"  Action distribution (total actions = {total_actions}):")
        for idx, count in enumerate(action_counts):
            name = ACTION_NAMES.get(idx, f"A{idx}")
            pct = (count / total_actions * 100.0) if total_actions > 0 else 0.0
            print(f"      [{idx}] {name:10s}: {pct:6.2f}% ({count} actions)")
        print("-" * 60)
