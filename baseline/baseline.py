import numpy as np
import gymnasium as gym
import highway_env


# ---------------------------
# 1. Environment constructor
# ---------------------------

def make_highway_env():
    """
    Create and configure a highway-v0 environment suitable for rule-based and RL agents.

    Returns
    -------
    env : gym.Env
        Configured highway environment.
    """
    config = {
        "observation": {
            "type": "Kinematics",
            # number of other vehicles besides ego
            "vehicles_count": 20,
            "features": ["x", "y", "vx", "vy"], # Removed 'lane_id' as it's not a direct Kinematics feature
            "absolute": True,   # we use absolute coordinates for simplicity
            "normalize": False, # keep values in physical units
        },
        "action": {
            "type": "DiscreteMetaAction"
            # default mapping is:
            # 0: "IDLE", 1: "LANE_LEFT", 2: "LANE_RIGHT",
            # 3: "FASTER", 4: "SLOWER"
        },
        "lanes_count": 4,
        "vehicles_density": 1.0,
        "duration": 40,                 # seconds
        "simulation_frequency": 15,     # Hz
        "policy_frequency": 5,          # Hz
        "collision_reward": -1.0,
        "high_speed_reward": 0.5,
        "right_lane_reward": 0.1,
        "lane_change_reward": 0.0,      # do not penalize lane change in env; we handle safety in rules
    }

    # Pass the config dictionary directly as a keyword argument to gym.make
    env = gym.make("highway-v0", render_mode=None, config=config)
    return env


# ---------------------------
# 2. Rule-based Agent
# ---------------------------

class RuleBasedAgent:
    """
    A simple rule-based driving agent for highway-env.

    High-level behaviour:)
    - Try to drive at a target speed when the road ahead is clear.
    - If there is a slower vehicle ahead and too close:
        - Try to change to a safe neighbouring lane (left first, then right).
        - If no safe lane is available, slow down to follow the lead vehicle.
    - Maintain simple safety constraints using headway and lane-change gap thresholds.
    """

    # We hard-code the mapping to make the logic explicit and readable.
    ACTION_ID = {
        "IDLE": 0,
        "LEFT": 1,
        "RIGHT": 2,
        "FASTER": 3,
        "SLOWER": 4,
    }

    def __init__(
        self,
        target_speed: float = 30.0,
        min_headway: float = 15.0,
        lane_change_gap: float = 12.0,
        speed_margin: float = 1.0,
    ):
        """
        Parameters
        ----------
        target_speed : float
            Desired cruising speed in m/s when the road is clear.
        min_headway : float
            Minimum safe distance (in meters) to the lead vehicle in the same lane.
            If the distance is smaller than this threshold, the agent will try to
            change lane or slow down.
        lane_change_gap : float
            Minimum required distance (in meters) to both the nearest front and
            rear vehicles in the target lane before performing a lane change.
        speed_margin : float
            Tolerance when comparing current speed to the target speed.
        """
        self.target_speed = target_speed
        self.min_headway = min_headway
        self.lane_change_gap = lane_change_gap
        self.speed_margin = speed_margin

    def select_action(self, obs: np.ndarray) -> int:
        """
        Select an action based on the current observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation from highway-env. Expected shape: (N, 4 or 5)
            Each row: [x, y, vx, vy, (optional) lane_id], with row 0 being the ego vehicle.

        Returns
        -------
        action : int
            Discrete action index compatible with DiscreteMetaAction.
        """
        obs = np.array(obs)
        ego = obs[0]
        others = obs[1:]

        # Unpack kinematic features. Lane ID will be inferred.
        ego_x, ego_y, ego_vx, ego_vy = ego[:4]
        ego_speed = float(np.hypot(ego_vx, ego_vy))

        # Infer ego_lane based on y-coordinate. Assuming lane width of 4m and lanes starting from y=0.
        # This is a heuristic as 'lane_id' is no longer directly observed.
        # Default highway-env has lane width 4m and y for lane 0 is typically centered around 0.
        # This heuristic might need adjustment based on specific environment road configuration.
        ego_lane = int(np.floor(ego_y / 4.0 + 0.5)) if ego_y >= 0 else int(np.ceil(ego_y / 4.0 - 0.5))
        # Ensure ego_lane is not negative if the environment setup implies non-negative lane IDs
        ego_lane = max(0, ego_lane)

        # For other vehicles, we assume their lane_id can be similarly inferred or is not critical for direct comparison in this part.
        # However, for `same_lane_mask` and `lane_is_safe`, we need `lane_id` for other vehicles too.
        # A more robust solution would involve adding a custom observation wrapper or accessing internal env state.
        # For now, let's assume `others` still provide a lane_id at index 4 (or we derive it).
        # Given the current setup, if `others` also lost `lane_id`, this would require further modification.
        # For this fix, we assume `others[:,4]` is still a valid way to get lane_id for other cars.
        # This is a temporary assumption. A robust solution would derive `lane_id` for all `others` as well.

        # -------------------------------
        # Step 1: find the lead vehicle in the same lane
        # -------------------------------

        # We need to infer lane_id for 'others' if it's not present.
        # This part of the code needs to be robust if others[:,4] is also not lane_id anymore.
        # For now, let's keep the original logic for `others` and assume it works for their `lane_id` for now,
        # or more precisely, the agent logic assumes `obs` for `others` has the same structure it expects.
        # If 'others' also only has [x, y, vx, vy], then `others[:,4]` will cause an IndexError.
        # Let's adjust for this possibility, inferring for others too.

        # Create a temporary array to hold observations with inferred lane_id
        obs_with_lane = np.zeros((obs.shape[0], obs.shape[1] + 1))
        obs_with_lane[:, :4] = obs[:, :4]
        obs_with_lane[0, 4] = ego_lane # Set ego vehicle's inferred lane_id

        # Infer lane_ids for other vehicles
        for i in range(1, obs.shape[0]):
            other_y = obs[i, 1]
            obs_with_lane[i, 4] = int(np.floor(other_y / 4.0 + 0.5)) if other_y >= 0 else int(np.ceil(other_y / 4.0 - 0.5))
            obs_with_lane[i, 4] = max(0, obs_with_lane[i, 4]) # Ensure non-negative
        
        # Reassign ego and others based on the new observation structure with inferred lane_id
        ego = obs_with_lane[0]
        others = obs_with_lane[1:]

        # Now ego_lane is available and others[:, 4] also contains inferred lane_id
        # ego_x, ego_y, ego_vx, ego_vy, ego_lane = ego # No longer needed, ego_lane is separately inferred


        same_lane_mask = (others[:, 4] == ego_lane)
        same_lane_cars = others[same_lane_mask]

        # lead cars are those ahead of the ego (x position larger than ego's)
        front_mask = same_lane_cars[:, 0] > ego_x
        front_cars = same_lane_cars[front_mask]

        lead_car = None
        lead_dist = np.inf
        lead_speed = 0.0

        if front_cars.shape[0] > 0:
            dists = front_cars[:, 0] - ego_x
            idx = int(np.argmin(dists))
            lead_car = front_cars[idx]
            lead_dist = float(dists[idx])
            lead_speed = float(np.hypot(lead_car[2], lead_car[3]))

        # -------------------------------
        # Helper: check if a target lane is safe for lane change
        # -------------------------------

        def lane_is_safe(target_lane: int) -> bool:
            """
            A target lane is considered safe if:
            - there is enough gap to the nearest front vehicle in that lane, and
            - there is enough gap to the nearest rear vehicle in that lane.
            """
            lane_cars = others[others[:, 4] == target_lane]
            if lane_cars.shape[0] == 0:
                # No cars at all in this lane -> safe.
                return True

            # distances along x-axis
            dx = lane_cars[:, 0] - ego_x

            # front vehicles in the target lane
            front = lane_cars[dx > 0]
            # rear vehicles in the target lane
            back = lane_cars[dx < 0]

            # Check front gap
            if front.shape[0] > 0:
                front_min = float(np.min(front[:, 0] - ego_x))
                if front_min < self.lane_change_gap:
                    return False

            # Check rear gap
            if back.shape[0] > 0:
                back_max = float(np.max(ego_x - back[:, 0]))
                if back_max < self.lane_change_gap:
                    return False

            return True

        # -------------------------------
        # Step 2: if there is no lead car ahead in the same lane
        # -------------------------------

        if lead_car is None:
            # road is clear, try to reach target speed
            if ego_speed < self.target_speed - self.speed_margin:
                return self.ACTION_ID["FASTER"]
            elif ego_speed > self.target_speed + self.speed_margin:
                return self.ACTION_ID["SLOWER"]
            else:
                return self.ACTION_ID["IDLE"]

        # -------------------------------
        # Step 3: there is a lead car in front
        # -------------------------------

        # If the lead car is too close, and typically slower, we try to change lane or slow down.
        if lead_dist < self.min_headway:
            # Note: lane numbering convention in highway-env is:
            # 0 = rightmost lane, increasing to the left.
            # We first try to change to the left lane, then to the right lane.
            # all_lanes = obs[:, 4] # Original line, now use inferred_obs_with_lane
            all_lanes = obs_with_lane[:, 4]
            min_lane = int(np.min(all_lanes))
            max_lane = int(np.max(all_lanes))

            left_lane = int(ego_lane + 1)
            right_lane = int(ego_lane - 1)

            # Try changing to the left lane if it exists and is safe
            if left_lane <= max_lane and lane_is_safe(left_lane):
                return self.ACTION_ID["LEFT"]

            # Otherwise, try changing to the right lane if it exists and is safe
            if right_lane >= min_lane and lane_is_safe(right_lane):
                return self.ACTION_ID["RIGHT"]

            # If no lane change is safe, slow down to follow the lead vehicle
            return self.ACTION_ID["SLOWER"]

        # -------------------------------
        # Step 4: lead car is not too close
        # -------------------------------

        # If ego is significantly slower than both the target speed and the lead car,
        # we can accelerate a bit. Otherwise, keep speed or slightly brake.
        desired_speed = min(self.target_speed, lead_speed)

        if ego_speed < desired_speed - self.speed_margin:
            return self.ACTION_ID["FASTER"]
        elif ego_speed > self.target_speed + self.speed_margin:
            return self.ACTION_ID["SLOWER"]
        else:
            return self.ACTION_ID["IDLE"]


# ---------------------------
# 3. Evaluation function
# ---------------------------

def evaluate_rule_based_agent(
    env: gym.Env,
    agent: RuleBasedAgent,
    num_episodes: int = 50,
):
    """
    Evaluate the rule-based agent on the given environment.

    We collect simple metrics:
    - collision_rate: fraction of episodes in which a collision occurred.
    - average_speed: mean ego speed over all timesteps and episodes.
    - average_return: mean cumulative reward per episode.

    Parameters
    ----------
    env : gym.Env
        An instance of highway-env (created by make_highway_env).
    agent : RuleBasedAgent
        The rule-based agent to evaluate.
    num_episodes : int
        Number of episodes to roll out.

    Returns
    -------
    metrics : dict
        Dictionary with keys: "collision_rate", "average_speed", "average_return".
    """
    total_collisions = 0
    episode_returns = []
    speeds = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action = agent.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

            # track speed of ego vehicle
            ego = np.array(obs[0])
            ego_vx, ego_vy = ego[2], ego[3] # Now obs[0] only has 4 elements
            ego_speed = float(np.hypot(ego_vx, ego_vy))
            speeds.append(ego_speed)

        episode_returns.append(episode_return)

        # highway-env uses "crashed" flag in info to signal collision
        if info.get("crashed", False):
            total_collisions += 1

    collision_rate = total_collisions / num_episodes
    average_speed = float(np.mean(speeds)) if speeds else 0.0
    average_return = float(np.mean(episode_returns)) if episode_returns else 0.0

    metrics = {
        "collision_rate": collision_rate,
        "average_speed": average_speed,
        "average_return": average_return,
    }
    return metrics


# ------------------------------------
# 4. Example usage (for quick testing)
# ------------------------------------

if __name__ == "__main__":
    env = make_highway_env()
    agent = RuleBasedAgent(
        target_speed=25.0,
        min_headway=15.0,
        lane_change_gap=12.0,
        speed_margin=1.0,
    )

    metrics = evaluate_rule_based_agent(env, agent, num_episodes=20)
    print("Rule-based baseline metrics over 20 episodes:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")