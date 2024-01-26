import os
import cv2
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from cartpole_env import CartPoleLeftRightEnv
import pickle


def read_config(checkpoint_path) -> dict:
    with open(os.path.join(checkpoint_path, "algorithm_state.pkl"), 'rb') as file:
        config = pickle.load(file)
    return config


base_dir = "results\\"
# checkpoint_path = "<exp_series>\\<PPO>\\<run_name>\<checkpoint_xxxxxx>"
checkpoint_path = "/content/results/PPO_2024-01-26_11-59-28/PPO_CartPoleLeftRightEnv_5b9b0_00000_0_2024-01-26_11-59-28/checkpoint_000009"
checkpoint_path = os.path.join(base_dir, checkpoint_path)

ray.init(local_mode=False)

# if you want to pass some args to env
env_config = {"time_limit": 300}

tune.register_env(
    "CartPoleLeftRightEnv",
    lambda c: CartPoleLeftRightEnv(env_config=c),
)

exp_config = read_config(checkpoint_path)['config']
exp_config['num_rollout_workers'] = 0

agent = PPO(exp_config)
agent.load_checkpoint(checkpoint_path)
policy = agent.get_policy()

env = CartPoleLeftRightEnv(env_config)

# Create a directory to store rendered frames
os.makedirs("rendered_frames", exist_ok=True)

for episode in range(10):
    score = 0.0
    state, _ = env.reset()
    for step in range(env_config["time_limit"]):
        # Render environment
        frame = env.render(mode='rgb_array')
        # Convert frame to BGR format for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Save frame as an image
        cv2.imwrite(f"rendered_frames/frame_{episode}_{step:04d}.png", frame_bgr)

        flat_obs = state
        act = agent.compute_single_action(flat_obs)
        s, r, d, t, i = env.step(act)
        state = s
        score += r
        if d:
            break

# Release any resources used by the environment
env.close()
ray.shutdown()
