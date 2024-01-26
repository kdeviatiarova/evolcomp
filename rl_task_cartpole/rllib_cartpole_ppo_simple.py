import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import os

from cartpole_env import CartPoleLeftRightEnv

# Set True for debug
local_mode = True
ray.init(local_mode=local_mode)

from_checkpoint = None
stop_iters = 100
stop_timesteps = 9999999
stop_reward = 9999.0
framework = "torch"

# Launch several envs for speedup
num_workers = 2 if not local_mode else 0


# if you want to pass some args to env
env_config = {
        "time_limit": 300,
        }

tune.register_env(
    "CartPoleLeftRightEnv",
    lambda c: CartPoleLeftRightEnv(env_config=c),
)


config = (
    PPOConfig()
    .environment(
        env="CartPoleLeftRightEnv",
        disable_env_checking=True,
        env_config=env_config,
        # Will render only 1 env
        render_env=True
    )
    .framework(framework)
    .rollouts(
        num_rollout_workers=num_workers,
        batch_mode="complete_episodes"
    )
    .training(
        lr=0.0003,
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=512,
        train_batch_size=2048,
        num_sgd_iter=8,
        vf_loss_coeff=0.8,
        clip_param=0.2,
        entropy_coeff=0.01,
        model={
            "fcnet_hiddens": [64, 64, 64],
            "vf_share_layers": False,
        },
    )
    .resources(num_gpus=1)
    .debugging(log_level="INFO")
)

stop = {
    "training_iteration": stop_iters,
    "timesteps_total": stop_timesteps,
    "episode_reward_mean": stop_reward,
}

tune.run(
    'PPO',
    config=config.to_dict(),
    stop=stop,
    verbose=3,
    checkpoint_freq=10,
    checkpoint_at_end=False,
    storage_path=os.path.abspath("results/"),
)

ray.shutdown()


