# flake8: noqa

# __rllib-first-config-begin__
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="CartPole-v1")
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
# __rllib-first-config-end__

import ray

ray.shutdown()

# __rllib-compute-action-begin__
# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.
try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False

from ray.rllib.algorithms.ppo import PPOConfig

env_name = "CartPole-v1"
env = gym.make(env_name)
algo = PPOConfig().environment(env_name).build()

episode_reward = 0
terminated = truncated = False

if gymnasium:
    obs, info = env.reset()
else:
    obs = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    if gymnasium:
        obs, reward, terminated, truncated, info = env.step(action)
    else:
        obs, reward, terminated, info = env.step(action)
    episode_reward += reward
# __rllib-compute-action-end__

# __rllib-get-state-begin__
from ray.rllib.algorithms.dqn import DQNConfig

algo = DQNConfig().environment(env="CartPole-v1").build()

# Get weights of the default local policy
algo.get_policy().get_weights()

# Same as above
algo.workers.local_worker().policy_map["default_policy"].get_weights()

# Get list of weights of each worker, including remote replicas
algo.workers.foreach_worker(lambda worker: worker.get_policy().get_weights())

# Same as above, but with index.
algo.workers.foreach_worker_with_id(
    lambda _id, worker: worker.get_policy().get_weights()
)
# __rllib-get-state-end__
