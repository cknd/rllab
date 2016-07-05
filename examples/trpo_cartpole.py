import numpy as np
from matplotlib import pyplot as plt

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=500,
    n_itr=500,
    discount=0.99,
    step_size=0.1,
)

rets_per_episode_batchwise = algo.train()
rets_per_episode = [x for lst in rets_per_episode_batchwise for x in lst]



print('mean return over all episodes', np.mean(rets_per_episode))

plt.plot(rets_per_episode, alpha=0.3)
plt.savefig('/tmp/upsi/test_rllab/trpo_cartpole.png')
