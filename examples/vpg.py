import numpy as np
from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.misc import logger


results = []

for rep in range(5):
    logfile = '/tmp/rllab/vpg/log'+str(rep)+'.csv'
    logger.add_tabular_output(logfile)

    env = normalize(CartpoleEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)  # section 5
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=500,    # table 2
        n_itr=500,         # table 2
        batch_size=50000,  # table 2 -- the rllab codebase measures batchsize in timesteps, no matter how many episodes there are in that time frame
        discount=0.99,   # table 2
        step_size=0.05,  # table 3
    )
    algo.train()

    log_table = np.genfromtxt(logfile, delimiter=',', skip_header=1)
    average_returns = log_table[:,2]
    mean_avg_return_all_iterations = np.mean(average_returns)
    print(mean_avg_return_all_iterations)
    results.append(mean_avg_return_all_iterations)

print('final avg', np.mean(results))

