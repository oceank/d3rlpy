import os
import random
import numpy as np
import torch
import gym
import d3rlpy
from d3rlpy.envs import ChannelFirst
from d3rlpy.algos import DISCRETE_ALGORITHMS


def set_seed(seed, env=None, eval_env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

def main(args):

    # get wrapped atari environment
    env = d3rlpy.envs.Atari(gym.make(args.env_name))
    eval_env = d3rlpy.envs.Atari(gym.make(args.env_name), is_eval=True)
    set_seed(args.seed, env=env, eval_env=eval_env)
    eval_env_scorer = d3rlpy.metrics.evaluate_on_environment(eval_env, n_trials=args.eval_episode_num, epsilon=0.001)

    online_algo_name = "DoubleDQN"
    offline_algo_name = "DiscreteCQL"
    cql = None
    num_critics = 2
    buffer_max_size = 1000000 # 1M transitions
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=buffer_max_size, env=env)

    ddqn = d3rlpy.algos.DoubleDQN(
        learning_rate=args.online_learning_rate,
        n_frames=args.stack_frames,
        batch_size=args.online_learning_batch_size,
        target_update_interval=args.online_learning_target_update_interval,
        q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=200),
        optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1.5e-4),
        reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
        scaler='pixel',
        use_gpu=True,
        n_critics = num_critics,
    )

    best_cql = None
    if args.bootstrap_online_with_best_offline_model:
        best_cql = d3rlpy.algos.DiscreteCQL(
            learning_rate=args.offline_learning_rate,
            optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
            batch_size=args.offline_learning_batch_size,
            alpha=args.cql_alpha,
            q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(
                n_quantiles=200),
            scaler="pixel",
            n_frames=args.stack_frames,
            target_update_interval=args.offline_learning_target_update_interval,
            reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
            use_gpu=True,
            n_critics = num_critics,
            )

    ddqn.build_with_env(env)
    of_and_on_flag = "of4on"
    
    num_steps_per_phase = args.num_steps // (2+args.num_offline_learning_bootstrap) # 2: excluding the initial and the last portions
    num_steps_per_epoch_online_learning = args.num_steps // args.num_online_epochs
  
    for offline_bootstrap_phase_idx in range(1, args.num_offline_learning_bootstrap+3):

        print(f"Offline Learning For Online Learning: Phase {offline_bootstrap_phase_idx}/{2+args.num_offline_learning_bootstrap}")
    
        # experiment_name: online_<algo1>_with_offline_<algo2>_on_<env_name>_seed_<seed>
        #    <env_name>_seed<seed>_<of4on/ofAon>_online<algo1>_phase<offline_bootstrap_phase_idx>
        #    <env_name>_seed<seed>_<of4on/ofAon>_offline<algo2>_phase<offline_bootstrap_phase_idx>
        experiment_name_online_algo = f"{args.env_name}_seed{args.seed}_{of_and_on_flag}_online{online_algo_name}_phase{offline_bootstrap_phase_idx}"
        experiment_name_offline_algo = f"{args.env_name}_seed{args.seed}_{of_and_on_flag}_offline{offline_algo_name}_phase{offline_bootstrap_phase_idx}"

        # Online Training
        # prepare algorithm for online learning
        print(f"Online Learning: Phase {offline_bootstrap_phase_idx}/{2+args.num_offline_learning_bootstrap}")

        # Bootstrap online learning with the offline-learned Q-function
        if  offline_bootstrap_phase_idx>1:
            target_cql = best_cql if args.bootstrap_online_with_best_offline_model else cql
            ddqn.copy_q_function_from(target_cql)
            ddqn.reset_optimizer_states() # might not be necessary
            ddqn._impl.update_target()

        # start online training
        if offline_bootstrap_phase_idx == 1: # only apply to the 1st phase
            update_start_step_online_learning = args.update_start_step_online_learning
            explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(
                start_epsilon=1.0, end_epsilon=0.01, duration=args.greedy_epsilon_exploration_duration)
        else:
            update_start_step_online_learning = 0
            explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(
                start_epsilon=1.0, end_epsilon=0.01, duration=0) # after the first phase, keep epsilon=0.01 for exploration

        ddqn.fit_online(
            env,
            buffer,
            explorer,
            n_steps=num_steps_per_phase, # number of max steps for online learning at the current phase
            n_steps_per_epoch=num_steps_per_epoch_online_learning,
            update_interval=1, # update every interaction
            update_start_step=update_start_step_online_learning,
            eval_env=eval_env, # 10 episodes are evaluated for each epoch. To modify the number episodes to evaluate, it needs to pass the info through this line fit_online()->train_single_env()->evaluate_on_environment()
            eval_epsilon=0.01,
            save_interval = (num_steps_per_phase//num_steps_per_epoch_online_learning), # save a model at the end of each online learning phase
            experiment_name = experiment_name_online_algo,
            show_progress = args.show_progress,
        )

        # Offline training
        # prepare algorithm
        print(f"Offline Learning: Phase {offline_bootstrap_phase_idx}/{2+args.num_offline_learning_bootstrap}")

        cql = d3rlpy.algos.DiscreteCQL(
            learning_rate=args.offline_learning_rate,
            optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
            batch_size=args.offline_learning_batch_size,
            alpha=args.cql_alpha,
            q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(
                n_quantiles=200),
            scaler="pixel",
            n_frames=args.stack_frames,
            target_update_interval=args.offline_learning_target_update_interval,
            reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
            use_gpu=True,
            n_critics = num_critics,
            )

        # Bootstrap the offline learning with the current online-learned policy:
        #   copy Q-function from the previous offline learning phase
        if args.bootstrap_offline_with_online:
            cql.build_with_env(env)
            cql.copy_q_function_from(ddqn)
            cql.reset_optimizer_states() # might not be necessary
            cql._impl.update_target()


        # start training
        mdp_dataset = buffer.to_mdp_dataset()
        num_steps_per_epoch_offline_learning = 125000
        num_training_steps_offline_learning = num_steps_per_epoch_offline_learning * args.num_offline_epochs
        cql.fit(
            mdp_dataset.episodes, #buffer._transitions._buffer[: num_transitions_in_buffer],
            eval_episodes=[None], 
            n_steps=num_training_steps_offline_learning, # number of training steps
            n_steps_per_epoch=num_steps_per_epoch_offline_learning, # number of training steps per epoch
            scorers={
                'environment': eval_env_scorer,
            },
            save_interval=args.num_offline_epochs, # save the model only at the end of the offline learning
            experiment_name = experiment_name_offline_algo,
            show_progress = args.show_progress, # show progress bar. Set to False when deploying in the cluster to save the log file,
            n_epochs_per_eval = 1, # evaluate the model at the end of each epoch
        )

        if args.bootstrap_online_with_best_offline_model:
            best_model_path = os.path.join(cql._learning_log_dir, "moeld_best.pt")
            best_cql.load_model(best_model_path)

    print('training finished!')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--algo', type=str, default="double_dqn") # check names of all supported algorightms at https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/__init__.py
    parser.add_argument('--env_name', required=True)
    #parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    #parser.add_argument('--seed_eval', type=int, default=100000)

    parser.add_argument('--num_steps', type=int, default=5000000, metavar='N',
                        help='maximum number of training steps during online learning (default: 5000000 (5M))')
    parser.add_argument('--num_offline_learning_bootstrap', type=int, default=3, metavar='N',
                        help='number of offline learning bootstraps for the entire online learning process(default: 3)')
    parser.add_argument('--num_online_epochs', type=int, default=50, metavar='N',
                        help='number of total online training epochs (default: 50).')
    parser.add_argument('--num_offline_epochs', type=int, default=10, metavar='N',
                        help='number of training epochs per phase during offline learning (default: 10)')
    parser.add_argument('--eval_episode_num', type=int, default=32,
                        help='Number of evaluation episodes (default: 32)')
    #parser.add_argument('--save_interval', type=int, default=10, metavar='N',
    #                    help='number of elapsed epochs when saving a model (default: 10)')

    parser.add_argument('--stack_frames', type=int, default=4)          
    parser.add_argument('--online_learning_batch_size', type=int, default=32)
    parser.add_argument('--offline_learning_batch_size', type=int, default=32)
    parser.add_argument('--online_learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--offline_learning_rate', type=float, default=5e-5)
    parser.add_argument('--online_learning_target_update_interval', type=int, default=8000) # 1000 gradient steps
    parser.add_argument('--offline_learning_target_update_interval', type=int, default=2000) # 1000 gradient steps
    parser.add_argument('--cql_alpha', type=float, default=4.0)
    parser.add_argument('--greedy_epsilon_exploration_duration', type=int, default=62500) # 2000 gradient steps
    parser.add_argument('--update_start_step_online_learning', type=int, default=50000) # 1000 gradient steps
    parser.add_argument('--show_progress', type=bool, default=False)
    parser.add_argument('--bootstrap_offline_with_online', type=bool, default=False) # Copy online-learned Q-function to bootstrap the offline learning
    parser.add_argument('--bootstrap_online_with_best_offline_model', type=bool, default=True)

    args = parser.parse_args()
    print(f"args: {args}")

    main(args)