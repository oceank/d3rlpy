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
    env = d3rlpy.envs.Atari(gym.make(args.env))
    eval_env = d3rlpy.envs.Atari(gym.make(args.env), is_eval=True)
    set_seed(args.seed, env=env, eval_env=eval_env)
    eval_env_scorer = d3rlpy.metrics.evaluate_on_environment(eval_env, n_trials=args.eval_episode_num, epsilon=0.001)

    online_algo_name = "DoubleDQN"
    offline_algo_name = "DiscreteCQL"
    ddqn = None
    cql = None
    num_critics = 2
    buffer_max_size = 100000 # 100k transitions
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=buffer_max_size, env=env)
       
    of_and_on_flag = "of4on"
    num_epochs_offline_learning = args.num_offline_learning_bootstrap

    num_steps_per_phase = args.num_steps // args.num_offline_learning_bootstrap
    update_start_step=5000 # start updating after 5k interactions

    for offline_bootstrap_phase_idx in range(1, args.num_offline_learning_bootstrap+1):

        print(f"Offline Learning For Online Learning: Phase {offline_bootstrap_phase_idx}/{args.num_offline_learning_bootstrap}")
    
        # experiment_name: online_<algo1>_with_offline_<algo2>_on_<env_name>_seed_<seed>
        #    <env_name>_seed<seed>_<of4on/ofAon>_online<algo1>_phase<offline_bootstrap_phase_idx>
        #    <env_name>_seed<seed>_<of4on/ofAon>_offline<algo2>_phase<offline_bootstrap_phase_idx>
        experiment_name_online_algo = f"{args.env_name}_seed{args.seed}_{of_and_on_flag}_online{online_algo_name}_phase{offline_bootstrap_phase_idx}"
        experiment_name_offline_algo = f"{args.env_name}_seed{args.seed}_{of_and_on_flag}_offline{offline_algo_name}_phase{offline_bootstrap_phase_idx}"

        # Online Training
        # prepare algorithm for online learning
        print(f"Online Learning: Phase {offline_bootstrap_phase_idx}/{args.num_offline_learning_bootstrap}")
        ddqn = d3rlpy.algos.DoubleDQN(
            learning_rate=args.online_learning_rate,
            n_frames=args.stack_frames,
            batch_size=args.online_learning_batch_size,
            target_update_interval=args.online_learning_target_update_interval,
            q_func_factory='qr',
            optim_factory=d3rlpy.models.optimizers.RMSpropFactory(),
            scaler='pixel',
            use_gpu=True,
            n_critics = num_critics,
        )

        # copy Q-function from the previous offline learning phase
        ddqn.build_with_env(env)
        if cql is not None:
            ddqn.copy_q_function(cql)
            ddqn.reset_optimizer_states() # might not be necessary
            ddqn._impl.update_target()

        explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0, end_epsilon=0.1, duration=500) # 500/10k, 2k/100k, 1M/50M

        # start online training
        num_epochs_online_learning = num_steps_per_phase // args.num_steps_per_epoch
        n_steps_phase = num_steps_per_phase if offline_bootstrap_phase_idx > 1 else num_steps_per_phase + update_start_step
        ddqn.fit_online(
            env,
            buffer,
            explorer,
            n_steps=n_steps_phase, # number of max steps for online learning at the current phase
            n_steps_per_epoch=args.num_steps_per_epoch,
            update_interval=1, # update every interaction
            update_start_step=update_start_step,
            eval_env=eval_env, # 10 episodes are evaluated for each epoch. To modify the number episodes to evaluate, it needs to pass the info through this line fit_online()->train_single_env()->evaluate_on_environment()
            eval_epsilon=0.01,
            save_interval = num_epochs_online_learning, # save a model at the end of each online learning phase
            experiment_name = experiment_name_online_algo,
            show_progress = args.show_progress,
        )

        # Offline training
        # prepare algorithm
        print(f"Offline Learning: Phase {offline_bootstrap_phase_idx}/{args.num_offline_learning_bootstrap}")
        num_transitions_in_buffer = buffer.size()
        num_steps_per_epoch = num_transitions_in_buffer // num_epochs_offline_learning
        #if num_transitions_in_buffer % num_epochs_offline_learning != 0:
        #    num_steps_per_epoch += 1
        cql = d3rlpy.algos.DiscreteCQL(
            learning_rate=args.offline_learning_rate,
            n_frames=args.stack_frames,
            batch_size=args.batch_size,
            target_update_interval=args.offline_learning_target_update_interval,
            q_func_factory='qr',
            scaler='pixel',
            use_gpu=True,
            n_critics = num_critics,
        )

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
            use_gpu=args.gpu)

        # Bootstrap the offline learning with the current online-learned policy:
        #   copy Q-function from the previous offline learning phase
        '''
        cql.build_with_env(env)
        cql.copy_q_function(ddqn)
        cql.reset_optimizer_states() # might not be necessary
        cql._impl.update_target()
        '''

        # start training
        n_offline_training_steps = num_transitions_in_buffer * 5 # 5X gradient steps of online learning
        cql.fit(
            buffer._transitions._buffer[: num_transitions_in_buffer],
            eval_episodes=[None], 
            n_steps=n_offline_training_steps, # number of training steps
            n_steps_per_epoch=n_offline_training_steps//num_epochs_offline_learning, # number of training steps per epoch
            scorers={
                'environment': eval_env_scorer,
            },
            save_interval=num_epochs_offline_learning, # save the model only at the end of the offline learning
            experiment_name = experiment_name_offline_algo,
            show_progress = args.show_progress, # show progress bar. Set to False when deploying in the cluster to save the log file,
            n_epochs_per_eval = 1, # evaluate the model at the end of each epoch
        )

    print('training finished!')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--algo', type=str, default="double_dqn") # check names of all supported algorightms at https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/__init__.py
    parser.add_argument('--env_name', required=True)
    #parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed_eval', type=int, default=100000)

    parser.add_argument('--num_steps', type=int, default=100000, metavar='N',
                        help='maximum number of training steps during online learning (default: 100000)')
    parser.add_argument('--num_steps_per_epoch', type=int, default=1000, metavar='N',
                        help='number of training steps per epoch during online learning (default: 1000)')
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                        help='number of elapsed epochs when saving a model (default: 10)')
    parser.add_argument('--num_offline_learning_bootstrap', type=int, default=10, metavar='N',
                        help='number of offline learning bootstraps for the entire online learning process(default: 10)')

    parser.add_argument('--stack_frames', type=int, default=4)                    
    parser.add_argument('--online_learning_batch_size', type=int, default=32)
    parser.add_argument('--offline_learning_batch_size', type=int, default=32)
    parser.add_argument('--online_learning_rate', type=float, default=3e-4)
    parser.add_argument('--offline_learning_rate', type=float, default=5e-4)
    parser.add_argument('--online_learning_target_update_interval', type=int, default=200) # 200 gradient steps
    parser.add_argument('--offline_learning_target_update_interval', type=int, default=1000) # 1000 gradient steps
    parser.add_argument('--cql_alpha', type=float, default=4.0)
    parser.add_argument('--show_progress', type=bool, default=False) # 500 gradient steps

    args = parser.parse_args()
    print(f"args: {args}")

    main(args)