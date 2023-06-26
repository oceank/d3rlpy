import random
import numpy as np
import torch
import gym
import d3rlpy
from d3rlpy.envs import ChannelFirst
import gc

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)

def main(args):
    total_offline_learnings = args.num_steps // args.num_steps_per_offline_learning
    if args.num_steps % args.num_steps_per_offline_learning != 0:
        total_offline_learnings += 1

    buffer_max_size = 100000
    buffer = None
    pre_cql = None


    # prepare environment
    env = ChannelFirst(gym.make(args.env_name)) # 'Pong-v0'
    eval_env = ChannelFirst(gym.make(args.env_name))

    if args.seed is not None:
        set_seed(args.seed, env=env)
    if eval_env is not None:
        eval_env.seed(args.seed_eval)

    online_algo_name = "DDQN"
    offline_algo_name = "CQL"

    for offline_learning_idx in range(1, total_offline_learnings+1):

        print(f"Offline Learning For Online Learning: Phase {offline_learning_idx}/{total_offline_learnings}")
    
        start_step = (offline_learning_idx-1)*args.num_steps_per_offline_learning
        end_step = offline_learning_idx*args.num_steps_per_offline_learning if offline_learning_idx < total_offline_learnings else args.num_steps
        # experiment_name: online_<algo1>_with_offline_<algo2>_on_<env_name>_seed_<seed>
        #    <env_name>_seed<seed>_<of4on/ofAon>_online<algo1>_start<Step1>_end<Step2>
        #    <env_name>_seed<seed>_<of4on/ofAon>_offline<algo2>_start<Step1>_end<Step2>
        of_and_on_flag = "of4on"
        experiment_name_online_algo = f"{args.env_name}_seed{args.seed}_{of_and_on_flag}_online{online_algo_name}_start{start_step}_end{end_step}"
        experiment_name_offline_algo = f"{args.env_name}_seed{args.seed}_{of_and_on_flag}_offline{offline_algo_name}_start{start_step}_end{end_step}"

        # Online Training
        # prepare algorithm for online learning
        print(f"Online Learning: Phase {offline_learning_idx}/{total_offline_learnings}")
        ddqn = d3rlpy.algos.DoubleDQN(
            learning_rate=args.learning_rate,
            n_frames=args.stack_frames,
            batch_size=args.batch_size,
            target_update_interval=args.target_update_interval,
            q_func_factory='qr',
            scaler='pixel',
            use_gpu=True,
            init_q_func = pre_cql._impl.q_function if pre_cql is not None else None,
        )

        # prepare replay buffer
        buffer = buffer if buffer is not None else d3rlpy.online.buffers.ReplayBuffer(maxlen=buffer_max_size, env=env)

        # start online training
        ddqn.fit_online(
            env,
            buffer,
            n_steps=args.num_steps_per_offline_learning, # number of max steps for online learning at the current phase
            n_steps_per_epoch=args.num_steps_per_epoch,
            update_interval=1, # update every 1 step
            eval_env=eval_env, # 10 episodes are evaluated for each epoch. To modify the number episodes to evaluate, it needs to pass the info through this line fit_online()->train_single_env()->evaluate_on_environment()
            save_interval = args.save_interval,
            experiment_name = experiment_name_online_algo,
            show_progress = False, # show progress bar. Set to False when deploying in the cluster to save the log file
        )

        # Offline training
        # prepare algorithm
        print(f"Offline Learning: Phase {offline_learning_idx}/{total_offline_learnings}")
        cql = d3rlpy.algos.DiscreteCQL(
            learning_rate=args.learning_rate,
            n_frames=args.stack_frames,
            batch_size=args.batch_size,
            target_update_interval=args.target_update_interval,
            q_func_factory='qr',
            scaler='pixel',
            use_gpu=True,
            init_q_func = ddqn._impl.q_function,
        )

        # start training
        num_transitions_in_buffer = buffer.size()
        num_epochs_offline_learning = 100
        num_steps_per_epoch = num_transitions_in_buffer // num_epochs_offline_learning
        if num_transitions_in_buffer % num_epochs_offline_learning != 0:
            num_steps_per_epoch += 1
        cql.fit(
            buffer._transitions._buffer[: num_transitions_in_buffer],
            eval_episodes=None, 
            n_steps=num_transitions_in_buffer, # number of transitions in the buffer
            n_steps_per_epoch=num_steps_per_epoch, # number of processed transitions per epoch (or per evaluation)
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(eval_env, n_trials=args.eval_episode_num),
            },
            save_interval=num_epochs_offline_learning, # save the model only at the end of the offline learning
            eval_only_env = True, # in order to trigger evaluation on the environment
            show_progress = False
        )

        pre_cql = cql
        gc.collect()

    print('training finished!')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--algo', type=str, default="double_dqn") # check names of all supported algorightms at https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/__init__.py
    parser.add_argument('--env_name', required=True)
    #parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed_eval', type=int, default=10000)

    parser.add_argument('--num_steps', type=int, default=100000, metavar='N',
                        help='maximum number of training steps (default: 1000000)')
    parser.add_argument('--num_steps_per_epoch', type=int, default=1000, metavar='N',
                        help='number of training steps per epoch (default: 10000)')
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                        help='number of elapsed epochs when saving a model (default: 10)')

    parser.add_argument('--num_steps_per_offline_learning', type=int, default=10000, metavar='N',
                        help='number of training steps per offline learning (default: 10000)')

    parser.add_argument('--stack_frames', type=int, default=4)                    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--target_update_interval', type=int, default=500) # 500 gradient steps

    args = parser.parse_args()
    print(f"args: {args}")

    main(args)