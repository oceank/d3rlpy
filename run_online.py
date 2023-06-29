import random
import os
import numpy as np
import torch
import gym
import d3rlpy
from d3rlpy.envs import ChannelFirst
from d3rlpy.algos import create_algo


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
    set_seed(args.seed, env=env, eval_env=eval_env)
    eval_env_scorer = d3rlpy.metrics.evaluate_on_environment(eval_env, n_trials=args.eval_episode_num, epsilon=0.001)

    num_steps_per_epoch_online_learning = args.num_steps // args.num_online_epochs
    num_critics = 2
    buffer_max_size = 100000 # 100k transitions
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=buffer_max_size, env=env)

    ddqn = d3rlpy.algos.DoubleDQN(
        learning_rate=args.online_learning_rate,
        n_frames=args.stack_frames,
        batch_size=args.online_learning_batch_size,
        target_update_interval=args.online_learning_target_update_interval,
        q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=200),
        optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
        reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
        scaler='pixel',
        use_gpu=True,
        n_critics = num_critics,
    )
    
    experiment_name_online_algo = f"{args.env_name}_seed{args.seed}_online{ddqn.__class__.__name__}_baselineAdam"
    
    # epilon-greedy explorer
    explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1, duration=args.greedy_epsilon_exploration_duration)

    # start training
    ddqn.fit_online(
        env,
        buffer,
        explorer,
        n_steps=args.num_steps,
        n_steps_per_epoch=num_steps_per_epoch_online_learning,
        update_interval=1, # update every 1 step
        eval_env=eval_env, # 10 episodes are evaluated for each epoch. To modify the number episodes to evaluate, it needs to pass the info through this line fit_online()->train_single_env()->evaluate_on_environment()
        eval_epsilon=0.01,
        update_start_step=args.update_start_step_online_learning,
        save_interval = args.num_online_epochs, # save the model when the last epoch finishes
        show_progress = args.show_progress, # disable progress bar calculation in tqdm
        experiment_name = experiment_name_online_algo,
    )

    # save the replay buffer
    buffer_dataset_filename = f"buffer_{args.env_name}_seed{args.seed}_online{ddqn.__class__.__name__}_steps{args.num_steps}_baselineAdam.h5"
    buffer_dataset_path = os.path.join(
        ddqn.active_logger._logdir,
        buffer_dataset_filename,
    )
    print(f"Saving the buffer dataset to {buffer_dataset_path}")
    buffer_dataset = buffer.to_mdp_dataset()
    buffer_dataset.dump(fname=buffer_dataset_filename)

    print('training finished!')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--algo', type=str, default="double_dqn") # check names of all supported algorightms at https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/__init__.py
    parser.add_argument('--env_name', required=True)
    #parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num_steps', type=int, default=100000, metavar='N',
                        help='maximum number of training steps (default: 100000 (100k))')
    parser.add_argument('--num_online_epochs', type=int, default=10, metavar='N',
                        help='number of training epochs per phase during online learning (default: 10)')
    parser.add_argument('--eval_episode_num', type=int, default=32,
                        help='Number of evaluation episodes (default: 32)')

    parser.add_argument('--stack_frames', type=int, default=4)          
    parser.add_argument('--online_learning_batch_size', type=int, default=32)
    parser.add_argument('--online_learning_rate', type=float, default=3e-4)
    parser.add_argument('--online_learning_target_update_interval', type=int, default=1000) # 1000 gradient steps
    parser.add_argument('--greedy_epsilon_exploration_duration', type=int, default=2000) # 2000 gradient steps
    parser.add_argument('--update_start_step_online_learning', type=int, default=1000) # 1000 gradient steps
    parser.add_argument('--show_progress', type=bool, default=False)

    args = parser.parse_args()
    print(f"args: {args}")

    main(args)
