import random
import numpy as np
import torch
import gym
import d3rlpy
from d3rlpy.envs import ChannelFirst
from d3rlpy.algos import create_algo

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)

def main(args):
    # prepare environment
    env = ChannelFirst(gym.make(args.env_name)) # 'Pong-v0'
    eval_env = ChannelFirst(gym.make(args.env_name))

    if args.seed is not None:
        set_seed(args.seed, env=env)
    if eval_env is not None:
        eval_env.seed(args.seed_eval)

    num_critics = 2
    # prepare algorithm
    model = create_algo(
            args.algo,
            True, # discrete action space
            learning_rate=args.learning_rate,
            n_frames=args.stack_frames,
            batch_size=args.batch_size,
            target_update_interval=args.target_update_interval,
            q_func_factory='qr',
            scaler='pixel',
            use_gpu=True,
            n_critics = num_critics,
    )

    # prepare replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)

    # start training
    experiment_name_online_algo = f"{args.env_name}_seed{args.seed}_online{model.__class__.__name__}"

    model.fit_online(
        env,
        buffer,
        n_steps=args.num_steps,
        n_steps_per_epoch=args.num_steps_per_epoch,
        update_interval=1, # update every 1 step
        eval_env=eval_env, # 10 episodes are evaluated for each epoch. To modify the number episodes to evaluate, it needs to pass the info through this line fit_online()->train_single_env()->evaluate_on_environment()
        save_interval = args.save_interval,
        show_progress = False, # disable progress bar calculation in tqdm
        experiment_name = experiment_name_online_algo,
    )

    # save the replay buffer
    buffer_dataset_filename = f"buffer_{args.env_name}_seed{args.seed}_online{args.algo}_steps{args.num_steps}.h5"
    buffer_dataset_path = os.path.join(
        model.active_logger._logdir,
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
    parser.add_argument('--seed_eval', type=int, default=100000)

    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='maximum number of training steps (default: 1000000)')
    parser.add_argument('--num_steps_per_epoch', type=int, default=10000, metavar='N',
                        help='number of training steps per epoch (default: 10000)')
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                        help='number of elapsed epochs when saving a model (default: 10)')

    parser.add_argument('--stack_frames', type=int, default=4)                    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--target_update_interval', type=int, default=5000)

    args = parser.parse_args()
    print(f"args: {args}")

    main(args)
