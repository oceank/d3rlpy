import random
import d3rlpy
import numpy as np
import torch

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)

def main(args):

    # prepare dataset
    dataset, env = d3rlpy.datasets.get_atari(args.env_name) #('breakout-expert-v0') # mixed, medium, expert; Qbert, Pong, Breakout, Seaquest, ASTERIX
    print(f"observation shape: {dataset.observations.shape}")

    train_episodes = dataset.episodes
    test_episodes = [train_episodes[0]]

    if args.seed is not None:
        set_seed(args.seed, env=env)
    if env is not None:
        env.seed(args.seed_eval)

    # prepare algorithm
    cql = d3rlpy.algos.DiscreteCQL(
        learning_rate=args.learning_rate,
        n_frames=args.stack_frames,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        q_func_factory='qr',
        scaler='pixel',
        use_gpu=True,
    )

    # start training
    cql.fit(
        train_episodes,
        eval_episodes=test_episodes, # in order to trigger evaluations
        n_steps=args.num_steps,
        n_steps_per_epoch=args.num_steps_per_epoch,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env, n_trials=args.eval_episode_num),
        },
        save_interval=args.save_interval,
        show_progress = False, # disable the progress bar calculation in tqdm
    )

    print('training finished!')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', required=True)
    #parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed_eval', type=int, default=10000)

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
