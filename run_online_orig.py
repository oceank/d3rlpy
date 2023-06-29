import argparse
import gym
import d3rlpy
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--show_progress', action='store_true')
    args = parser.parse_args()

    # get wrapped atari environment
    env = d3rlpy.envs.Atari(gym.make(args.env_name))
    eval_env = d3rlpy.envs.Atari(gym.make(args.env_name), is_eval=True)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    num_critics = 2
    # setup algorithm
    ddqn = d3rlpy.algos.DoubleDQN(
        batch_size=32,
        learning_rate=2.5e-4,
        optim_factory=d3rlpy.models.optimizers.RMSpropFactory(),
        target_update_interval=1000,
        q_func_factory='mean',
        scaler='pixel',
        n_frames=4,
        use_gpu=args.gpu,
        num_critics = 2)

    # replay buffer for experience replay
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

    # epilon-greedy explorer
    explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1, duration=2000)

    n_steps = 100000
    n_steps_per_epoch = 10000
    experiment_name_online_algo = f"{args.env_name}_seed{args.seed}_online{ddqn.__class__.__name__}_baselineOrig"
    # start training
    ddqn.fit_online(env,
                    buffer,
                    explorer,
                    eval_env=eval_env,
                    eval_epsilon=0.01,
                    n_steps=n_steps,
                    n_steps_per_epoch=n_steps_per_epoch,
                    update_interval=1,
                    update_start_step=1000,
                    save_interval = (n_steps//n_steps_per_epoch),
                    show_progress = args.show_progress, # disable progress bar calculation in tqdm
                    experiment_name = experiment_name_online_algo,)


    # save the replay buffer
    buffer_dataset_filename = f"buffer_{args.env_name}_seed{args.seed}_online{ddqn.__class__.__name__}_steps{num_steps}_baselineOrig.h5"
    buffer_dataset_path = os.path.join(
        ddqn.active_logger._logdir,
        buffer_dataset_filename,
    )
    print(f"Saving the buffer dataset to {buffer_dataset_path}")
    buffer_dataset = buffer.to_mdp_dataset()
    buffer_dataset.dump(fname=buffer_dataset_filename)

    print('training finished!')

if __name__ == '__main__':
    main()
