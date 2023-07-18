import d3rlpy

def main(args):

    d3rlpy.seed(args.seed)

    # prepare dataset
    dataset, env = d3rlpy.datasets.get_atari(args.env_name) #('breakout-expert-v0') # mixed, medium, expert; Qbert, Pong, Breakout, Seaquest, ASTERIX
    print(f"Dataset {args.env_name}: {dataset.size()} episodes, {dataset.observations.shape[0]} observations")

    env.seed(args.seed)

    cql = d3rlpy.algos.DiscreteCQL(
        learning_rate=args.learning_rate,
        optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
        batch_size=args.batch_size,
        alpha=args.cql_alpha,
        q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(
            n_quantiles=200),
        scaler="pixel",
        n_frames=args.stack_frames,
        target_update_interval=args.target_update_interval,
        reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
        use_gpu=True)

    env_scorer = d3rlpy.metrics.evaluate_on_environment(env, n_trials=args.eval_episode_num, epsilon=0.001)


    n_steps=50000000 // 8, # divide by 4 - 100 epochs, divide by 8 - 50 epochs
    n_steps_per_epoch=125000, # 50000000 // 4 // 125000 = 100 epochs
    save_interval = n_steps//n_steps_per_epoch # save the model after the last epoch
    cql.fit(
        dataset,
        eval_episodes=[None],
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        scorers={
            'environment': env_scorer,
        },
        experiment_name=f"DiscreteCQL_{args.env_name}_Seed{args.seed}",
        save_interval=save_interval,
        show_progress = True, # disable the progress bar calculation in tqdm
    )

    print('training finished!')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', required=True)
    #parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')

    parser.add_argument('--stack_frames', type=int, default=4)                    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--cql_alpha', type=float, default=4.0)
    parser.add_argument('--target_update_interval', type=int, default=2000)

    args = parser.parse_args()
    print(f"args: {args}")

    main(args)
