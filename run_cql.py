import d3rlpy
#from sklearn.model_selection import train_test_split

# prepare dataset
dataset, env = d3rlpy.datasets.get_atari('breakout-medium-v0') #('breakout-expert-v0') # mixed, medium, expert; Qbert, Pong, Breakout, Seaquest, ASTERIX
print(f"observation shape: {dataset.observations.shape}")


# split dataset
#train_episodes, test_episodes = train_test_split(dataset, test_size=0.1) #0.1
train_episodes = dataset.episodes
test_episodes = [train_episodes[0]]

# prepare algorithm
cql = d3rlpy.algos.DiscreteCQL(
    n_frames=4,
    batch_size=256,
    target_update_interval=5000,
    q_func_factory='qr',
    scaler='pixel',
    use_gpu=True,
)

# start training
cql.fit(
    train_episodes,
    eval_episodes=test_episodes, # in order to trigger evaluations
    #n_epochs=10,
    n_steps=1000000,
    n_steps_per_epoch=10000,
    scorers={
        'environment': d3rlpy.metrics.evaluate_on_environment(env),
        #'td_error': d3rlpy.metrics.td_error_scorer,
    },
    save_interval=10,
)

print('training finished!')