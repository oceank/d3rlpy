import d3rlpy
import gym
from d3rlpy.envs import ChannelFirst

# prepare environment
env = ChannelFirst(gym.make('Pong-v0')) #('breakout-expert-v0')
eval_env = ChannelFirst(gym.make('Pong-v0'))

# prepare algorithm
ddqn = d3rlpy.algos.DoubleDQN(
    n_frames=4,
    batch_size=256,
    target_update_interval=5000,
    q_func_factory='qr',
    scaler='pixel',
    use_gpu=True,
)

# prepare replay buffer
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)

# start training
ddqn.fit_online(
    env,
    buffer,
    n_steps=1000000,
    n_steps_per_epoch=10000,
    update_interval=1,
    eval_env=eval_env,
    save_interval = 10)

print('training finished!')