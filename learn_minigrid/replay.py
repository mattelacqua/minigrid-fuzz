import random, datetime
from pathlib import Path

import minigrid
from minigrid.wrappers import FullyObsWrapper

from metrics import MetricLogger
from agent import Agent 

env = minigrid.make('MiniGrid-UnlockPickup-v0')

env = FullyObsWrapper(env)


env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
agent = Agent(state_dim=env.observation_space, action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
agent.exploration_rate = agent.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()

        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        agent.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )
