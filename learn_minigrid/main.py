import os
import datetime
import logging
import configparser
import pickle
from pathlib import Path

import minigrid
import torch
from minigrid.wrappers import FullyObsWrapper, ReseedWrapper, ImgObsWrapper, RGBImgObsWrapper
from wrappers import ResizeObservation, SkipFrame
from gym.wrappers import TransformObservation
from agent import Agent
import numpy as np

import util
from fuzzing import fuzz, search

import random
import gymnasium as gym
from gym import spaces
from metrics import MetricLogger, EvaluationLogger
from agent import Agent


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

log = logging.getLogger("FooBar")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s | %(levelname)-10s | %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
log.addHandler(handler)

params = configparser.ConfigParser()
seed = 3

eval_logger = None

def main(eval_mode = False):
    global params
    global seed
   
    params.read('learn_minigrid/params.ini')
    seed = params.getint('MODE_FUZZ', 'SEED')

    log_level = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }

    # Set Log Level
    log.setLevel(log_level.get(params.get('LOGGING', 'LOG_LEVEL'), logging.DEBUG))

    # Initialize Minigrid environment
    env, unwrapped_env = setup_env()
    
    # Run the Training
    run(env, unwrapped_env,eval_mode)

def setup_env():
    global params
    global seed
    
    env = gym.make(f"MiniGrid-{params.get('SETUP', 'STAGE')}-{params.get('SETUP', 'STYLE')}")
    unwrapped_env = env.env
    # due to an episode limit, make in the above line returns TimeLimit environment,
    # so to get the mario environment directly, we need to unwrap

    # Limit the action-space
    env.action_space = spaces.Discrete(params.getint('SETUP', 'ACTION_SPACE'))
    #env = action_space.get(params.get('SETUP', 'ACTION_SPACE'), JoypadSpace(env, actions.SIMPLE_MOVEMENT))
    #env = action_space.get(params.get('SETUP', 'ACTION_SPACE'))
    #print(env)
    #exit(0)
    
    # Apply Wrappers to environment
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = ResizeObservation(env, shape=84)
    env = ReseedWrapper(env, seeds=[seed])

    #print(env.observation_space)

    # THIS SHOULD BE PARAMS. SEED
    #unwrapped_env.env.render_mode = 'human'
    env.reset()
    return env, unwrapped_env


def run(env, unwrapped_env, eval_mode):
    global params
    global eval_logger
    global seed
    
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)
    eval_logger = EvaluationLogger(save_dir, eval_mode)

    checkpoint_path = params.get('TRAINING', 'CHECKPOINT')
    checkpoint = Path(checkpoint_path) if checkpoint_path != 'None' else None
    load_only_conv = params.getboolean('TRAINING', 'LOAD_ONLY_CONV')

    if checkpoint and not checkpoint.exists():
        log.fatal("Checkpoint not found")
        exit(-1)

    agent = Agent(state_dim=env.observation_space.shape, action_dim=env.action_space.n, save_dir=save_dir, params=params,
                  checkpoint=checkpoint,load_only_conv=load_only_conv, env_w=env.width, env_h=env.height)

    if eval_mode:
        evaluate_training(env, unwrapped_env, agent, 20)
    else:
        pretrain_steps = params.getint('TRAINING', 'PRETRAIN_STEPS')
        if pretrain_steps > 0:
            search_trace = run_search(env, unwrapped_env)
            print("FIRST")
            print(search_trace)
            fuzz_traces = run_fuzz(env, unwrapped_env, search_trace)
            #load_expert_memory_from_traces(mario, env, search_trace)
            load_expert_memory_from_traces(agent, env, fuzz_traces, dump=True)

            #mario.dump_expert_memory()
            #mario.load_expert_memory()

            pretrain_agent(agent, pretrain_steps)
        train_agent(agent, env, unwrapped_env)
    
def scale_reward(reward):
    return reward / 100 # np.sign(reward) * np.log(1+np.abs(reward))
    
def load_expert_memory_from_traces(mario, env, traces_to_learn,dump=False):
    log.debug("Loading pre-computed trace into expert-memory")
    #if mario.load_expert_memory(params):
    #    return
    
    trace_cnt = 0
    n_step_return = params.getint('TRAINING', 'N_STEP_RET')
    sample_traces = traces_to_learn # random.sample(traces_to_learn,50)
    for trace in sample_traces:
        log.debug(f"Running with pre-computed trace {trace_cnt + 1}")

        # Reset environment
        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()

        episode = []
        for action in trace:
            next_state, reward, done, info = env.step(action)
            reward = scale_reward(reward)
            next_state = torch.from_numpy(np.array(next_state)).float()
            reward = mario.compute_added_reward(info, reward, coin=params.getboolean('TRAINING', 'REWARD_COINS'),
                                                score=params.getboolean('TRAINING', 'REWARD_SCORE'))
            episode.append((state, next_state, action, reward, done))
            state = next_state
            if done:
                break

        for i in range(len(episode)):
            (state, next_state, action, reward, done) = episode[i]
            if i + n_step_return < len(episode):
                last_state = episode[i+n_step_return-1][1] if not episode[i+n_step_return-1][4] else None
                n_rewards = (list(map(lambda step : step[3],episode[i:i+n_step_return])),last_state)
            else:
                last_state = None
                n_rewards = (list(map(lambda step : step[3],episode[i:])),last_state)
            mario.cache_expert(state, next_state, action, reward, n_rewards, done)
            #mario.cache_expert(state, next_state, action, reward, done)

        trace_cnt += 1
    if dump:
        mario.dump_expert_memory(params)
        
def run_search(env, unwrapped_env):
    global params

    if params.getboolean('MODE_SEARCH', 'ENABLE'):
        log.info("Running Mode: SEARCH")

        # Load previously generated trace
        if params.getboolean('MODE_SEARCH', 'LOAD_SAVED_TRACE'):
            # Get path
            search_trace_load_path = params.get('MODE_SEARCH', 'LOAD_PATH')
            prev_search_trace = Path(search_trace_load_path)

            # Check if file exists
            if not prev_search_trace.exists():
                log.fatal("Previous search trace not found")
                exit(-1)

            log.info(f"Loading previously generated trace at {search_trace_load_path}")

            # Load saved path
            with open(prev_search_trace, 'rb') as file:
                search_trace = pickle.load(file)

        # Generate new path
        else:
            # Reset environment to start
            log.debug("Resetting environment")
            env.reset()

            # Run the algorithm
            log.debug("Generating search traces")
            search_trace = search(env, unwrapped_env)
            #print(f"Search trace {search_trace}")
            log.debug("Search traces generated")
            # run_trace(unwrapped_env, env, search_trace, True)

            # Save the generated trace
            if params.getboolean('MODE_SEARCH', 'SAVE_GENERATED_TRACE'):
                search_trace_save_path = params.get('MODE_SEARCH', 'SAVE_PATH')
                save_search_trace = Path(search_trace_save_path)
                save_search_trace.parent.mkdir(parents=True) if not save_search_trace.parent.exists() else None

                with open(save_search_trace, 'wb') as file:
                    pickle.dump(search_trace, file)
    else:
        return None

    return search_trace


def run_fuzz(env, unwrapped_env, search_trace):
    global params

    # Extract search trace from list, this is necessary for other code parts to work
    #search_trace = search_trace[0]

    if params.getboolean('MODE_FUZZ', 'ENABLE'):
        log.info("Running Mode: FUZZ")
        # Load previously generated trace
        if params.getboolean('MODE_FUZZ', 'LOAD_SAVED_TRACE'):
            # Get path
            fuzz_trace_load_path = params.get('MODE_FUZZ', 'LOAD_PATH')
            prev_fuzz_trace = Path(fuzz_trace_load_path)

            # Check if file exists
            if not prev_fuzz_trace.exists():
                log.fatal("Previous fuzz trace not found")
                exit(-1)

            log.info(f"Loading previously generated trace at {fuzz_trace_load_path}")

            # Load saved path
            with open(prev_fuzz_trace, 'rb') as file:
                success_traces = pickle.load(file)
         
        else:
            # Generate search trace, which is necessary for fuzzing - if not exists
            if search_trace is None:
                log.debug("No previous search trace found, generating a new one")
                search_trace = search(env, unwrapped_env)
                log.debug("Search trace generated")

            # Reset environment to start
            log.debug("Resetting environment")
            env.reset()

            # Run the algorithm
            #action_indexes = []
            #action_meanings = env.get_action_meanings()
            #for i in range(len(action_meanings)):
            #    meaning = action_meanings[i]
            #    if ('right' in meaning) or ('down' in meaning) or ('NOOP' in meaning):
            #        action_indexes.append(i)
            #print(action_indexes)
            fuzzing_generations = params.getint('MODE_FUZZ', 'GENERATIONS')
            success_traces = fuzz(unwrapped_env, env, search_trace, population_size=10, num_generations=fuzzing_generations,
                                     elite_cull_ratio=0.20, probability_crossover=0.8, probability_mutation=0.5)

            # Save the generated traces
            if params.getboolean('MODE_FUZZ', 'SAVE_GENERATED_TRACE'):
                fuzz_trace_save_path = params.get('MODE_FUZZ', 'SAVE_PATH')
                save_fuzz_trace = Path(fuzz_trace_save_path)
                save_fuzz_trace.parent.mkdir(parents=True) if not save_fuzz_trace.parent.exists() else None

                #success_traces.extend(best_traces)
                with open(save_fuzz_trace, 'wb') as file:
                    pickle.dump(success_traces, file)
            #print("=="*40)
            #print("Run trace steps after fuzzing: ", run_trace_steps)
            #print("=="*40)
            print(f"Fuzzed traces.")


        return success_traces
    else:
        return None

def pretrain_agent(agent, runs):
    n_refresh_expert = params.getint('TRAINING', 'REFRESH_EXPERT') * 50
    for i in range(runs):
        agent.pretrain()
        if n_refresh_expert > 0 and i > 0 and i % n_refresh_expert == 0:
            agent.refresh_expert_cache()
        
def train_agent(agent, env, unwrapped_env):
    global params

    # Read settings for RL mode
    do_render = params.getboolean("ADMIN", "RENDER")
    episodes = params.getint('TRAINING', 'EPISODES')
    n_step_return = params.getint('TRAINING', 'N_STEP_RET')
    refresh_expert_cache = params.getint('TRAINING', 'REFRESH_EXPERT') # needed for n_step_return

    logger = MetricLogger(agent.save_dir)

    """
    MARIO RL MODE
    """

    log.debug("Starting RL Mode")

    # Start with episode 1
    # For better readability
    agent.curr_episode += 1

    # Train the model until max. episodes is reached
    while agent.curr_episode <= episodes:
        e = agent.curr_episode
        log.debug(f"Running episode {e}")

        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()

        episode = []

        # Play the game!
        while True:
            if do_render:
                env.render()

            # Pick an action
            action = agent.act(state)

            # Perform action
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(np.array(next_state)).float()
            reward = agent.compute_added_reward(info, reward, coin=params.getboolean('TRAINING', 'REWARD_COINS'),
                                                score=params.getboolean('TRAINING', 'REWARD_SCORE'))

            reward = scale_reward(reward)
            episode.append((state, next_state, action, reward, done))

            # Update state
            state = next_state

            # Check if end of game
            if done or info['flag_get']:
                break

        for i in range(len(episode)):
            (state, next_state, action, reward, done) = episode[i]
            # Remember
            if i + n_step_return < len(episode):
                last_state = episode[i+n_step_return-1][1]
                n_rewards = (list(map(lambda step : step[3],episode[i:i+n_step_return])),last_state)
            else:
                last_state = None
                n_rewards = (list(map(lambda step : step[3],episode[i:])),last_state)
            agent.cache(state, next_state, action, reward, n_rewards, done)
            # Learn
            q, loss = agent.learn()

            # Log
            logger.log_step(reward, loss,q)

        if refresh_expert_cache > 0 and e % refresh_expert_cache == 0 and e > 0:
            agent.refresh_expert_cache()
        
        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=agent.exploration_rate,
                step=agent.curr_step
            )

        if e % agent.save_every == 0:
            agent.save(params)

        if e % agent.eval_every == 0:
            evaluate_training(env, unwrapped_env, agent, 20)

        agent.curr_episode += 1

    log.info("Training done, saving...")
    agent.save(params)


def evaluate_training(env, unwrapped_env, agent, episodes=100):
    global params
    global eval_logger

    assert isinstance(eval_logger, EvaluationLogger)

    for ep in range(episodes):
        log.debug(f"Running eval cycle {ep + 1}")
        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()

        while True:
            action = agent.act(state, True)
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(np.array(next_state)).float()
            eval_logger.log_step(reward)

            state = next_state
            if done or info['flag_get']:
                break

        eval_logger.log_episode()

    eval_logger.log_evaluation_cycle(agent, 0)


if __name__ == '__main__':
    import sys
    if "eval" in sys.argv:
        main(eval_mode = True)
    else:
        main(eval_mode = False)
