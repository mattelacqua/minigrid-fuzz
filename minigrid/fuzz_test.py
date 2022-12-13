#!/usr/bin/env python3
import gymnasium as gym

from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

import random
import time

from fuzzing import genetic_algorithm, initialize_population

class Fuzz:
    def __init__(
        self,
        env: MiniGridEnv,
        agent_view: bool = False,
        window: Window = None,
        seed=None,
    ) -> None:
        self.env = env
        self.agent_view = agent_view
        self.seed = seed

        if window is None:
            window = Window("minigrid - " + str(env.__class__))
        self.window = window
        self.window.reg_key_handler(self.key_handler)

    def start(self, render=False):
        """Start the window display with blocking event loop"""
        self.reset(self.seed, render=render)
        if render:
            self.window.show(block=False)

    def step(self, action: MiniGridEnv.Actions, render=False):
        _, reward, terminated, truncated, _ = self.env.step(action)
        if render: 
            print(f"action={action.name} step={self.env.step_count}, reward={reward:.2f}")
            print(f"carrying={self.env.carrying}")

        if terminated:
            if render: 
                print("terminated!")
            #self.reset(self.seed)
        elif truncated:
            if render:
                print("truncated!")
            #self.reset(self.seed)

        if render:
            self.redraw()
        
        return _, reward, terminated, truncated, _

    def redraw(self):
        frame = self.env.get_frame(agent_pov=self.agent_view)
        self.window.show_img(frame)

    def reset(self, seed=None, render=False):
        self.env.reset(seed=seed)

        #if hasattr(self.env, "mission"):
        #    print("Mission: %s" % self.env.mission)
        #    self.window.set_caption(self.env.mission)

        if render:
            self.redraw()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.window.close()
            return
        if key == "backspace":
            self.reset()
            return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-MultiRoom-N6-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    parser.add_argument("--generations", default=1, type=int)  #  Set the default number of generations to run for
    parser.add_argument("--crossover", default=.7, type=float)  #  Set the crossover probability
    parser.add_argument("--mutation", default=.1, type=float)  #  Set the mutation probability
    parser.add_argument("--pop-size", default=4, type=int)  #  Set the size of the initial population. Must be at least 4.

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(args.env, tile_size=args.tile_size)

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    fuzz = Fuzz(env, agent_view=args.agent_view, seed=args.seed)

    generational_stats = []
    # Set initial values to choose from
    legal_moves = list(fuzz.env.Actions)[0:6] # First 3 actions for lava test
    population = initialize_population(fuzz, args.pop_size, legal_moves)

    max_fitness, max_fitness_trace, max_fitness_trial, generational_stats, population, final_fitness, final_population, winners = \
        genetic_algorithm(fuzz, legal_moves, population, args.generations, args.crossover, args.mutation, generational_stats, 0)

    #for stat in generational_stats:
    #    print(stat)
    #print(f"Population: {population}")
    #print(f"Final_Population: {final_population}")
    #print(f"Final_Fitness: {final_fitness}")

    print(f"Max_Fitness_Trial = {max_fitness_trial}")
    print(f"Max_Fitness_Trace = {max_fitness_trace}")
    print(f"Max_Fitness = {max_fitness}")
    

    # Play out the winning scenaro
    for winner in winners:
        print(winner)
        fuzz.start(render=True)
        for move in max_fitness_trace:
            time.sleep(0.25)
            fuzz.step(move, render=True) 


    


