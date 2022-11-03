#!/usr/bin/env python3
from argparse import Action
import gymnasium as gym

from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

import random
import time

import fuzzing

class DFS:
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

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(args.env, tile_size=args.tile_size)

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    dfs = DFS(env, agent_view=args.agent_view, seed=args.seed)

    
    #time.sleep(500)
    traces = []
    preferences = ['u', 'd', 'l', 'r']
    for i in preferences:
        dfs.start(render=True)
        actions = fuzzing.get_first_trace(dfs, preference_order=i)

        dfs.start(render=True)
        time.sleep(2)
        for action in actions:
            time.sleep(0.5)
            dfs.step(action, render=True)

