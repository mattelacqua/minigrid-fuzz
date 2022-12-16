import logging
import pickle
from pathlib import Path

import torch
import random
import numpy as np

from neural import MinigridNet
from collections import deque

from torch.autograd import Variable
log = logging.getLogger("FooBar")


class Agent:

    def __init__(self, state_dim, action_dim, save_dir, params, checkpoint=None, load_only_conv=False, env_w=0, env_h=0):
        global log

        log.debug("Initializing Minigrid Agent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=params.getint('TRAINING', 'MEMORY_SIZE'))

        # Expert memory can grow indefinitely
        self.expert_memory = deque()
        self.use_cuda = torch.cuda.is_available()
        self.use_mps = torch.has_mps
        #print(self.use_cuda)
        if self.use_cuda:
            self.device = 'cuda' 
        elif self.use_mps:
            self.device = 'mps' 
        else: 'cpu'
        self.batch_size = params.getint('TRAINING', 'BATCH_SIZE')

        self.pretrain_steps = params.getint('TRAINING', 'PRETRAIN_STEPS')
        self.expert_knowledge_share = 0 if self.pretrain_steps == 0 else 0.5

        # CHANGED from 0.25
        self.cache_device = 'cpu' if params.getint('TRAINING', 'MEMORY_SIZE') > 5000 else self.device 
        self.expert_recall_size = int(self.batch_size * self.expert_knowledge_share)
        self.margin = params.getfloat('TRAINING', 'MARGIN')

        self.exploration_rate = params.getfloat('TRAINING', 'EXPLORATION_RATE_INIT')
        self.exploration_rate_decay = params.getfloat('TRAINING', 'EXPLORATION_RATE_DECAY')
        self.exploration_rate_min = params.getfloat('TRAINING', 'EXPLORATION_RATE_MIN')
        self.gamma = params.getfloat('TRAINING', 'GAMMA')

        self.curr_step = 0
        self.curr_pretrain_step = 0
        self.curr_episode = 0
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 10000
        self.eval_every = params.getint('LOGGING', 'EVAL_EVERY')

        self.save_every = params.getint('LOGGING', 'SAVE_EVERY')
        self.save_dir = save_dir
        self.evaluation_randomness = self.exploration_rate_min
        self.expert_cache_save = []
        
        log.debug(f"Cuda available: {self.use_cuda}")
        log.debug(f"MPS available: {self.use_mps}")

        # Minigrid's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MinigridNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        elif self.use_mps:
            self.net = self.net.to(device='mps')

        if checkpoint:
            log.debug(f"Loading previous checkpoint: {checkpoint}")
            self.load(checkpoint,load_only_conv)

        #lowered lr from 0.00025 to 0.00005
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay = 5e-5) # added weight_decay
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.n_step_loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        log.debug("Minigrid Agent initialized")

    def act(self, state, eval_mode=False):

        # EXPLORE
        if np.random.rand() < (self.exploration_rate if not eval_mode else self.evaluation_randomness):
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.from_numpy(np.array(state)).float().to(self.device)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        if eval_mode:
            return action_idx

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1

        return action_idx
    
    def compute_n_step_return(self,n_rewards, device):
        # n_step_return
        (rewards, last_state) = n_rewards
        n_step_return = 0
            #there is probably a more efficient, batched way to do this
            #last_state_Q = self.net(last_state.to(self.device).unsqueeze(0).repeat(self.batch_size,1,1,1), model='target')[0]
            #last_Q_max = torch.max(last_state_Q, axis=0)[0] 
            #n_step_return = last_Q_max
            
        for r in reversed(rewards):
            n_step_return *= self.gamma
            n_step_return += r
        last_Q_discount = self.gamma ** len(rewards)
        if last_state is None:
            last_Q_discount = 0.0
            last_state = torch.zeros(self.state_dim).to(device).float()
        
        n_step_return = torch.tensor(n_step_return).to(device).float()
        last_Q_discount = torch.tensor(last_Q_discount).to(device).float()
        # n_step_return end
        return (n_step_return, last_Q_discount, last_state.to(device))

    def refresh_expert_cache(self):
        print("Refreshing expert cache")
        self.expert_memory = deque()
        for (state, next_state, action, reward, n_rewards, done) in self.expert_cache_save:
            self._cache_expert(state, next_state, action, reward,n_rewards, done)

    def cache_expert(self, state, next_state, action, reward,n_rewards, done):
        self.expert_cache_save.append((state, next_state, action, reward, n_rewards, done,))
        self._cache_expert(state, next_state, action, reward,n_rewards, done)

    def _cache_expert(self, state, next_state, action, reward,n_rewards, done):
        state = state.to('cpu').float()
        next_state = next_state.to('cpu').float()
        action = torch.tensor(action).to('cpu').long()
        reward = torch.tensor(reward).to('cpu').double()
        n_step_return = self.compute_n_step_return(n_rewards, 'cpu')
        done = torch.tensor(done).to('cpu').bool()
        self.expert_memory.append((state, next_state, action, reward, n_step_return, done,))
    
    def cache(self, state, next_state, action, reward,n_rewards,done):
        state = state.to(self.cache_device).float()
        next_state = next_state.to(self.cache_device).float()
        action = torch.tensor(action).to(self.cache_device).long()
        reward = torch.tensor(reward).to(self.cache_device).double()
        n_step_return = self.compute_n_step_return(n_rewards, self.cache_device)
        done = torch.tensor(done).to(self.cache_device).bool()
        self.memory.append((state, next_state, action, reward,n_step_return, done,))

    def recall(self, bs=None):
        if bs is None:
            replay_batch = random.sample(self.memory, self.batch_size)
        else:
            replay_batch = random.sample(self.memory, bs)

        #state, next_state, action, reward,n_step_return, done = map(torch.stack, zip(*replay_batch))
        state, next_state, action, reward,n_step_return_info, done = zip(*replay_batch)
        n_step_return, last_Q_discount, last_state = zip(*n_step_return_info)
         
        state, next_state, action, reward, done = map(torch.stack, [state, next_state, action, reward, done])
        n_step_return, last_Q_discount, last_state = map(torch.stack, [n_step_return, last_Q_discount, last_state])
        if self.device != self.cache_device:
            return state.to(self.device), next_state.to(self.device), action.to(self.device), reward.to(self.device),(n_step_return.to(self.device), last_Q_discount.to(self.device), last_state.to(self.device)),done.to(self.device)
        else:
            return state, next_state, action, reward,(n_step_return, last_Q_discount, last_state), done

    def expert_recall(self, bs=None):
        if bs is None:
            replay_batch = random.sample(self.expert_memory, self.batch_size)
        else:
            replay_batch = random.sample(self.expert_memory, bs)

        state, next_state, action, reward,n_step_return_info, done = zip(*replay_batch)
        n_step_return, last_Q_discount, last_state = zip(*n_step_return_info)
         
        state, next_state, action, reward, done = map(torch.stack, [state, next_state, action, reward, done])
        n_step_return, last_Q_discount, last_state = map(torch.stack, [n_step_return, last_Q_discount, last_state])
            
        #state, next_state, action, reward,n_step_return, done = map(torch.stack, zip(*replay_batch))
        return state.to(self.device), next_state.to(self.device), action.to(self.device), reward.to(self.device),(n_step_return.to(self.device), last_Q_discount.to(self.device), last_state.to(self.device)),done.to(self.device)

    def td_estimate(self, state, action):
        state = self.net(state, model='online')
        state_action = state[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return state, state_action

    def q_online(self, states, actions):
        state_Q = self.net(states, model='online')[np.arange(0, self.batch_size), actions]
        return state_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def supervised_loss(self, q_state, q_state_action, actions, pt=False):
        # Implementation approach taken from https://github.com/nabergh/doom_dqfd

        ex_s = self.batch_size if pt else self.expert_recall_size

        margins = (torch.ones(self.action_dim, self.action_dim) -
                   torch.eye(self.action_dim)) * self.margin
        state_margins = q_state + margins[actions].to(self.device)
        
        #supervised_loss = (state_margins.max(1)[0].unsqueeze(1) - q_state_action).pow(2)[:ex_s].mean()
        supervised_loss = (state_margins.max(1)[0].unsqueeze(1) - q_state_action).abs()[:ex_s].mean()
        #supervised_loss = state_margins,q_state
        return supervised_loss

    @torch.no_grad()
    def n_step_Q(self, last_state):
        last_state_Q = self.net(last_state, model='online')
        best_action = torch.argmax(last_state_Q, axis=1)
        last_Q = self.net(last_state, model='target')[np.arange(0, self.batch_size), best_action]
        return last_Q


    def n_step_loss(self,td_est, states,actions,n_step_info,pt):
        ex_s = self.batch_size #if pt else self.expert_recall_size
        n_step_return, last_Q_discount, last_state = n_step_info
        last_Q = self.n_step_Q(last_state)        
        n_step_return = n_step_return + last_Q * last_Q_discount
        return self.n_step_loss_fn(n_step_return[:ex_s],td_est[:ex_s])        
    
    def update_Q_online(self, td_estimate, td_target, q_states, states,actions, n_step_info, pt=False):        
        # trying the following line
        #states.requires_grad = True 
        
        dqn_loss = self.loss_fn(td_estimate, td_target)
        l0 = 0
        l1 = 1 # 0.25 # 0.25
        l2 = 1 # 0.25 # 0.25
        ns_loss = self.n_step_loss(td_estimate,states,actions,n_step_info,pt)
        loss = dqn_loss + l1 * ns_loss

        if self.expert_recall_size > 0:
            sup_loss = self.supervised_loss(q_states, td_estimate, actions, pt)
            loss += l2*sup_loss
           
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target_conv.load_state_dict(self.net.online_conv.state_dict())
        self.net.target_linear.load_state_dict(self.net.online_linear.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if len(self.memory) < self.batch_size:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # Sample from memory
        state, next_state, action, reward, ns_return, done = self.mixed_recall()

        # Get TD Estimate
        q_states, td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt, q_states,state, action, ns_return)

        return td_est.mean().item(), loss

    def save(self, params):
        save_path = self.save_dir / f"minigrid_net_{int(self.curr_episode)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        with open(self.save_dir / "params.ini","w") as f:
            params.write(f)
        print(f"MinigridNet saved to {save_path} at step {self.curr_episode}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=self.device)

        log.info(f"Loading previously saved model at {load_path}")
        self.net.load_state_dict(ckp.get('model'))
        self.exploration_rate = ckp.get('exploration_rate')
        #if load_only_conv:
        #    self.net.reset_linear(self.use_cuda)
        #    self.exploration_rate = params.getfloat('TRAINING', 'EXPLORATION_RATE_INIT')
        log.info(f"Successfully loaded model")

    def get_gpu_memory(self):
        if self.use_cuda:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            return f'Total memory: {t / 1e9}GB, reserved: {r / 1e9}GB, allocated: {a / 1e9}GB, free: {(r - a) / 1e9}GB \n'
        return 'Cuda not used \n'

    ### THIS WILL HAVE TO BE CHANGED
    ### TODO 
    def compute_added_reward(self, info, reward, coin=False, score=False):
        # add 5 points per coin
        # add 15/1000 points per score increase
        current_score = info["score"]
        current_coins = info["coins"]
        if coin:
            reward += 5 * (current_coins - self.previous_coins)
        if score:
            reward += (15 / 1000) * (current_score - self.previous_score)
        self.previous_coins = current_coins
        self.previous_score = current_score
        reward = min(15, reward)  # dont need to clip down here
        return reward

    def dump_expert_memory(self,params):
        fuzz_load_path = params.get("MODE_FUZZ","LOAD_PATH")
        save_path = Path(fuzz_load_path.replace('.traces','_init_exp.memory'))

        with open(save_path, 'wb') as file:
            pickle.dump((self.expert_memory,self.expert_cache_save), file)

    def load_expert_memory(self,params):
        fuzz_load_path = params.get("MODE_FUZZ","LOAD_PATH")
        save_path = Path(fuzz_load_path.replace('.traces','_init_exp.memory'))
        if save_path.exists():
            with open(save_path, 'rb') as file:
                (self.expert_memory,self.expert_cache_save) = pickle.load(file)
                return True
        return False

    def pretrain(self):
        if self.curr_pretrain_step % 100 == 0:
            log.debug(f"Pretrain step {self.curr_pretrain_step}")

        if self.curr_pretrain_step % self.sync_every == 0:
            self.sync_Q_target()

        state, next_state, action, reward, ns_return, done = self.expert_recall()

        q_states, td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        self.update_Q_online(td_est, td_tgt, q_states, state, action, ns_return, True)

        td_est.mean().item()

        self.curr_pretrain_step += 1

    def mixed_recall(self):
        expert_batch_size = self.expert_recall_size
        mario_batch_size = self.batch_size - expert_batch_size

        if expert_batch_size > 0:
            ex_state, ex_next_state, ex_action, ex_reward, ex_ns_return, ex_done = self.expert_recall(expert_batch_size)
            state, next_state, action, reward,ns_return, done = self.recall(mario_batch_size)

            ex_n_step_return, ex_last_Q_discount, ex_last_state = ex_ns_return
            n_step_return, last_Q_discount, last_state = ns_return
        
            
            r_state = torch.cat((ex_state, state))
            r_next_state = torch.cat((ex_next_state, next_state))
            r_action = torch.cat((ex_action, action))
            r_reward = torch.cat((ex_reward, reward))
            #r_ns_return = torch.cat((ex_ns_return, ns_return))
            r_n_step_return = torch.cat((ex_n_step_return,n_step_return))
            r_last_Q_discount = torch.cat((ex_last_Q_discount,last_Q_discount))
            r_last_state = torch.cat((ex_last_state,last_state))
            #
            r_done = torch.cat((ex_done, done))

            #indices = torch.randperm(r_state.size()[0])
            #r_state = r_state[indices]
            #r_next_state = r_next_state[indices]
            #r_action = r_action[indices]
            #r_reward = r_reward[indices]
            #r_done = r_done[indices]

            return r_state, r_next_state, r_action, r_reward, (r_n_step_return,r_last_Q_discount,r_last_state), r_done

        else:
            return self.recall()



