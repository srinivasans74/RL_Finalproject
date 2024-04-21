from itertools import count
from copy import deepcopy
import time

import scipy.optimize

from torch.autograd import Variable
import torch

from models.models import *
from algorithms.trpo import trpo_step
from utils.replay_memory import *
from utils.running_state import ZFilter
from utils.utils import *

from configs.config import get_config
from configs.config_mujoco import get_env_mujoco_config
import os
import time
from datetime import datetime

currentDateAndTime = datetime.now()
start_run_date_and_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
args = get_config().parse_args()
env = get_env_mujoco_config(args)

folder = f"experiments/{args.env_name}/{start_run_date_and_time}"
if not os.path.exists(folder):
    os.makedirs(folder)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]


torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
cost_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch, cost_batch, i_episode):
    costs = torch.Tensor(batch.cost)                            #-----
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))
    values_cost = cost_net(Variable(states))                    #-----

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    returns_cost = torch.Tensor(actions.size(0),1)              #-----
    deltas_cost = torch.Tensor(actions.size(0),1)               #-----
    advantages_cost = torch.Tensor(actions.size(0),1)           #-----
    ###----------------------------------------------------------------------------------------------------
    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]
    # print(f"rewards variable size: {rewards.shape} , returns variable size: {returns.shape}-----")
    ###----------------------------------------------------------------------------------------------------
    ###----------------------------------------------------------------------------------------------------
    prev_return_cost = 0
    prev_value_cost = 0
    prev_advantage_cost = 0
    for i in reversed(range(costs.size(0))):
        returns_cost[i] = costs[i] + args.gamma * prev_return_cost * masks[i]
        deltas_cost[i] = costs[i] + args.gamma * prev_value_cost * masks[i] - values_cost.data[i]
        advantages_cost[i] = deltas_cost[i] + args.gamma * args.tau * prev_advantage_cost * masks[i]

        prev_return_cost = returns_cost[i, 0]
        prev_value_cost = values_cost.data[i, 0]
        prev_advantage_cost = advantages_cost[i, 0]
    ###----------------------------------------------------------------------------------------------------
    targets = Variable(returns)
    targets_cost = Variable(returns_cost)                       #-----
    ###----------------------------------------------------------------------------------------------------   
    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())
    ###----------------------------------------------------------------------------------------------------
    ###----------------------------------------------------------------------------------------------------
    # Original code uses the same LBFGS to optimize the value loss
    def get_cost_loss(flat_params):
        set_flat_params_to(cost_net, torch.Tensor(flat_params))
        for param in cost_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        costs_ = cost_net(Variable(states))

        cost_loss = (costs_ - targets_cost).pow(2).mean()

        # weight decay
        for param in cost_net.parameters():
            cost_loss += param.pow(2).sum() * args.l2_reg
        cost_loss.backward()
        return (cost_loss.data.double().numpy(), get_flat_grad_from(cost_net).data.double().numpy())
    ###----------------------------------------------------------------------------------------------------
    
    ###----------------------------------------------------------------------------------------------------
    if cost_batch >= -args.cost_limit or i_episode <= args.exploration_iteration:
        print("ok")
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
        set_flat_params_to(value_net, torch.Tensor(flat_params))

        advantages = (advantages - advantages.mean()) / advantages.std()

        action_means, action_log_stds, action_stds = policy_net(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
            else:
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()


        def get_kl():
            mean1, log_std1, std1 = policy_net(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)
    ###----------------------------------------------------------------------------------------------------
    ###----------------------------------------------------------------------------------------------------
    else:
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_cost_loss, get_flat_params_from(cost_net).double().numpy(), maxiter=25)
        set_flat_params_to(cost_net, torch.Tensor(flat_params))

        advantages_cost = (advantages_cost - advantages_cost.mean()) / advantages_cost.std()

        action_means, action_log_stds, action_stds = policy_net(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_cost_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
            else:
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages_cost) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()


        def get_kl():
            mean1, log_std1, std1 = policy_net(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        trpo_step(policy_net, get_cost_loss, get_kl, 0.05, args.damping)
    ###----------------------------------------------------------------------------------------------------

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

EPISODE_LENGTH = 1000
EPISODE_PER_BATCH = 16
Epoch = 500
state_datas = []
velocity_datas = []
pos_datas = []
reward_datas = []
cost_datas = []
average_step_reward_datas = []
average_step_cost_datas = []
goal_vels = []
goal_vels.append(0.6)
for i_episode in count(1):
    print(f'Iteraion ={i_episode}')
    memory = Memory()
    num_steps = 0
    reward_batch = 0
    cost_step = 0                                                           #-----
    num_episodes = 0
    state = 0
    tic = time.perf_counter()
    state_data = []
    velocity_data = []
    pos_data = []
    reward_data = []
    cost_data = []
    while num_steps < EPISODE_LENGTH*EPISODE_PER_BATCH:
        state, info = env.reset()
        state = running_state(state)
        reward_sum = 0
        cost_sum = 0                                                        #-----
        for t in range(EPISODE_LENGTH):
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            
            cost = info["cost"]                                             #-----
            reward_sum += info["reward"]
            cost_sum += info["cost"]                                        #-----
            state_data.append(state)
            velocity_data.append(info["x_velocity"])
            pos_data.append(info["x_position"])
            reward_data.append(info["reward"])
            cost_data.append(info["cost"])
            next_state = running_state(next_state)
            mask = 1
            # if num_steps==1 and t==1:
            # print(f"reward: {reward} , cost: {cost} , cost_sum: {cost_sum} , reward_sum: {reward_sum}")
            if t==EPISODE_LENGTH-1:
                mask = 0
            memory.push(state, np.array([action]), mask, next_state, reward, cost)            
            if done or truncated:
                break
            state = next_state
        num_steps += EPISODE_LENGTH
        num_episodes += 1
        reward_batch += reward_sum
        cost_step += cost_sum                                               #-----
    
    state_datas.append(state_data)
    velocity_datas.append(velocity_data)
    pos_datas.append(pos_data)
    reward_datas.append(reward_data)
    cost_datas.append(cost_data)
    average_step_reward_datas.append(reward_batch / num_steps)
    average_step_cost_datas.append(cost_step / num_steps)
    
    np.save('logs/'+str(args.cost_limit)+'reward.npy',np.array(average_step_reward_datas))
    np.save('logs/'+str(args.cost_limit)+'cost.npy',np.array(average_step_cost_datas))
    
    batch = memory.sample()    
    update_params(batch, cost_step/num_steps, i_episode)
    
    np.save(folder + "/average_step_reward.npy", np.array(average_step_reward_datas))
    np.save(folder + "/average_step_cost.npy", np.array(average_step_cost_datas))

    if i_episode % args.log_interval == 0:
        print(f'Episode {i_episode}\tAverage step reward {reward_batch/num_steps:.2f}\t Average step cost {-cost_step/num_steps:.2f}')
    if i_episode >= args.exps_epoch:
        break
