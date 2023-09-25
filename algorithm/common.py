import torch
import sys
import os
import numpy as np
from utils import to_device
from train.config import Config


def estimate_advantages(rewards, masks, values, device, gamma=Config.gamma, tau=Config.tau):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0

    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def estimate_advantages_tdn(rewards, masks, values, device, gamma=Config.gamma, tau=Config.tau):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)

    accu_rewards = torch.zeros(rewards.size(0), 1)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)  # advantage will detach while training

    td_step = Config.TD_step
    # prev_value = 0
    prev_advantage = 0

    for i in reversed(range(rewards.size(0))):
        to_end = False
        for j in range(td_step + 1):  # add accumulate reward
            accu_rewards[i] += gamma ** j * rewards[i + j]
            if masks[i + j] == 0:  # to end
                to_end = True
                break

        if to_end:
            deltas[i] = accu_rewards[i] - values[i]
        else:
            deltas[i] = accu_rewards[i] + gamma ** (j + 1) * values[i + j + 1] - values[i]

        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def estimate_advantages_gae_independ(rewards, masks, mask_solo_done, values, device, gamma=Config.gamma, lam=Config.lam):
    rewards, masks, mask_solo_done, values = to_device(torch.device('cpu'), rewards, masks, mask_solo_done, values)
    done = masks * mask_solo_done
    values = values.squeeze(2)
    tensor_type = type(rewards)

    deltas = tensor_type(rewards.size(0), 2)
    gae_advantages = tensor_type(rewards.size(0), 2)
    prev_value = torch.tensor(np.array([0, 0]))

    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * done[i] - values[i]
        prev_value = values[i]

    prev_gae_advantages = torch.tensor(np.array([0, 0]))
    for i in reversed(range(deltas.size(0))):
        gae_advantages[i] = deltas[i] + gamma * lam * prev_gae_advantages * done[i]
        prev_gae_advantages = gae_advantages[i]

    returns = values + gae_advantages
    gae_advantages = (gae_advantages - gae_advantages.mean()) / gae_advantages.std()
    gae_advantages, returns = to_device(device, gae_advantages, returns)
    return gae_advantages, returns


def estimate_advantages_gae(rewards, masks, values, device, gamma=Config.gamma, lam=Config.lam):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)

    deltas = tensor_type(rewards.size(0), 1)
    gae_advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        prev_value = values[i, 0]

    prev_gae_advantages = 0
    for i in reversed(range(deltas.size(0))):
        gae_advantages[i] = deltas[i] + gamma * lam * prev_gae_advantages * masks[i]
        prev_gae_advantages = gae_advantages[i, 0]

    returns = values + gae_advantages
    gae_advantages = (gae_advantages - gae_advantages.mean()) / gae_advantages.std()
    gae_advantages, returns = to_device(device, gae_advantages, returns)
    return gae_advantages, returns

# def estimate_LTR_advantages(rewards, masks, values, steps, device,gamma = Config.gamma, tau  = Config.tau):
#     rewards, masks, values, steps = to_device(torch.device('cpu'), rewards, masks, values, steps)
#     tensor_type = type(rewards)
#     deltas = tensor_type(rewards.size(0), 1)
#     advantages = tensor_type(rewards.size(0), 1)
#
#     prev_value = 0
#     prev_advantage = 0
#
#     # mask[i][1] for game mask, mask[i][0] for agent id #
#     for i in reversed(range(rewards.size(0))):
#         cur_step = steps[i].tolist()
#         # print(masks[i].tolist()[0], type(masks[i].tolist()[0]))
#         cur_agent_step = cur_step[int(masks[i].tolist()[0])]
#         power = 1 if np.isnan(cur_agent_step) else cur_agent_step
#
#         deltas[i] = rewards[i] + (gamma ** power) * prev_value * masks[i][1] - values[i]
#         advantages[i] = deltas[i] + (gamma ** power) * tau * prev_advantage * masks[i][1]
#         # print("origin mask ", masks[i][1])
#
#         prev_value = values[i, 0]
#         prev_advantage = advantages[i, 0]
#
#     returns = values + advantages
#     advantages = (advantages - advantages.mean()) / advantages.std()
#
#     advantages, returns = to_device(device, advantages, returns)
#     return advantages, returns


def estimate_LTR_advantages(rewards, masks, values, steps, device, gamma=Config.gamma, tau=Config.tau):
    rewards, masks, values, steps = to_device(torch.device('cpu'), rewards, masks, values, steps)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0

    for i in reversed(range(rewards.size(0))):
        # cur_aircraft = masks[i].tolist()[0]

        cur_step = steps[i].tolist()
        # print(masks[i].tolist()[0], type(masks[i].tolist()[0]))
        cur_agent_step = cur_step[int(masks[i].tolist()[0])]
        power = 0 if np.isnan(cur_agent_step) else cur_agent_step

        deltas[i] = rewards[i] + (gamma ** (power + 1)) * prev_value * masks[i][1] - values[i]
        advantages[i] = deltas[i] + (gamma ** (power + 1)) * tau * prev_advantage * masks[i][1]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def estimate_LTR_advantange_single(rewards, masks, values, steps, device, gamma=Config.gamma, tau=Config.tau):
    rewards, masks, values, steps = to_device(torch.device('cpu'), rewards, masks, values, steps)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0

    for i in reversed(range(rewards.size(0))):
        # cur_aircraft = masks[i].tolist()[0]

        cur_step = steps[i].tolist()
        # # print(masks[i].tolist()[0], type(masks[i].tolist()[0]))
        # cur_agent_step = cur_step[int(masks[i].tolist()[0])]
        power = 0 if np.isnan(cur_step) else cur_step

        deltas[i] = rewards[i] + (gamma ** (power + 1)) * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + (gamma ** (power + 1)) * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


# origin loss step method, without momentum record, change in 2021/01/20 #
def loss_step_origin(model, loss):
    optimizer_multihead_nets = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_multihead_nets.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer_multihead_nets.step()


def loss_step(model, model_optimizer, loss):
    # optimizer_multihead_nets = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer_multihead_nets.zero_grad()
    model_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    model_optimizer.step()
