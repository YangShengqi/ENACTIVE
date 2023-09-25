import torch
from train.config import Config


def simple_ppo_loss(log_probs, fixed_log_probs, advantages, clip_epsilon = Config.PPO.clip_epsilon):
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages.detach()
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages.detach()
    policy_surr = -torch.min(surr1, surr2).mean()
    return policy_surr


def dual_ppo_loss(log_probs, fixed_log_probs, advantages, clip_epsilon=Config.PPO.clip_epsilon, clip_c=Config.PPO.clip_c):
    ratio = torch.exp(log_probs - fixed_log_probs)
    ratio_clamp = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    ratio_clamp_c = torch.clamp(ratio, 1.0, 1.0)
    # ratio_dual = torch.clamp(ratio, 1.0 - clip_c, 1.0)

    surr1 = ratio * advantages.detach()
    surr2 = ratio_clamp * advantages.detach()

    negative_advantages = torch.clamp(advantages, -1e6, 0)
    surr3 = ratio_clamp_c * clip_c * negative_advantages.detach()

    surr_min = torch.min(surr1, surr2)

    # paper method #
    surr_max_min = torch.max(surr_min, surr3)

    policy_surr = -surr_max_min.mean()

    return policy_surr


def simple_ppo_loss_gather(log_probs, fixed_log_probs, advantages, clip_epsilon = Config.PPO.clip_epsilon):
    ratio_0 = torch.exp(log_probs[0] - fixed_log_probs[0])
    ratio_1 = torch.exp(log_probs[1] - fixed_log_probs[1])
    ratio = torch.cat([ratio_0, ratio_1], dim=0)

    advantages_gather = torch.cat([advantages[0], advantages[1]], dim=0)

    surr1 = ratio * advantages_gather.detach()
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_gather.detach()
    policy_surr = -torch.min(surr1, surr2).mean()

    return policy_surr
