import torch
from logger import Logger
from train.args_init import args

# Set the logger
logger = Logger('./logs') # dive in later
step=0

log_protect = 1e-10
multinomial_protect = 1e-10

def to_np(x): # from tensor to numpy
    return x.data.cpu().numpy()

def to_var(x): # from tensor to Variable
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def ppo_step(env, multihead_net, steer_policy_net,
             optimizer_multihead_nets, optimizer_steer_policy,
             states, steer_states, maneuvers, steers, shots, targets,
             returns, advantages, fixed_log_probs, steer_fixed_log_probs, clip_epsilon, l2_reg,
             log, i_iter, invoke_times_per_iter, agent_id):

    # multi head nn prediction
    # multihead_net.to(torch.device('cuda', index=0))
    # states.to(torch.device('cuda', index=0))
    # maneuvers.to(torch.device('cuda', index=0))
    # shots.to(torch.device('cuda', index=0))
    # targets.to(torch.device('cuda', index=0))
    log_probs, values_pred = multihead_net.get_log_prob_and_values( states, maneuvers, shots, targets )

    # calculate value loss
    value_loss = (values_pred - returns).pow(2).mean()
    # weight decay
    for param in multihead_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg

    # calculate policy loss
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages.detach()
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages.detach()
    policy_surr = -torch.min(surr1, surr2).mean()

    # calculate policy entropy loss
    maneuver_prob, shoot_prob, target_prob, _ = multihead_net(states)
    maneuver_entropy_loss = - torch.mean((maneuver_prob + log_protect) * torch.log(maneuver_prob + log_protect))
    target_entropy_loss = - torch.mean((target_prob + log_protect) * torch.log(target_prob + log_protect))
    shoot_entropy_loss = - torch.mean((shoot_prob + log_protect) * torch.log(shoot_prob + log_protect))

    # calculate combined loss
    if args.entropy_loss:  # use entropy loss
        loss = policy_surr + 5e-6 * value_loss - 1e-4 * maneuver_entropy_loss - 1e-4 * target_entropy_loss - 1e-4 * shoot_entropy_loss
    else:
        loss = policy_surr + 5e-6 * value_loss

    optimizer_multihead_nets.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(multihead_net.parameters(), 1)
    optimizer_multihead_nets.step()

    # update steer policy
    if "HybridManeuver" in env.action_interface["AMS"][0].keys():
        steer_log_probs = steer_policy_net.get_log_prob(steer_states, steers)
        steer_ratio = torch.exp(steer_log_probs - steer_fixed_log_probs)
        steer_surr1 = steer_ratio * advantages
        steer_surr2 = torch.clamp(steer_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        steer_policy_surr = -torch.min(steer_surr1, steer_surr2).mean()
        optimizer_steer_policy.zero_grad()
        steer_policy_surr.backward()
        torch.nn.utils.clip_grad_norm_(steer_policy_net.parameters(), 1)
        optimizer_steer_policy.step()
    else:
        steer_policy_surr = torch.zeros(1)

    # tensorboard logging
    if agent_id == 0:  # log only red[0] agent
        global step
        if invoke_times_per_iter % 10 == 0:
            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'loss': loss.item(),  # scalar
                'value_loss': value_loss.item(),  # scalar
                'maneuver_entropy': maneuver_entropy_loss.item(),
                'target_entropy': target_entropy_loss.item(),
                'shoot_entropy': shoot_entropy_loss.item(),
                'maneuver_policy_surr': policy_surr.item(),  # scalar
                'steer_policy_surr': steer_policy_surr.item(),  # scalar
                'Iteration': i_iter,
                'Episodes_Sampled': log['num_episodes'][agent_id],
                'Min_Reward': log['min_reward'][agent_id],
                'Max_Reward': log['max_reward'][agent_id],
                'Avg_Reward': log['avg_reward'][agent_id],
                'Avg_Positive_Reward': log['avg_positive_reward'][agent_id]
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step + 1)

            for tag, value in multihead_net.named_parameters():
                tag = tag.replace('.', '/multihead_policy/')
                logger.histo_summary(tag, to_np(value), step + 1)  # from Parameter to np.array
                logger.histo_summary(tag + '/grad', to_np(value.grad), step + 1)  # from Variable to np.array

            if "HybridManeuver" in env.action_interface["AMS"][0].keys():
                for tag, value in steer_policy_net.named_parameters():
                    tag = tag.replace('.', '/steer_policy/')
                    logger.histo_summary(tag, to_np(value), step + 1)  # from Parameter to np.array
                    logger.histo_summary(tag + '/grad', to_np(value.grad), step + 1)  # from Variable to np.array

        step += 1