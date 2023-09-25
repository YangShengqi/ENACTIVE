from environment.battlespace import BattleSpace
from train.config import Config


def get_reward(env: BattleSpace, k: int, reward_config: dict):
    # add shaped reward for training # give team reward #
    shoot_down_reward = reward_config["shoot_down_reward"]
    death_reward = reward_config["death_reward"]
    fire_reward = reward_config["fire_reward"]
    in_border_reward = reward_config["in_border_reward"]
    out_border_reward = reward_config["out_border_reward"]

    ret = 0.0
    team_kill = 0
    team_die = 0
    team_launch = 0
    team_in_border = 0
    team_out_border = 0
    for i in range(k * env.red, env.red + k * env.blue):
        aircraft_i = env.state_interface["AMS"][i]
        team_kill += aircraft_i["shoot_down_event"]["value"]
        team_die += aircraft_i["death_event"]["value"]
        team_in_border += aircraft_i["in_border_event"]["value"]
        team_out_border += aircraft_i["out_border_event"]["value"]
        for missile in aircraft_i["SMS"]:
            team_launch += missile["fire_event"]["value"]

    ret += team_kill * (team_kill + 1) / 2 * shoot_down_reward
    ret += team_die * (team_die + 1) / 2 * death_reward
    ret += fire_reward * team_launch * (team_launch + 1) / 2
    ret += team_in_border * in_border_reward  # border relative reward
    ret += team_out_border * out_border_reward
    ret_origin = ret

    # normalize ret #
    ret_min, ret_max = get_reward_range(reward_config)
    ret = (ret - ret_min) / (ret_max - ret_min)  # normalize #

    return ret, ret_origin


def get_reward_range(reward_config: dict):
    # compute max or min reward for normalization
    shoot_down_reward = reward_config["shoot_down_reward"]
    death_reward = reward_config["death_reward"]
    fire_reward = reward_config["fire_reward"]
    out_border_reward = reward_config["out_border_reward"]
    missile_num = 4

    ret_max = shoot_down_reward - fire_reward
    ret_min = death_reward + out_border_reward + fire_reward * missile_num * (missile_num + 1) / 2

    return ret_min, ret_max


def get_shape_reward(env: BattleSpace, k: int, interval, reward_config: dict):
    if k > 0:
        i = 1
        j = 0
    else:
        i = 0
        j = 1
    aircraft = env.state_interface["AMS"][i]
    msl_threat = get_ith_aircraft_missile_threat(env, i, k)

    shaped_reward = 0
    # shaped reward 1 #
    # penalty while velocity less than 300 m/s #
    current_step_tas = aircraft["TAS"]["value"]
    if current_step_tas < 300:
        shaped_reward -= (300 - current_step_tas) / 50 * (150 / Config.whole_time * interval)  # magic number 200

    # shaped reward 2 #
    # penalty while ao with nearest bandit > 60 degree if no msl threat #
    if msl_threat:
        pass
    else:
        ao_with_bandit = abs(aircraft["relative_observation"][j]["AO"]["value"]) * 57.3
        if ao_with_bandit > 60 and int(env.state_interface["AMS"][j]["alive"]["value"] + 0.1):
            shaped_reward -= 80 / Config.whole_time * interval  # magic number 100

    # shaped reward 8, encourage shooting
    aircraft_sms = env.state_interface["AMS"][i]["SMS"]
    for msl_i in range(len(aircraft_sms)):
        if int(aircraft_sms[msl_i]["state"]["value"] + 0.1) == 2 or int(aircraft_sms[msl_i]["state"]["value"] + 0.1) == 3: # msl flying #
            target_id = int(aircraft_sms[msl_i]["target_index"]["value"] + 0.1) + (1 - k) * env.red
            if int(env.state_interface["AMS"][target_id]["alive"]["value"]):
                # target alive #
                relative_obs = env.state_interface["AMS"][target_id]["relative_observation"]
                cur_target_ao = relative_obs[i]["AO"]["value"] * 57.3
                if abs(cur_target_ao) <= 80:  # target is not escaping this msl
                    pass
                else:
                    if 80 < abs(cur_target_ao) <= 90:
                        shaped_reward += (5 - msl_i) / 2.0 * (30 * 1 / 90 * interval) #30
                    elif 90 < abs(cur_target_ao) <= 135:
                        shaped_reward += (5 - msl_i) / 2.0 * (30 * 2 / 90 * interval)  # 90 mean msl avg flying time
                    elif 135 < abs(cur_target_ao) <= 180:
                        shaped_reward += (5 - msl_i) / 2.0 * (30 * 3 / 90 * interval)

                    # break for only one msl valid for this shaped reward #
                    break

    ret_min, ret_max = get_reward_range(reward_config)
    ret_shaped_reward = shaped_reward / (ret_max - ret_min)  # normalize #

    return ret_shaped_reward, shaped_reward


def get_ith_aircraft_missile_threat(env: BattleSpace, i: int, k: int):
    for j in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
        bandit_aircraft = env.state_interface["AMS"][j]
        for msl in bandit_aircraft["SMS"]:
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying #
                if int(msl["target_index"]["value"] + 0.1) == i - k * env.red:
                    return True
    return False