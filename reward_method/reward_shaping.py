"""
    some shape reward
    1. velocity less than 200m/s, give punishment(-80 in total for the whole battle time respectively)
    2. when without msl_threat
           if abs(AO) with the nearest bandit aircraft > 60 degrees, give punishment
           (-100 in total for the whole battle time respectively)
"""
from environment.battlespace import BattleSpace
from reward_method import reward_method
from train.config import Config
from reward_method.reward_hyperparam_dict import origin_reward_parameters
import numpy as np


def get_ith_aircraft_missile_threat(env: BattleSpace, i: int, k: int):
    for j in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
        bandit_aircraft = env.state_interface["AMS"][j]
        for msl in bandit_aircraft["SMS"]:
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying #
                if int(msl["target_index"]["value"] + 0.1) == i - k * env.red:
                    return True
    return False


def threat_cal(env, i: int, j: int):
    i_alive = env.state_interface["AMS"][i]["alive"]["value"]
    j_alive = env.state_interface["AMS"][j]["alive"]["value"]
    r = env.state_interface["AMS"][i]["relative_observation"][j]["r"]["value"]
    r_max = env.state_interface["AMS"][0]["relative_observation"][0]["r"]["max"]

    r_threat = 100 - 100 * r / (r_max - 300000)

    if int(i_alive + 0.1) and int(j_alive + 0.1):
        threat = r_threat
    else:
        threat = 0

    return threat


def threat_sort(env, i: int, k: int):
    threat_i = []
    for j in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
        threat_i.append(threat_cal(env, i, j))
    threat_i = np.array(threat_i)
    bandit_threat_absolute_idx = (np.argsort(-threat_i) + (1 - k) * env.red).tolist()

    return bandit_threat_absolute_idx


def get_center_of_kteam_aircraft(env, i: int, k: int, center_consider_self: bool):
    # did not consider altitude
    position_x = 0
    position_y = 0
    aircraft_num = 0
    for aircraft_i in range(k * env.red, env.red + k * env.blue):
        aircraft = env.state_interface["AMS"][aircraft_i]
        if int(aircraft["alive"]["value"] + 0.1) and (center_consider_self or aircraft_i + k * env.red != i):
            position_x += aircraft["Xg_0"]["value"]
            position_y += aircraft["Xg_1"]["value"]
            aircraft_num += 1

    if aircraft_num == 0:
        center_x = np.nan
        center_y = np.nan
    else:
        center_x = position_x / aircraft_num
        center_y = position_y / aircraft_num

    return center_x, center_y


def threat_sort_with_kteam_center(env, k: int, center_x, center_y):
    distance_array = []
    for i in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
        aircraft = env.state_interface["AMS"][i]
        if int(aircraft["alive"]["value"] + 0.1):
            distance_array.append(np.sqrt((center_x - aircraft["Xg_0"]["value"]) ** 2 + (center_y - aircraft["Xg_1"]["value"]) ** 2))
        else:
            distance_array.append(1e8)
    # if all die, judge in outer iteration #
    nearest_target_from_center = np.argsort(distance_array) + (1 - k) * env.red
    return nearest_target_from_center


def get_angle_between_velocity_and_attack_direction(env, k: int):
    team_center_x, team_center_y = get_center_of_kteam_aircraft(env, 0, k, center_consider_self=True)
    # if team_center_x is np.nan # team center will not be np.nan while center_consider_self=True
    nearest_target_from_center = threat_sort_with_kteam_center(env, k, team_center_x, team_center_y)
    aircraft_target = env.state_interface["AMS"][nearest_target_from_center[0]]
    target_x = aircraft_target["Xg_0"]["value"]
    target_y = aircraft_target["Xg_1"]["value"]
    attack_direction = [target_x - team_center_x, target_y - team_center_y]  # main attack direction for team
    angle_array = []
    for aircraft_i in range(k * env.red, env.red + k * env.blue):
        cur_aircraft = env.state_interface["AMS"][aircraft_i]
        if int(cur_aircraft["alive"]["value"] + 0.1):  # this aircraft alive
            velocity = [cur_aircraft["Vg_0"]["value"], cur_aircraft["Vg_1"]["value"]]
            angle_i = np.arccos((attack_direction[0] * velocity[0] + attack_direction[1] * velocity[1]) /
                                ((np.sqrt(attack_direction[0] * attack_direction[0] + attack_direction[1] * attack_direction[1])) * np.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])))
            angle_array.append(angle_i)
        else:
            angle_array.append(np.nan)
    return angle_array


def relay_guide_shaped_reward(env: BattleSpace, interval, i: int, k: int):
    shaped_reward = 0
    for aircraft_id in range(k * env.red, env.red + k * env.blue):
        if aircraft_id == i:  # current aircraft is self, not computing reward
            pass
        else:
            cur_aircraft = env.state_interface["AMS"][aircraft_id]
            for msl in cur_aircraft["SMS"]:
                if int(msl["state"]["value"] + 0.1) == 2:  # only consider flying(except seeking msl)
                    cur_guide_aircraft_num = 0  # for flying msl, count num of aircrafts which are guiding this msl
                    cur_aircraft_guide = None
                    for i_guide in range(len(msl["RadarGuide"])):
                        guide = int(msl["RadarGuide"][i_guide]["radar_guide_info"]["value"] + 0.1)
                        cur_guide_aircraft_num += guide
                        if i_guide == (aircraft_id - k * env.red):  # cur guide info of aircraft which launch this msl
                            msl_launch_aircraft_guide = guide
                        elif i_guide == (i - k * env.red):
                            cur_aircraft_guide = guide
                    if cur_aircraft_guide == 1:  # cur aircraft is guiding this flying msl
                        shaped_reward += cur_aircraft_guide / cur_guide_aircraft_num * 100 / Config.whole_time * interval
                        # magic number 100
    return shaped_reward


def attack_same_target_shaped_reward(env: BattleSpace, interval, i: int, k: int, target_list, hors_list):
    # clear np.nan
    shaped_reward = 0
    while True:
        if np.nan in target_list:
            target_list.remove(np.nan)
        else:
            break
    # while True:
    #     if np.nan in hors_list:
    #         hors_list.remove(np.nan)
    #     else:
    #         break

    # print(target_list)
    team_target_same = False
    if len(target_list) >= 2:
        team_target_same = True
        for i in range(len(target_list)):
            if target_list[i] != target_list[0]:
                team_target_same = False
                break
    all_attack_same_target = True
    if team_target_same:
        for i_self in range(k * env.red, env.red + k * env.blue):  # self team
            cur_aircraft = env.state_interface["AMS"][i_self]
            if int(cur_aircraft["alive"]["value"] + 0.1) == 0:
                continue
            cur_ao = cur_aircraft["relative_observation"][target_list[0]]["AO"]["value"]
            hors_decision = int(hors_list[i_self - k * env.red] + 0.1)
            """
                0: maintain heading
                1: heading towards target
                2: crank 30 deg
                3: crank 50 deg
                4: turn 90 deg to target
                5: turn 180 to target
            """
            if (cur_ao < np.pi / 180 * 35 and hors_decision == 0) or int(1 <= hors_decision <= 2):
                pass
            else:
                all_attack_same_target = False

    if all_attack_same_target:
        shaped_reward += 100 / Config.whole_time * interval  # magic number 100
    return shaped_reward


def get_ith_aircraft_shaped_reward(env: BattleSpace, i: int, k: int, interval, self_hor=None, last_hor=None, self_ver=None, last_ver=None, self_target=None, last_target=None,
                                   rewards_hyperparam_dict=origin_reward_parameters):

    shaped_reward = 0
    aircraft = env.state_interface["AMS"][i]
    msl_threat = get_ith_aircraft_missile_threat(env, i, k)

    # shaped reward 1 #
    # # penalty while velocity less than 200 m/s #
    # current_step_tas = aircraft["TAS"]["value"]
    # if current_step_tas < 200:
    #     shaped_reward -= 80 / Config.whole_time * interval  # magic number 80
    current_step_tas = aircraft["TAS"]["value"]
    if current_step_tas < 300:
        shaped_reward -= (300 - current_step_tas) / 50 * 200 / Config.whole_time * interval  # magic number 80

    # shaped reward 2 #
    # penalty while ao with nearest bandit > 60 degree if no msl threat #
    if msl_threat:
        pass
    else:
        bandit_threat_absolute_idx = threat_sort(env, i, k)
        nearest_bandit = bandit_threat_absolute_idx[0]
        ao_with_nearest_bandit = abs(aircraft["relative_observation"][nearest_bandit]["AO"]["value"]) * 57.3
        if ao_with_nearest_bandit > 60 and int(env.state_interface["AMS"][nearest_bandit]["alive"]["value"] + 0.1):
            shaped_reward -= 100 / Config.whole_time * interval  # magic number 100

    # shaped reward 3 #
    # prevent sleepwalk
    aircraft_self = env.state_interface["AMS"][i]
    self_x = aircraft_self["Xg_0"]["value"]
    self_y = aircraft_self["Xg_1"]["value"]
    nearest_teammate_distance = 1e8
    nearest_bandit_distance = 1e8
    if msl_threat:
        pass
    else:
        for aircraft_i in range(k * env.red, env.red + k * env.blue):
            if aircraft_i == i:  # self
                pass
            else:
                aircraft = env.state_interface["AMS"][aircraft_i]
                if int(aircraft["alive"]["value"]):
                    cur_x = aircraft["Xg_0"]["value"]
                    cur_y = aircraft["Xg_1"]["value"]
                    r = np.sqrt((self_x - cur_x) ** 2 + (self_y - cur_y) ** 2)
                    if r < nearest_teammate_distance:
                        nearest_teammate_distance = r

        for aircraft_i in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
            aircraft = env.state_interface["AMS"][aircraft_i]
            if int(aircraft["alive"]["value"]):
                cur_x = aircraft["Xg_0"]["value"]
                cur_y = aircraft["Xg_1"]["value"]
                r = np.sqrt((self_x - cur_x) ** 2 + (self_y - cur_y) ** 2)
                if r < nearest_bandit_distance:
                    nearest_bandit_distance = r

        # if (50000 <= nearest_teammate_distance < 1e8 - 1) and (70000 <= nearest_bandit_distance < 1e8 - 1):
        #     nearest_teammate_distance = nearest_teammate_distance if nearest_teammate_distance < 100000 else 100000
        #     shaped_reward -= (nearest_teammate_distance - 50000) / 50000 * 100 / Config.whole_time * interval  # todo change to -= #
        if 100000 < nearest_bandit_distance < 1e8 - 1:
            nearest_bandit_distance = 100000

        if 40000 <= nearest_teammate_distance < 60000:
            shaped_reward -= (nearest_teammate_distance - 40000) / 20000 * 100 / Config.whole_time * interval
        elif 60000 <= nearest_teammate_distance < 80000:
            shaped_reward -= ((nearest_teammate_distance - 60000) / 20000 * 200 / Config.whole_time * interval + 100 / Config.whole_time * interval)
        elif 80000 <= nearest_bandit_distance <= 100000:
            shaped_reward -= ((nearest_teammate_distance - 80000) / 20000 * 300 / Config.whole_time * interval + 300 / Config.whole_time * interval)

    # # shaped reward 4 #
    # # bonus while team aircraft choose same target and AO with this target less than 40 degree
    # # this reward base on decision(output), target_list: [self_target_num]
    # while True:  # clear np.nan
    #     if np.nan in target_list:
    #         target_list.remove(np.nan)
    #     else:
    #         break
    # # print(target_list)
    # team_target_same = False
    # if len(target_list) >= 2:
    #     team_target_same = True
    #     for aircraft_i in range(len(target_list)):
    #         if target_list[aircraft_i] != target_list[0]:
    #             team_target_same = False
    #             break
    # if team_target_same:
    #     target_in_all_team_aircraft_ao_range = True
    #     cur_bandit_aircraft = env.state_interface["AMS"][target_list[0]]
    #     if int(cur_bandit_aircraft["alive"]["value"] + 0.1):  # this target alive
    #         for i_self in range(k * env.red, env.red + k * env.blue):  # self team
    #             cur_aircraft = env.state_interface["AMS"][i_self]
    #             if int(cur_aircraft["alive"]["value"] + 0.1) == 0:
    #                 continue
    #             cur_ao = cur_aircraft["relative_observation"][target_list[0]]["AO"]["value"]
    #             if cur_ao < np.pi / 180 * 40:
    #                 continue
    #             else:
    #                 target_in_all_team_aircraft_ao_range = False
    #                 break
    #     if target_in_all_team_aircraft_ao_range:
    #         shaped_reward += 100 / Config.whole_time * interval  # magic number 100
    #
    # # shaped reward 5 #
    # # bonus while aircraft guide teammates' msl
    # shaped_reward += relay_guide_shaped_reward(env, interval, i, k)
    #
    # # shaped reward 6 #
    # # bonus while choose to attack same target
    # shaped_reward += attack_same_target_shaped_reward(env, interval, i, k, target_list, hors_list)

    # shaped reward 7, h_best in [4000, 5000]
    if msl_threat:
        pass
    else:
        aircraft_h = - env.state_interface["AMS"][i]["Xg_2"]["value"]
        if aircraft_h >= 5000:
            dis_from_best = aircraft_h - 5000
            if dis_from_best >= 5000:
                dis_from_best = 5000
            shaped_reward -= dis_from_best / 5000 * 200 / Config.whole_time * interval
        if aircraft_h < 5000:
            h = aircraft_h
            shaped_reward += aircraft_h / 5000 * 200 / Config.whole_time * interval

    # shaped reward 8, encourage shooting
    aam_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"] + 0.1)
    aircraft_sms = env.state_interface["AMS"][i]["SMS"]
    for msl_i in range(len(aircraft_sms)):
        if int(aircraft_sms[msl_i]["state"]["value"] + 0.1) == 2 or int(aircraft_sms[msl_i]["state"]["value"] + 0.1) == 3:
            # msl flying #
            target_id = int(aircraft_sms[msl_i]["target_index"]["value"] + 0.1) + (1 - k) * env.red
            if int(env.state_interface["AMS"][target_id]["alive"]["value"]):
                # target alive #
                relative_obs = env.state_interface["AMS"][target_id]["relative_observation"]
                cur_target_ao = relative_obs[i]["AO"]["value"] * 180 / np.pi
                if np.abs(cur_target_ao) <= 80:  # target is not escaping this msl
                    pass
                else:
                    if 80 < np.abs(cur_target_ao) <= 90:
                        shaped_reward += (5 - msl_i) / 2.0 * interval * 1 / 90 * 30
                    elif 90 < np.abs(cur_target_ao) <= 135:
                        shaped_reward += (5 - msl_i) / 2.0 * interval * 2 / 90 * 30  # 90 mean msl avg flying time
                    elif 135 < np.abs(cur_target_ao) <= 180:
                        shaped_reward += (5 - msl_i) / 2.0 * interval * 3 / 90 * 30

                    # break for only one msl valid for this shaped reward #
                    break

    # add shaking rewards #
    if last_hor is None:
        pass
    else:
        if last_hor[i - k * env.red] is np.nan or self_hor[i - k * env.red] is np.nan:
            # no penalty for first step and dead agent
            pass
        else:
            if self_hor[i - k * env.red] != last_hor[i - k * env.red]:
                shaped_reward -= 100 / Config.whole_time * interval
        if last_ver[i - k * env.red] is np.nan or self_ver[i - k * env.red] is np.nan:
            # no penalty for first step and dead agent
            pass
        else:
            if last_ver[i - k * env.red] != self_ver[i - k * env.red]:
                shaped_reward -= 100 / Config.whole_time * interval
        if last_target[i - k * env.red] is np.nan or self_target[i - k * env.red] is np.nan:
            # no penalty for first step and dead agent
            pass
        else:
            if last_target[i - k * env.red] != self_target[i - k * env.red]:
                shaped_reward -= 100 / Config.whole_time * interval

    # add Radar Locked reward for pomdp
    if msl_threat:
        pass
    else:
        for ith in range(env.red if k else env.blue):
            shaped_reward += 200 * env.state_interface["AMS"][i]["RadarModel"][ith]["FCR_locked"]["value"] / Config.whole_time * interval
            # print(200 * env.state_interface["AMS"][i]["RadarModel"][ith]["FCR_locked"]["value"] / Config.whole_time * interval)

    ret_min, ret_max = reward_method.get_solo_reward_range(env, i, rewards_hyperparam_dict)
    ret_shaped_reward = shaped_reward / (ret_max - ret_min)  # normalize #

    return ret_shaped_reward


if __name__ == "__main__":
    env = BattleSpace()
    env.random_init()
    env.reset()

    a = threat_sort(env, 2, 1)
    b = get_ith_aircraft_missile_threat(env, 2, 1)
    # c = get_ith_aircraft_shaped_reward(env, 0, 0, 400, 12)


