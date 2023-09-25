from environment.battlespace import BattleSpace
from train.config import Config
import numpy as np


def relay_guide_shaped_reward(env: BattleSpace, interval, i: int, k: int):
    shaped_reward = 0
    for aircraft_id in range(k * env.red, env.red + k * env.blue):
        if aircraft_id == i:  # current aircraft is self, not computing reward
            pass
        else:
            cur_aircraft = env.state_interface["AMS"][i]
            for msl in cur_aircraft["SMS"]:
                if int(msl["state"]["value"] + 0.1) == 2:  # only consider flying(except seeking msl)
                    cur_guide_aircraft_num = 0  # for flying msl, count num of aircrafts which are guiding this msl
                    cur_aircraft_guide = None
                    for i_guide in range(len(msl["radar_guide_info"])):
                        guide = int(msl["radar_guide_info"][i_guide]["value"] + 0.1)
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


if __name__ == "__main__":
    shaped_reward = relay_guide_shaped_reward(Config.env, 12, 0, 0)
    print(shaped_reward)
    shaped_reward = attack_same_target_shaped_reward(Config.env, 12, 0, 0, [0, 0], [1, 2])
    print(shaped_reward)

