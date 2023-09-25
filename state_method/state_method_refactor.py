# refactor verion on 2020/08/28, add some state and all not available aircraft related state set to 0 #
from environment.battlespace import BattleSpace
from train.config import Config
import numpy as np
import copy


def get_aircraft_available(env, i):
    aircraft_i = env.state_interface["AMS"][i]
    flying_missile_num = 0
    for m in aircraft_i["SMS"]:
        if abs(m["state"]["value"] - 2) < 0.1 or abs(m["state"]["value"] - 3) < 0.1:
            flying_missile_num = flying_missile_num + 1
    if env.state_interface["AMS"][i]["alive"]["value"] < 0.1 and flying_missile_num is 0:
        return False
    return True


def index_to_one_hot(i, total):
    ret = np.zeros(total)
    ret[i] = 1
    return ret


def get_ith_aircraft_self_state(env: BattleSpace, i: int, omit_token_state_info=False, omit_msl_token_state=True):
    ret = []
    aircraft_i = env.state_interface["AMS"][i]
    ret.append(env.normalize(aircraft_i["AAM_remain"]))
    # ret.append(env.normalize(aircraft_i["FCR_locked"]))
    ret.append(env.normalize(aircraft_i["RWR_nailed"]))
    ret.append(env.normalize(aircraft_i["RWR_spiked"]))
    ret.append(env.normalize(aircraft_i["TAS"]))
    ret.append(env.normalize(aircraft_i["h_dot"]))
    ret.append(env.normalize(aircraft_i["residual_chi"]))
    ret.append(env.normalize(aircraft_i["Xg_2"]))

    if omit_token_state_info:
        pass
    else:
        for j in aircraft_i["FCSModel"]:
            ret.append(env.normalize(j["FCS_available"]))
        for j in aircraft_i["RWRModel"]:
            ret.append(env.normalize(j["RWR_fetched"]))
        for j in aircraft_i["RadarModel"]:
            ret.append(env.normalize(j["FCR_locked"]))

        if omit_msl_token_state:
            pass
        else:
            for m in aircraft_i["SMS"]:
                for j in m["FCSGuide"]:
                    ret.append(env.normalize(j["fcs_guide_info"]))
            for m in aircraft_i["SMS"]:
                for j in m["RadarGuide"]:
                    ret.append(env.normalize(j["radar_guide_info"]))
            for m in aircraft_i["SMS"]:
                ret.append(env.normalize(m["state"]))
                if int(m["state"]["value"] + 0.1) == 2 or int(m["state"]["value"] + 0.1) == 3:  # missile flying
                    ret.append(env.normalize(m["target_index"]))
                    ret.append(1.0 - env.normalize(m["TGO"]))
                    ret.append(env.normalize(m["r_dot_m"]))
                    ret.append(env.normalize(m["TAS_m"]))
                    ret.append(env.normalize(m["TA_m"]))
                    ret.append(env.normalize(m["AO_m"]))
                else:
                    ret.append(0.0)
                    ret.append(1.0)
                    ret.append(0.0)
                    ret.append(0.0)
                    ret.append(0.0)
                    ret.append(0.0)

    # add body angle
    ret.append(env.normalize(aircraft_i["attg_0"]))
    ret.append(env.normalize(aircraft_i["attg_1"]))
    # add overload
    ret.append(env.normalize(aircraft_i["n_y"]))
    # add out boarder time
    ret.append(env.normalize(aircraft_i["out_of_border_time"]))
    ret.append(env.normalize(aircraft_i["out_of_border_distance"]))
    # add alive
    ret.append(env.normalize(aircraft_i["alive"]))
    # add mask
    # not add maneuver mask, all time 1 #
    # add shoot and target mask # todo
    # ret.append(env.action_interface(aircraft_i))

    return ret


def get_aircraft_relative_obs(env: BattleSpace, i: int, j: int, position_encoding=True):  # this may use for token #
    # relative obs of i to j #
    if i == j:
        print("no relative obs between same aircraft")
        exit(0)
    ret = []
    temp = []
    cur_relative_available = get_aircraft_available(env, i) and get_aircraft_available(env, j)
    env.to_list(temp, env.state_interface["AMS"][i]["relative_observation"][j])
    for k in temp:
        if cur_relative_available:
            ret.append(env.normalize(k))
        else:
            ret.append(0.0)
    if position_encoding:
        aircraft_num = len(env.state_interface["AMS"])
        ret_i = index_to_one_hot(i, aircraft_num)
        ret_j = index_to_one_hot(j, aircraft_num)
        for item in ret_i:
            ret.append(item)
        for item in ret_j:
            ret.append(item)

    return ret


def get_ith_aircraft_dlz(env: BattleSpace, i: int, j: int):
    if (i - env.red + 0.1) * (j - env.red + 0.1) > 0:
        print("no dlz info between teammates")
        exit(0)
    aircraft_i = env.state_interface["AMS"][i]
    aircraft_j = env.state_interface["AMS"][j]
    ret = []
    if aircraft_i["alive"] and aircraft_j["alive"]:  # only alive add dlz info, not available #
        dlz_list = aircraft_i["attack_zone_list"]
        if i < env.red:
            enemy_id = j - env.red
        else:
            enemy_id = j
        dlz = dlz_list[enemy_id]
        ret.append(env.normalize(dlz["Raero"]))
        ret.append(env.normalize(dlz["Rmax"]))
        ret.append(env.normalize(dlz["Rmin"]))
        ret.append(env.normalize(dlz["Ropt"]))
        ret.append(env.normalize(dlz["Rpi"]))
        ret.append(env.normalize(dlz["Rtr"]))
    else:
        for _ in range(6):
            ret.append(0.0)

    return ret


def get_token_relative_state(env, i: int, j: int, position_encoding=True):
    # relative of i to j #
    # input i for self aircraft id, j for opponent id, considering 2 vs 2 #
    i_friend = (env.red - i - 1) if i < env.red else (2 * env.red + env.blue - i - 1)
    relative_i = i - env.red if i >= env.red else i
    relative_j = j - env.red if j >= env.red else j

    ret_token = []
    # step 1: add alive
    ret_token.append(env.normalize(env.state_interface["AMS"][i]["alive"]))
    ret_token.append(env.normalize(env.state_interface["AMS"][j]["alive"]))
    # ret_token.append(env.normalize(env.state_interface["AMS"][i_friend]["alive"]))  # i_friend may equals j

    # step 2: add if teammate
    if (i - env.red + 0.1) * (j - env.red + 0.1) > 0:  # teammate
        teammate = True
    else:
        teammate = False

    if teammate:
        ret_token.append(0.0)  # teammate
    else:
        ret_token.append(1.0)  # enemy

    # step 3: add relative obs
    temp = []
    env.to_list(temp, env.state_interface["AMS"][i]["relative_observation"][j])
    for k in temp:
        ret_token.append(env.normalize(k) if get_aircraft_available(env, i) and get_aircraft_available(env, j) else 0.0)

    # step 4: add sensor state(FCR and RWR)
    if teammate:
        ret_token.append(0.0)  # FCR
        ret_token.append(0.0)  # RWR
        ret_token.append(0.0)  # FCS
    else:
        ret_token.append(
            (1 - env.normalize(env.state_interface["AMS"][i]["RadarModel"][relative_j]["FCR_locked"])) * (-0.4) + 1)
        # add RWR state for cur target #
        ret_token.append(
            (1 - env.normalize(env.state_interface["AMS"][i]["RWRModel"][relative_j]["RWR_fetched"])) * (-0.4) + 1)
        ret_token.append(
            (1 - env.normalize(env.state_interface["AMS"][i]["FCSModel"][relative_j]["FCS_available"])) * (-0.4) + 1)

    # step 5: add tgo (this info in msl token)
    # step 6: add DLZ
    if teammate:  # for teammate, no valid DLZ info
        for _ in range(len(env.state_interface["AMS"][i]["attack_zone_list"][0]) - 1 - 3):  # - 1 for ASE circle, -3 for some dlz info #
            ret_token.append(0.0)
    else:
        DLZ_list = env.state_interface["AMS"][i]["attack_zone_list"][relative_j]
        # ret_token.append(env.normalize(DLZ_list["Raero"]))
        ret_token.append(env.normalize(DLZ_list["Rmax"]))
        # ret_token.append(env.normalize(DLZ_list["Rmin"]))
        # ret_token.append(env.normalize(DLZ_list["Ropt"]))
        ret_token.append(env.normalize(DLZ_list["Rpi"]))
        ret_token.append(env.normalize(DLZ_list["Rtr"]))
    # add positional encoding #
    if position_encoding:
        aircraft_num = len(env.state_interface["AMS"])
        ret_i = index_to_one_hot(i, aircraft_num)
        ret_j = index_to_one_hot(j, aircraft_num)
        for item in ret_i:
            ret_token.append(item)
        for item in ret_j:
            ret_token.append(item)

    return ret_token


def get_kteam_token_relative_state(env, k: int):
    relative_state_token = []
    for self_id in range(k * env.red, env.red + k * env.blue):
        if k == 0:
            teammate_id = env.red - 1 - self_id
        else:
            teammate_id = 2 * env.red + env.blue - 1 - self_id
        relative_state_token.append(get_token_relative_state(env, self_id, teammate_id, position_encoding=True))
        for enemy_id in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
            relative_state_token.append(get_token_relative_state(env, self_id, enemy_id, position_encoding=True))

    return relative_state_token


def get_kteam_global_ground_truth_state(env: BattleSpace, k: int, all_aircraft_info=False, omit_token_state_info=False, omit_msl_token_state=False):
    ret_global = []
    if all_aircraft_info:
        for i in range(env.red + env.blue):
            # add global state #
            aircraft_id = (i + env.red * k) % (env.red + env.blue)
            cur_state = get_ith_aircraft_self_state(env, aircraft_id,
                                                    omit_token_state_info=omit_token_state_info,
                                                    omit_msl_token_state=omit_msl_token_state)
            cur_aircraft_available = get_aircraft_available(env, aircraft_id)
            for item in cur_state:
                if cur_aircraft_available:
                    ret_global.append(item)
                else:
                    ret_global.append(0)
            # add DLZ state #
            if aircraft_id < env.red:
                for j in range(env.blue):
                    target_id = j + env.red
                    cur_dlz = get_ith_aircraft_dlz(env, aircraft_id, target_id)  # dlz part have alive judgment inside
                    for item in cur_dlz:
                        ret_global.append(item)
    else:
        for enemy_aircraft_id in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
            cur_state = get_ith_aircraft_self_state(env, enemy_aircraft_id,
                                                    omit_token_state_info=omit_token_state_info,
                                                    omit_msl_token_state=omit_msl_token_state)
            cur_aircraft_available = get_aircraft_available(env, enemy_aircraft_id)
            for item in cur_state:
                if cur_aircraft_available:
                    ret_global.append(item)
                else:
                    ret_global.append(0.0)
            # add DLZ state #
            for aircraft_id in range(k * env.red, env.red + k * env.blue):
                cur_dlz = get_ith_aircraft_dlz(env, enemy_aircraft_id, aircraft_id)
                for item in cur_dlz:
                    ret_global.append(item)  # dlz part have alive judgment inside

    # add relative observation between enemys # this info is not in attention net input state #
    for enemy_aircraft_id_0 in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
        for enemy_aircraft_id_1 in range(enemy_aircraft_id_0 + 1, env.red + (1 - k) * env.blue):
            cur_relative_state = get_aircraft_relative_obs(env, enemy_aircraft_id_0, enemy_aircraft_id_1,
                                                           position_encoding=False)  # not token part, position not necessary #
            for item in cur_relative_state:
                ret_global.append(item)

    return ret_global


def get_kteam_native_state(env: BattleSpace, k: int, omit_token_state_info=False):
    ret_native = []
    for i in range(k * env.red, env.red + k * env.blue):
        cur_state = get_ith_aircraft_self_state(env, i, omit_token_state_info=omit_token_state_info, omit_msl_token_state=True)
        cur_aircraft_available = get_aircraft_available(env, i)
        for item in cur_state:
            if cur_aircraft_available:
                ret_native.append(item)
            else:
                ret_native.append(0.0)
        # add DLZ state #
        for enemy_aircraft_id in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
            # print(i, enemy_aircraft_id)
            cur_dlz = get_ith_aircraft_dlz(env, i, enemy_aircraft_id)  # dlz part have alive judgment inside
            for item in cur_dlz:
                ret_native.append(item)
    ret_native.append(env.normalize(env.state_interface["tick"]))

    return ret_native


def get_aircraft_relative_state(env: BattleSpace, i, j):
    # this function for adding relative state which will cause redundancy if write in native state(like relative obs between aircrafts)
    ret_relative = []

    temp = []
    env.to_list(temp, env.state_interface["AMS"][i]["relative_observation"][j])
    for item in temp:
        ret_relative.append(env.normalize(item))

    return ret_relative


def get_msl_token(env: BattleSpace, i_aircraft, i_th_msl):
    # 2020/08/31 consider launched msl token are same, no need to add aircraft(SMS) id as position #
    aircraft = env.state_interface["AMS"][i_aircraft]
    msl = aircraft["SMS"][i_th_msl]
    token_valid = 0.0
    ret_msl = []

    if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying
        msl_target_index = int(msl["target_index"]["value"] + 0.1)
        msl_absolute_targt_index = msl_target_index + env.red if i_aircraft < env.red else msl_target_index
        target_aircraft = env.state_interface["AMS"][msl_absolute_targt_index]
        target_alive = target_aircraft["alive"]["value"]
        if int(target_alive + 0.1) == 1:
            token_valid = 1.0

    ret_msl.append(env.normalize(aircraft["alive"]))
    ret_msl.append(env.normalize(msl["state"]) * token_valid)
    ret_msl.append(env.normalize(msl["AO_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TAS_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TA_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TGO"]) * token_valid)
    # ret_msl.append(env.normalize(msl["Xg_m_0"]) * token_valid)
    # ret_msl.append(env.normalize(msl["Xg_m_1"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_2"]) * token_valid)
    ret_msl.append(env.normalize(msl["r_dot_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["target_index"]) * token_valid)

    return ret_msl


def get_kteam_msl_tokens(env: BattleSpace, k: int):
    msl_token = []
    for i in range(k * env.red, env.red + k * env.blue):
        aircraft = env.state_interface["AMS"][i]
        for msl_id in range(len(aircraft["SMS"])):
            msl_token.append(get_msl_token(env, i, msl_id))
    return msl_token


def get_kteam_aircraft_state(env: BattleSpace, k: int):
    # state consist of global/native/tokens #
    ground_truth_state = get_kteam_global_ground_truth_state(env, k, all_aircraft_info=False,
                                                             omit_token_state_info=False,
                                                             omit_msl_token_state=True)
    native_state = get_kteam_native_state(env, k, omit_token_state_info=True)
    aircraft_token_state = get_kteam_token_relative_state(env, k)
    self_msl_token_state = get_kteam_msl_tokens(env, k)
    bandit_msl_token_state = get_kteam_msl_tokens(env, (1 - k))

    state_input = [ground_truth_state, native_state, aircraft_token_state, self_msl_token_state, bandit_msl_token_state]
    return state_input


if __name__ == "__main__":
    env = Config.env
    env.reset()
    # ground_truth_state_r = get_kteam_global_ground_truth_state(env, 0)
    # ground_truth_state_b = get_kteam_global_ground_truth_state(env, 1)
    # print(ground_truth_state_r)
    # print(ground_truth_state_b)
    # print(len(ground_truth_state_r), len(ground_truth_state_b))
    #
    # state_native_r = get_kteam_native_state(env, 0)
    # state_native_b = get_kteam_native_state(env, 1)
    # print(state_native_r)
    # print(state_native_b)
    # print(len(state_native_r), len(state_native_b))
    #
    # state_native_token_r = get_kteam_token_relative_state(env, 0)
    # state_native_token_b = get_kteam_token_relative_state(env, 1)
    # print(state_native_token_r)
    # print(state_native_token_b)
    # print(len(state_native_token_r), len(state_native_token_b))
    # print(len(state_native_token_r[0]), len(state_native_token_b[0]))
    #
    # msl_token_self = get_kteam_msl_tokens(env, 0)
    # msl_token_bandit = get_kteam_msl_tokens(env, 1)
    # print(msl_token_self)
    # print(msl_token_bandit)
    # print(len(msl_token_self), len(msl_token_bandit))
    # print(len(msl_token_self[0]), len(msl_token_bandit[0]))

    state_input = get_kteam_aircraft_state(env, 0)
    print(len(state_input[0]))
    print(len(state_input[1]))
    print(len(state_input[2][0]))
    print(len(state_input[3][0]))






