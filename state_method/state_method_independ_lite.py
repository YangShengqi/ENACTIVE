from environment.battlespace import BattleSpace
from train.config import Config
import numpy as np


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
    return ret.tolist()


def get_kteam_ids_one_hot_state(env, k):
    native_aircraft_num = env.blue if k else env.red
    ids_one_hot_state = []
    for id in range(native_aircraft_num):
        ids_one_hot_state.append(index_to_one_hot(id, native_aircraft_num))
    return ids_one_hot_state


def get_friend_idx_list(env, i: int, k: int):
    native_idx_list = [j for j in range(k * env.red, env.red + k * env.blue)]
    native_idx_list.remove(i)
    return native_idx_list


def get_bandit_idx_list(env, k: int):
    bandit_idx_list = [j for j in range((1 - k) * env.red, env.red + (1 - k) * env.blue)]
    return bandit_idx_list


def get_aircraft_index_list(env, k: int):
    aircraft_index_list = []
    # get final aircraft_index_list
    for i in range(k * env.red, env.red + k * env.blue):
        friend_idx_list = get_friend_idx_list(env, i, k)
        bandit_idx_list = get_bandit_idx_list(env, k)
        aircraft_index_list.append([i] + friend_idx_list + bandit_idx_list)

    return aircraft_index_list


def get_kteam_global_ground_truth_state(env, k, aircraft_index_list):
    ret_global = []
    native_num = env.blue if k else env.red
    bandit_num = env.red if k else env.blue

    for id_list in aircraft_index_list:
        sub_ret_global = []
        for other_id in id_list[native_num:]:
            cur_state = get_ith_aircraft_self_state(env, other_id)
            cur_aircraft_available = get_aircraft_available(env, other_id)
            for item in cur_state:
                if cur_aircraft_available:
                    sub_ret_global.append(item)
                else:
                    sub_ret_global.append(0.0)

            # add DLZ state #
            if id_list.index(other_id) >= native_num:
                opponent_id_list = id_list[:native_num]
            else:
                opponent_id_list = id_list[-bandit_num:]
            for opponent_id in opponent_id_list:
                cur_dlz = get_ith_aircraft_dlz(env, other_id, opponent_id)
                for item in cur_dlz:
                    sub_ret_global.append(item)  # dlz part have alive judgment inside

        for index0 in range(native_num, len(id_list)):
            for index1 in range(index0 + 1, len(id_list)):
                cur_relative_state = get_aircraft_relative_obs(env, id_list[index0], id_list[index1], position_encoding=False)
                for item in cur_relative_state:
                    sub_ret_global.append(item)

        ret_global.append(sub_ret_global)

    return ret_global


def get_ith_aircraft_self_state(env, i: int):
    ret = []
    aircraft_i = env.state_interface["AMS"][i]
    ret.append(env.normalize(aircraft_i["AAM_remain"]))
    ret.append(env.normalize(aircraft_i["RWR_nailed"]))
    ret.append(env.normalize(aircraft_i["RWR_spiked"]))
    ret.append(env.normalize(aircraft_i["TAS"]))
    ret.append(env.normalize(aircraft_i["h_dot"]))
    ret.append(env.normalize(aircraft_i["residual_chi"]))
    ret.append(env.normalize(aircraft_i["Xg_2"]))

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


def get_ith_aircraft_dlz(env, i: int, j: int):
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
        ret.append(env.normalize(dlz["Rmax"]))
        ret.append(env.normalize(dlz["Rpi"]))
        ret.append(env.normalize(dlz["Rtr"]))
    else:
        for _ in range(6):
            ret.append(0.0)

    return ret


def get_aircraft_relative_obs(env, i: int, j: int, position_encoding=True):  # this may use for token #
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


def get_kteam_native_state(env, k, aircraft_index_list, omit_token_state_info=True, omit_msl_token_state=True):
    ret_native = []
    bandit_num = env.red if k else env.blue

    for id_list in aircraft_index_list:
        sub_ret_native = []
        self_id = id_list[0]
        cur_state = get_ith_aircraft_self_state(env, self_id)
        cur_aircraft_available = get_aircraft_available(env, self_id)
        for item in cur_state:
            if cur_aircraft_available:
                sub_ret_native.append(item)
            else:
                sub_ret_native.append(0.0)

        sub_ret_native.append(env.normalize(env.state_interface["tick"]))

        ret_native.append(sub_ret_native)

    return ret_native


def get_kteam_token_relative_state(env, aircraft_index_list):
    relative_state_token = []

    for id_list in aircraft_index_list:
        sub_relative_state_token = []
        self_id = id_list[0]
        for other_id in id_list[1:]:
            sub_relative_state_token.append(get_token_relative_state(env, self_id, other_id, position_encoding=True))
        relative_state_token.append(sub_relative_state_token)

    return relative_state_token


def get_token_relative_state(env, i: int, j: int, position_encoding=True):
    # relative of i to j #
    # input i for self aircraft id, j for opponent id, considering 2 vs 2 #
    # i_friend = (env.red - i - 1) if i < env.red else (2 * env.red + env.blue - i - 1)
    # relative_i = i - env.red if i >= env.red else i
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
        ret_token.append(env.normalize(DLZ_list["Rmax"]))
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


def get_self_kteam_msl_tokens(env, k, aircraft_index_list):
    msl_token = []
    native_num = env.blue if k else env.red

    for id_list in aircraft_index_list:
        sub_msl_token = []
        tgo_list = []
        for i in id_list[:native_num]:
            aircraft = env.state_interface["AMS"][i]
            for msl_id in range(len(aircraft["SMS"])):
                tgo_list.append(aircraft["SMS"][msl_id]["TGO"]["value"])
        tgo_list = np.array(tgo_list)
        tgo_sort_list = (np.argsort(tgo_list)).tolist()
        for j in range(Config.max_msl_token_num_blue if k else Config.max_msl_token_num_red):
            air_i = int(tgo_sort_list[j] / 4)
            air_i = id_list[air_i]
            msl_i = tgo_sort_list[j] % 4
            sub_msl_token.append(get_self_msl_token(env, air_i, msl_i))

        msl_token.append(sub_msl_token)

    return msl_token


def get_self_msl_token(env, i_aircraft, i_th_msl):
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
    ret_msl.append(env.normalize(msl["Xg_m_2"]) * token_valid)
    ret_msl.append(env.normalize(msl["r_dot_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["target_index"]) * token_valid)

    return ret_msl


def get_bandit_kteam_msl_tokens(env, k, aircraft_index_list):
    msl_token = []
    bandit_num = env.red if k else env.blue
    native_num = env.blue if k else env.red

    for id_list in aircraft_index_list:
        sub_msl_token = []
        tgo_list = []
        self_id = id_list[0]
        for i in id_list[-bandit_num:]:
            aircraft = env.state_interface["AMS"][i]
            for msl_id in range(len(aircraft["SMS"])):
                tgo_list.append(aircraft["SMS"][msl_id]["TGO"]["value"])
        tgo_list = np.array(tgo_list)
        tgo_sort_list = (np.argsort(tgo_list)).tolist()
        for j in range(Config.max_msl_token_num_red if k else Config.max_msl_token_num_blue):
            air_i = int(tgo_sort_list[j] / 4)
            air_i = id_list[air_i + native_num]
            msl_i = tgo_sort_list[j] % 4
            sub_msl_token.append(get_bandit_msl_token(env, air_i, msl_i, self_id))

        msl_token.append(sub_msl_token)

    return msl_token


def get_bandit_msl_token(env, i_aircraft, i_th_msl, self_id):
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
        if int(target_alive + 0.1) == 1 and msl_absolute_targt_index == self_id:
            token_valid = 1.0

    ret_msl.append(env.normalize(aircraft["alive"]))
    ret_msl.append(env.normalize(msl["state"]) * token_valid)
    ret_msl.append(env.normalize(msl["AO_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TAS_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TA_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TGO"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_2"]) * token_valid)
    ret_msl.append(env.normalize(msl["r_dot_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["target_index"]) * token_valid)

    return ret_msl


def get_kteam_aircraft_state(env, k: int):
    # state consist of global/native/tokens #
    aircraft_index_list = get_aircraft_index_list(env, k)

    ground_truth_state = get_kteam_global_ground_truth_state(env, k, aircraft_index_list)
    native_state = get_kteam_native_state(env, k, aircraft_index_list)
    aircraft_token_state = get_kteam_token_relative_state(env, aircraft_index_list)
    self_msl_token_state = get_self_kteam_msl_tokens(env, k, aircraft_index_list)
    bandit_msl_token_state = get_bandit_kteam_msl_tokens(env, k, aircraft_index_list)
    id_one_hot_state = get_kteam_ids_one_hot_state(env, k)

    state_input = [ground_truth_state, native_state, aircraft_token_state, self_msl_token_state, bandit_msl_token_state,
                   id_one_hot_state]

    return state_input


if __name__ == "__main__":
    env = Config.env
    env.reset()
    a = get_aircraft_index_list(env, 0)
    b = get_aircraft_index_list(env, 1)

    ground_truth_state_r = get_kteam_global_ground_truth_state(env, 0, a)
    ground_truth_state_b = get_kteam_global_ground_truth_state(env, 1, b)
    # print(ground_truth_state_r)
    # print(ground_truth_state_b)
    # print(len(ground_truth_state_r), len(ground_truth_state_b))
    #
    # state_native_r = get_kteam_native_state(env, 0, a)
    # state_native_b = get_kteam_native_state(env, 1, b)
    # print(state_native_r)
    # print(state_native_b)
    # print(len(state_native_r), len(state_native_b))
    #
    # state_native_token_r = get_kteam_token_relative_state(env, a)
    # print(state_native_token_r)
    # print(state_native_token_b)
    # print(len(state_native_token_r), len(state_native_token_b))
    # print(len(state_native_token_r[0]), len(state_native_token_b[0]))
    #
    # msl_token_self_r = get_self_kteam_msl_tokens(env, 0, a)
    # msl_token_self_b =                 ground_truth_size_before_cat=(256, 256),
    #              native_hidden_size=(256, 128),
    #              policy_hidden_size=(128, 128, 64),
    #              value_hidden_size=(128, 64),
    #              state_token_embed_dim=100, state_token_num_heads=4, atten_depth=2,
    #              msl_token_embed_dim=32, msl_token_num_heads=4,
    #              activation='tanh', init_method='xavier', aircraft_num=2):
    #     super().__init__() get_self_kteam_msl_tokens(env, 1, b)
    # msl_token_bandit_r = get_bandit_kteam_msl_tokens(env, 0, a)
    # msl_token_bandi_b = get_bandit_kteam_msl_tokens(env, 1, b)

    # threat_b = threat_sort(Config.env, 0)
    # threat_r = threat_sort(Config.env, 1)

    # a = get_friend_idx_list(env, 1, 0)
    # b = get_friend_idx_list(env, 3, 1)
    #
    # a = get_bandit_idx_list(env, 2, 1, True)
    # b = get_bandit_idx_list(env, 0, 0, True)

    # a = get_shot_mask(env, 0, 0, action_sort=True, state_sort=False)
    # b = get_shot_mask(env, 2, 1, action_sort=True, state_sort=False)

    # target_mask_env = env.action_interface["AMS"][1]["SemanticManeuver"]["maneuver_target"]["mask"]
    # a = get_target_mask(env, 1, 0, target_mask_env, action_sort=True, state_sort=False)
    # target_mask_env = env.action_interface["AMS"][3]["SemanticManeuver"]["maneuver_target"]["mask"]
    # b = get_target_mask(env, 3, 1, target_mask_env, action_sort=True, state_sort=False)

    state_input = get_kteam_aircraft_state(env, 0)
    print(len(state_input[0][0]))
    print(len(state_input[1][0]))
    print(len(state_input[2][0][0]))
    print(len(state_input[3][0][0]))






