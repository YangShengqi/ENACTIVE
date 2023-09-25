from environment.battlespace import BattleSpace
from train.config import Config
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


def get_self_state(env, i):
    ret = []
    aircraft_i = env.state_interface["AMS"][i]
    k = 0 if i < env.red else 1  # team id

    ret.append(env.normalize(aircraft_i["AAM_remain"]))
    ret.append(env.normalize(aircraft_i["TAS"]))
    ret.append(env.normalize(aircraft_i["h_dot"]))
    ret.append(env.normalize(aircraft_i["Xg_2"]))
    ret.append(env.normalize(aircraft_i["residual_chi"]))
    ret.append(env.normalize(aircraft_i["attg_0"]))
    ret.append(env.normalize(aircraft_i["attg_1"]))
    ret.append(env.normalize(aircraft_i["n_y"]))
    ret.append(env.normalize(aircraft_i["out_of_border_time"]))
    ret.append(env.normalize(aircraft_i["out_of_border_distance"]))
    ret.append(env.normalize(aircraft_i["alive"]))

    # 2. sensor relative state #
    ret.append(env.normalize(aircraft_i["RWR_nailed"]))
    ret.append(env.normalize(aircraft_i["RWR_spiked"]))
    for bandit_relative_id in range(len(aircraft_i["RWRModel"])):
        bandit_absolute_id = bandit_relative_id + (1 - k) * env.red
        if int(env.state_interface["AMS"][bandit_absolute_id]["alive"]["value"] + 0.1) == 1:  # enemy alive
            ret.append(env.normalize_with_bias(aircraft_i["RWRModel"][bandit_relative_id]["RWR_fetched"]))
        else:
            ret.append(0.0)
    for bandit_relative_id in range(len(aircraft_i["RadarModel"])):
        bandit_absolute_id = bandit_relative_id + (1 - k) * env.red
        if int(env.state_interface["AMS"][bandit_absolute_id]["alive"]["value"] + 0.1) == 1:  # enemy alive
            ret.append(env.normalize_with_bias(aircraft_i["RadarModel"][bandit_relative_id]["FCR_locked"]))
        else:
            ret.append(0.0)

    # 3. DLZ info #
    for bandit_relative_id in range(len(aircraft_i["attack_zone_list"])):
        bandit_absolute_id = bandit_relative_id + (1 - k) * env.red
        if int(env.state_interface["AMS"][bandit_absolute_id]["alive"]["value"] + 0.1) == 1:  # enemy alive
            DLZ_list = aircraft_i["attack_zone_list"][bandit_relative_id]
            ret.append(env.normalize(DLZ_list["Rmax"]))
            ret.append(env.normalize(DLZ_list["Rpi"]))
            ret.append(env.normalize(DLZ_list["Rtr"]))
        else:
            ret.append(0.0)
            ret.append(0.0)
            ret.append(0.0)

    return ret


def get_relative_state(env: BattleSpace, i, j):
    ret_relative = []

    temp = []
    env.to_list(temp, env.state_interface["AMS"][i]["relative_observation"][j])
    for item in temp:
        ret_relative.append(env.normalize(item))

    return ret_relative


def get_native_state(env, k):
    if k > 0:
        i = 1
    else:
        i = 0

    ret_native = []
    native_state = get_self_state(env, i)
    aircraft_available = get_aircraft_available(env, i)
    native_available = 1.0 if aircraft_available else 0.0
    for item in native_state:
        ret_native.append(item * native_available)

    j = 1 - i
    relative_state = get_relative_state(env, i, j)
    if get_aircraft_available(env, i) and get_aircraft_available(env, j):
        relative_valid = 1.0
    else:
        relative_valid = 0.0
    for item in relative_state:
        ret_native.append(item * relative_valid)

    return ret_native


# def get_global_state(env: BattleSpace, k: int):
#     # add aircraft native state #
#     ret_global = []
#     for aircraft_id in range(env.red + env.blue):
#         aircraft_absolute_id = (aircraft_id + k * env.red) % (env.red + env.blue)
#         ret_native = get_self_state(env, aircraft_absolute_id)
#         aircraft_available = get_aircraft_available(env, aircraft_absolute_id)
#         native_available = 1.0 if aircraft_available else 0.0
#         for item in ret_native:
#             ret_global.append(item * native_available)
#     # add aircraft relative state #
#     for id_i in range(env.blue + env.red):
#         for id_j in range(id_i + 1, env.red + env.blue):
#             i = (k * env.red + id_i) % (env.red + env.blue)
#             j = (k * env.red + id_j) % (env.red + env.blue)
#             ret_relative = get_relative_state(env, i, j)
#             if get_aircraft_available(env, i) and get_aircraft_available(env, j):
#                 relative_valid = 1.0
#             else:
#                 relative_valid = 0.0
#             for item in ret_relative:
#                 ret_global.append(item * relative_valid)
#
#     return ret_global
def get_global_state(env: BattleSpace, k: int):
    if k > 0:
        i = 1
        j = 0
    else:
        i = 0
        j = 1
    # add aircraft native state #
    ret_global = []
    ret_native = get_self_state(env, j)
    aircraft_available = get_aircraft_available(env, j)
    native_available = 1.0 if aircraft_available else 0.0
    for item in ret_native:
        ret_global.append(item * native_available)
    # add aircraft relative state #
    ret_relative = get_relative_state(env, j, i)
    if get_aircraft_available(env, i) and get_aircraft_available(env, j):
        relative_valid = 1.0
    else:
        relative_valid = 0.0
    for item in ret_relative:
        ret_global.append(item * relative_valid)

    return ret_global


def get_self_msl(env: BattleSpace, i_aircraft, i_th_msl):
    aircraft = env.state_interface["AMS"][i_aircraft]
    msl = aircraft["SMS"][i_th_msl]
    token_valid = 0.0
    ret_msl = []
    if int(aircraft["alive"]["value"] + 0.1) == 1:  # aircraft alive
        if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying
            msl_target_index = int(msl["target_index"]["value"] + 0.1)
            msl_absolute_targt_index = msl_target_index + env.red if i_aircraft < env.red else msl_target_index
            target_aircraft = env.state_interface["AMS"][msl_absolute_targt_index]
            target_alive = target_aircraft["alive"]["value"]
            if int(target_alive + 0.1) == 1:
                token_valid = 1.0

    ret_msl.append(env.normalize(msl["state"]) * token_valid)
    ret_msl.append(env.normalize(msl["AO_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TA_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TAS_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["r_dot_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_2"]) * token_valid)
    ret_msl.append(env.normalize(msl["TGO"]) * token_valid)
    ret_msl.append(env.normalize(msl["flying_time"]) * token_valid)
    ret_msl.append(env.normalize(msl["lost_fcs_guide_timer"]) * token_valid)
    ret_msl.append(env.normalize(msl["lost_radar_guide_timer"]) * token_valid)
    ret_msl.append(env.normalize(msl["target_index"]) * token_valid)

    return ret_msl


def get_bandit_msl(env: BattleSpace, i_aircraft, i_th_msl, self_id):
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

    ret_msl.append(env.normalize(msl["state"]) * token_valid)
    ret_msl.append(env.normalize(msl["AO_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TA_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TAS_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["r_dot_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_2"]) * token_valid)
    ret_msl.append(env.normalize(msl["TGO"]) * token_valid)
    ret_msl.append(env.normalize(msl["flying_time"]) * token_valid)
    ret_msl.append(env.normalize(msl["lost_fcs_guide_timer"]) * token_valid)
    ret_msl.append(env.normalize(msl["lost_radar_guide_timer"]) * token_valid)

    return ret_msl


def get_self_msl_tokens(env: BattleSpace, k: int):
    msl_token = []
    for i in range(k * env.red, env.red + k * env.blue):
        aircraft = env.state_interface["AMS"][i]
        for msl_id in range(len(aircraft["SMS"])):
            msl_token.append(get_self_msl(env, i, msl_id))
    return msl_token


def get_bandit_msl_tokens(env: BattleSpace, k: int):
    msl_token = []
    if k > 0:
        self_id = 1
        bandit_id = 0
    else:
        self_id = 0
        bandit_id = 1
    aircraft = env.state_interface["AMS"][bandit_id]
    for msl_id in range(len(aircraft["SMS"])):
        msl_token.append(get_bandit_msl(env, bandit_id, msl_id, self_id))
    return msl_token


if __name__ == "__main__":
    Config.env.random_init()
    Config.env.reset()

    # state = get_kteam_aircraft_state(Config.env, 1)

    # state = get_kteam_aircraft_state_for_attention(Config.env, 1)
    # print(state)

    # state_0 = get_kteam_global_ground_truth_state(Config.env, 0)
    # print(len(state_0))
    # state_1 = get_kteam_global_ground_truth_state(Config.env, 1)
    # print(len(state_1))
    #
    # state_native = get_kteam_aircraft_state_for_attention(Config.env, 1)
    # print(len(state_native[0]))
    #
    # get_msl_token(Config.env, 1, 1)

    # state = get_ith_aircraft_ground_truth_native_state(Config.env, 0)
    # print(state)
    # print(len(state))

    # state = get_aircraft_ground_truth_relative_state(Config.env, 0, 1)
    # print(state, len(state))

    # ret_msl = get_msl_token(Config.env, 0, 1)
    # print(ret_msl)
    #
    # msl_token = get_kteam_msl_tokens(Config.env, 1)
    # print(msl_token)
    # print(len(msl_token))
    env = Config.env
    last_time_maneuver = [[[0, 0, 0, 0], [0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0]]]
    # list = get_kteam_aircraft_state_for_attention(env, 0, last_time_maneuver=last_time_maneuver)
    list = get_kteam_aircraft_state_for_attention_refactor(env, 0, last_time_maneuver_one_hot=last_time_maneuver)
    print(len(list[0]))
    print(len(list[1]))
    # print(len(list))
    # print(list[0][0])
    # print(list[0][1])
