# 2021/02/25 write by peng, state for independ method(commnet) without target_sorting, for normal link method
# all function input id use absolute id

from environment.battlespace import BattleSpace
from train.config import Config
import numpy as np


def index_to_one_hot(i, total):
    ret = np.zeros(total)
    ret[i] = 1
    return ret.tolist()


def get_aircraft_available(env, i):
    aircraft_i = env.state_interface["AMS"][i]
    flying_missile_num = 0
    for m in aircraft_i["SMS"]:
        if abs(m["state"]["value"] - 2) < 0.1 or abs(m["state"]["value"] - 3) < 0.1:
            flying_missile_num = flying_missile_num + 1
    if env.state_interface["AMS"][i]["alive"]["value"] < 0.1 and flying_missile_num is 0:
        return False
    return True


# self state is aircraft state without relative state
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

    return ret


# aircraft_i's DLZ to aircraft_j
def get_ith_aircraft_dlz_to_j(env, i: int, j: int):
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
        # ret.append(env.normalize(dlz["Raero"]))
        ret.append(env.normalize(dlz["Rmax"]))
        # ret.append(env.normalize(dlz["Rmin"]))
        # ret.append(env.normalize(dlz["Ropt"]))
        ret.append(env.normalize(dlz["Rpi"]))
        ret.append(env.normalize(dlz["Rtr"]))
    else:
        for _ in range(3):  # use 3 typical DLZ
            ret.append(0.0)

    return ret


# i to j aircraft relative obs
def get_aircraft_relative_obs(env, i: int, j: int, position_encoding=False):
    # relative obs of i to j #
    if i == j:
        print("no relative obs between same aircraft")
        exit(0)
    ret = []
    cur_relative_valid = get_aircraft_available(env, i) and get_aircraft_available(env, j)
    relative_obs = env.state_interface["AMS"][i]["relative_observation"][j]

    ret.append(env.normalize(relative_obs["AO"]) * cur_relative_valid)
    ret.append(env.normalize(relative_obs["TA"]) * cur_relative_valid)
    ret.append(env.normalize(relative_obs["TA_abs_dot"]) * cur_relative_valid)
    ret.append(env.normalize(relative_obs["Truth"]) * cur_relative_valid)
    ret.append(env.normalize(relative_obs["h_delta"]) * cur_relative_valid)
    ret.append(env.normalize(relative_obs["r"]) * cur_relative_valid)
    ret.append(env.normalize(relative_obs["r_dot"]) * cur_relative_valid)

    if position_encoding:
        aircraft_num = len(env.state_interface["AMS"])
        ret_i = index_to_one_hot(i, aircraft_num)
        ret_j = index_to_one_hot(j, aircraft_num)
        for item in ret_i:
            ret.append(item * cur_relative_valid)
        for item in ret_j:
            ret.append(item * cur_relative_valid)

    return ret


# i_th_msl of i_aircraft
def get_msl_token(env: BattleSpace, i_aircraft, i_th_msl):
    # 2020/08/31 consider launched msl token are same, no need to add aircraft(SMS) id as position
    # 2021/02/25 launched msl are not considered same while using normal_link
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
    ret_msl.append(env.normalize(msl["flying_time"]) * token_valid)
    ret_msl.append(env.normalize(msl["lost_fcs_guide_timer"]) * token_valid)
    ret_msl.append(env.normalize(msl["lost_radar_guide_timer"]) * token_valid)
    ret_msl.append(env.normalize(msl["flying_time"]) * token_valid)
    ret_msl.append(env.normalize(msl["r_dot_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["target_index"]) * token_valid)
    if i_aircraft < env.red:
        rel_i = i_aircraft
        launcher_id_one_hot = index_to_one_hot(rel_i, env.red)
    else:
        rel_i = i_aircraft - env.red
        launcher_id_one_hot = index_to_one_hot(rel_i, env.blue)
    ret_msl += launcher_id_one_hot

    return ret_msl


##################
# main functions #
##################

# ground truth state
def get_kteam_global_ground_truth_state_independ(env, k):
    ret_global = []
    team_num = env.blue if k else env.red
    bandit_num = env.red if k else env.blue

    for bandit_id in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
        # add self state of bandit
        bandit_state = get_ith_aircraft_self_state(env, bandit_id)
        bandit_state_len = len(bandit_state)
        bandit_valid = get_aircraft_available(env, bandit_id)
        bandit_state = bandit_state if bandit_valid else [0.0] * bandit_state_len
        # add bandit DLZ state to self team
        total_dlz_state = []
        for self_id in range(k * env.red, env.red + k * env.blue):
            dlz_state = get_ith_aircraft_dlz_to_j(env, bandit_id, self_id)
            total_dlz_state += dlz_state
        # total state #
        ret_global = ret_global + bandit_state + total_dlz_state

    # todo, did not add relative between bandits
    ret_global_tot = [ret_global] * team_num

    return ret_global_tot


# native state
def get_kteam_native_state_independ(env, k):
    ret_native = []
    for self_id in range(k * env.red, env.red + k * env.blue):
        self_state = get_ith_aircraft_self_state(env, self_id)
        self_state_len = len(self_state)
        self_valid = get_aircraft_available(env, self_id)
        self_state = self_state if self_valid else [0.0] * self_state_len
        ret_native.append(self_state)

    return ret_native


# aircraft relative token state
def get_ith_aircraft_token_index_pair(env, aircraft_id):
    index_pair = []
    k = 0 if aircraft_id < env.red else 1
    team_aircraft_num = env.blue if k else env.red
    bandit_aircraft_num = env.red if k else env.blue
    for i in range(team_aircraft_num):
        id_i = (i + aircraft_id) % team_aircraft_num + k * env.red
        for j in range(team_aircraft_num):
            id_j = j + k * env.red
            if id_i != id_j:  # add current pair if not to self
                pair = [id_i, id_j]
                index_pair.append(pair)
        for j in range(bandit_aircraft_num):
            id_j = j + (1 - k) * env.red
            pair = [id_i, id_j]
            index_pair.append(pair)

    return index_pair


def get_kteam_aircraft_tokens_independ(env, k):
    aircraft_tokens = []
    for self_id in range(k * env.red, env.red + k * env.blue):
        single_aircraft_tokens = []
        token_pair = get_ith_aircraft_token_index_pair(env, self_id)
        for pair in token_pair:
            single_aircraft_tokens.append(get_aircraft_relative_obs(env, pair[0], pair[1]))
        aircraft_tokens.append(single_aircraft_tokens)
    return aircraft_tokens


# aircraft self msl state, for normal link, agent only view missiles launched by self
def get_self_kteam_msl_tokens_independ_normal_link(env, k):
    self_team_msl_tokens = []
    for aircraft_id in range(k * env.red, env.red + k * env.blue):
        self_msl_tokens = []
        for msl_id in range(len(env.state_interface["AMS"][aircraft_id]["SMS"])):
            self_msl_tokens.append(get_msl_token(env, aircraft_id, msl_id))
        self_team_msl_tokens.append(self_msl_tokens)
    return self_team_msl_tokens


# aircraft bandit msl state
def get_bandit_kteam_msl_tokens_independ(env, k):
    bandit_team_msl_tokens = []
    for aircraft_id in range(k * env.red, env.red + k * env.blue):
        cur_aircraft_bandit_team_msl_token = []
        for bandit_id in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
            for msl_id in range(len(env.state_interface["AMS"][bandit_id]["SMS"])):
                cur_msl_token = get_msl_token(env, bandit_id, msl_id)
                cur_msl_target = int(env.state_interface["AMS"][bandit_id]["SMS"][msl_id]["target_index"]["value"] + 0.1)
                if cur_msl_target != -1 and cur_msl_target == (aircraft_id - k * env.red):
                    cur_aircraft_bandit_team_msl_token.append(cur_msl_token)
                else:
                    cur_aircraft_bandit_team_msl_token.append([0.0] * len(cur_msl_token))
        bandit_team_msl_tokens.append(cur_aircraft_bandit_team_msl_token)

    return bandit_team_msl_tokens


# id one hot state
def get_kteam_ids_one_hot_state(env, k):
    native_aircraft_num = env.blue if k else env.red
    ids_one_hot_state = []
    for id in range(native_aircraft_num):
        ids_one_hot_state.append(index_to_one_hot(id, native_aircraft_num))
    return ids_one_hot_state


# main state function
def get_kteam_aircraft_state(env, k: int):
    # state consist of global/native/tokens #

    ground_truth_state = get_kteam_global_ground_truth_state_independ(env, k)
    native_state = get_kteam_native_state_independ(env, k)
    aircraft_token_state = get_kteam_aircraft_tokens_independ(env, k)
    self_msl_token_state = get_self_kteam_msl_tokens_independ_normal_link(env, k)
    bandit_msl_token_state = get_bandit_kteam_msl_tokens_independ(env, k)
    id_one_hot_state = get_kteam_ids_one_hot_state(env, k)

    state_input = [ground_truth_state, native_state, aircraft_token_state, self_msl_token_state, bandit_msl_token_state,
                   id_one_hot_state]

    return state_input


if __name__ == "__main__":
    env = BattleSpace()
    env.random_init()
    env.reset()

    aircraft_state = get_kteam_aircraft_state(env, 0)
    print(aircraft_state)
    # state len: global 38, native 13, token 15 * 6, msl 15 * 4, bandit 15 * 8, one_hot 2

