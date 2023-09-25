from environment.battlespace import BattleSpace
from train.config import Config
import copy
from utils.math import index_to_one_hot


def get_id_list(env: BattleSpace, k: int, threat_bandit):
    threat_bandit_rel = []
    for element in threat_bandit:
        threat_bandit_rel.append([element[i] + env.red if k else element[i] for i in range(len(element))])
    threat_bandit_rel_true = []
    for element in threat_bandit:
        threat_bandit_rel_true.append([element[i] if k else element[i] - env.red for i in range(len(element))])

    threat_friend_rel = [[0, 1], [1, 0]]
    id_list = []
    for i in range(env.blue if k else env.red):
        origin_list = []
        origin_list.extend(threat_friend_rel[i])
        origin_list.extend(threat_bandit_rel[i])
        id_list.append(origin_list)
    return id_list, threat_bandit_rel_true, threat_bandit


def shot_mask_sort(env: BattleSpace, i: int, k: int, threat_global_idx: list):
    shot_mask = env.action_interface["AMS"][i]["action_shoot_target"]["mask"]
    threat_bandit = [threat_global_idx[j if k else j + env.red] for j in range(len(shot_mask))]
    threat_bandit_rel = [threat_bandit[j] if k else threat_bandit[j] - env.red for j in range(len(threat_bandit))]
    shot_mask_sorted = [shot_mask[threat_bandit_rel[j]] for j in range(len(shot_mask))]
    return [1] + shot_mask_sorted


def target_mask_sort(env: BattleSpace, i: int, threat_local_idx: list, target_type):
    target_mask_sorted = []
    target_mask = env.action_interface["AMS"][i]["HybridManeuver"]["maneuver_target"]["mask"]
    if target_type == "origin":
        target_mask_sorted = [target_mask[threat_local_idx[j]] for j in range(len(target_mask))]
    elif target_type == "without_self":
        cur_target_mask = []
        for j in range(len(target_mask)):
            if threat_local_idx[j] == i:
                pass
            else:
                cur_target_mask.append(target_mask[threat_local_idx[j]])
        target_mask_sorted.extend(cur_target_mask)
    elif target_type == "only_enemy":
        team_num = env.red if i < env.red else env.blue
        bandit_num = env.blue if i < env.red else env.red
        target_mask_sorted = [target_mask[threat_local_idx[j + team_num]] for j in range(bandit_num)]
    return target_mask_sorted


def get_ith_aircraft_state(env: BattleSpace, i: int, all_relative_obs=False):
    ret = []
    aircraft_i = env.state_interface["AMS"][i]
    ret.append(env.normalize(aircraft_i["AAM_remain"]))
    ret.append(env.normalize(aircraft_i["FCR_locked"]))
    ret.append(env.normalize(aircraft_i["RWR_nailed"]))
    ret.append(env.normalize(aircraft_i["RWR_spiked"]))
    ret.append(env.normalize(aircraft_i["TAS"]))
    ret.append(env.normalize(aircraft_i["Vg_0"]))
    ret.append(env.normalize(aircraft_i["Vg_1"]))
    ret.append(env.normalize(aircraft_i["Vg_2"]))
    ret.append(env.normalize(aircraft_i["Xg_0"]))
    ret.append(env.normalize(aircraft_i["Xg_1"]))
    ret.append(env.normalize(aircraft_i["Xg_2"]))
    ret.append(env.normalize(aircraft_i["h_dot"]))
    ret.append(env.normalize(aircraft_i["residual_chi"]))
    for j in aircraft_i["FCSModel"]:
        for k in j:
            ret.append(env.normalize(k["FCS_available"]))
    for j in aircraft_i["RWRModel"]:
        for k in j:
            ret.append(env.normalize(k["RWR_fetched"]))
    for j in aircraft_i["RadarModel"]:
        for k in j:
            ret.append(env.normalize(k["FCR_locked"]))

    for m in aircraft_i["SMS"]:
        for j in m["FCSGuide"]:
            for k in j:
                ret.append(env.normalize(k["fcs_guide_info"]))
    for m in aircraft_i["SMS"]:
        for j in m["RadarGuide"]:
            for k in j:
                ret.append(env.normalize(k["radar_guide_info"]))
    for m in aircraft_i["SMS"]:
        ret.append(env.normalize(m["state"]))

    for j in range(len(env.state_interface["AMS"])):
        for k in env.state_interface["AMS"][env.resort(i, j)]["SMS"]:
            ret.append(1.0 - env.normalize(k["TGO"]))
    for j in range(len(env.state_interface["AMS"])):
        ret.append(env.normalize(env.state_interface["AMS"][env.resort(i, j)]["alive"]))
    temp = []
    if not all_relative_obs:
        for j in range(env.red + env.blue):
            env.to_list(temp, aircraft_i["relative_observation"][env.resort(i, j)])
    else:
        for j in range(env.blue + env.red):
            for t in range(j + 1, env.red + env.blue):
                env.to_list(temp, env.state_interface["AMS"][env.resort(i, j)]["relative_observation"][
                    env.resort(i, t)])
    for k in temp:
        ret.append(env.normalize(k))
    return ret


def get_ith_aircraft_state_without_tgo(env: BattleSpace, i: int, all_relative_obs=False):
    ret = []
    aircraft_i = env.state_interface["AMS"][i]
    ret.append(env.normalize(aircraft_i["AAM_remain"]))
    ret.append(env.normalize(aircraft_i["FCR_locked"]))
    ret.append(env.normalize(aircraft_i["RWR_nailed"]))
    ret.append(env.normalize(aircraft_i["RWR_spiked"]))
    ret.append(env.normalize(aircraft_i["TAS"]))
    ret.append(env.normalize(aircraft_i["Vg_0"]))
    ret.append(env.normalize(aircraft_i["Vg_1"]))
    ret.append(env.normalize(aircraft_i["Vg_2"]))
    ret.append(env.normalize(aircraft_i["Xg_0"]))
    ret.append(env.normalize(aircraft_i["Xg_1"]))
    ret.append(env.normalize(aircraft_i["Xg_2"]))
    ret.append(env.normalize(aircraft_i["h_dot"]))
    ret.append(env.normalize(aircraft_i["residual_chi"]))

    index = []
    for j in range(env.red + env.blue):
        index.append(env.resort(i, j))
    if i < env.red:
        for j in range(env.red):
            for k in env.state_interface["AMS"][index[j]]["SMS"]:
                ret.append(1.0 - env.normalize(k["TGO"]))
    else:
        for j in range(env.blue):
            for k in env.state_interface["AMS"][index[j]]["SMS"]:
                ret.append(1.0 - env.normalize(k["TGO"]))

    for j in range(len(env.state_interface["AMS"])):
        ret.append(env.normalize(env.state_interface["AMS"][env.resort(i, j)]["alive"]))
    temp = []
    if not all_relative_obs:
        for j in range(env.red + env.blue):
            env.to_list(temp, aircraft_i["relative_observation"][env.resort(i, j)])
    else:
        for j in range(env.blue + env.red):
            for t in range(j + 1, env.red + env.blue):
                env.to_list(temp,
                             env.state_interface["AMS"][env.resort(i, j)]["relative_observation"][env.resort(i, t)])
    for k in temp:
        ret.append(env.normalize(k))
    return ret


def get_kteam_aircraft_state(env: BattleSpace, k: int):
    ret = []
    for i in range(k * env.red, env.red + k * env.blue):
        aircraft_i = env.state_interface["AMS"][i]
        ret.append(env.normalize(aircraft_i["AAM_remain"]))
        #ret.append(env.normalize(aircraft_i["FCR_locked"]))
        ret.append(env.normalize(aircraft_i["RWR_nailed"]))
        ret.append(env.normalize(aircraft_i["RWR_spiked"]))
        ret.append(env.normalize(aircraft_i["TAS"]))
        ret.append(env.normalize(aircraft_i["Vg_0"]))
        ret.append(env.normalize(aircraft_i["Vg_1"]))
        ret.append(env.normalize(aircraft_i["Vg_2"]))
        ret.append(env.normalize(aircraft_i["Xg_0"]))
        ret.append(env.normalize(aircraft_i["Xg_1"]))
        ret.append(env.normalize(aircraft_i["Xg_2"]))
        ret.append(env.normalize(aircraft_i["h_dot"]))
        ret.append(env.normalize(aircraft_i["residual_chi"]))
        for j in aircraft_i["FCSModel"]:
            ret.append(env.normalize(j["FCS_available"]))
        for j in aircraft_i["RWRModel"]:
            ret.append(env.normalize(j["RWR_fetched"]))
        for j in aircraft_i["RadarModel"]:
            ret.append(env.normalize(j["FCR_locked"]))
        for m in aircraft_i["SMS"]:
            for j in m["FCSGuide"]:
                ret.append(env.normalize(j["fcs_guide_info"]))
        for m in aircraft_i["SMS"]:
            for j in m["RadarGuide"]:
                ret.append(env.normalize(j["radar_guide_info"]))
        for m in aircraft_i["SMS"]:
            ret.append(env.normalize(m["state"]))

        # add attack zone list
        for DLZ_list in aircraft_i["attack_zone_list"]:
            ret.append(env.normalize(DLZ_list["Raero"]))
            ret.append(env.normalize(DLZ_list["Rmax"]))
            ret.append(env.normalize(DLZ_list["Rmin"]))
            ret.append(env.normalize(DLZ_list["Ropt"]))
            ret.append(env.normalize(DLZ_list["Rpi"]))
            ret.append(env.normalize(DLZ_list["Rtr"]))
        # add attitude angle
        ret.append(env.normalize(aircraft_i["attg_0"]))
        ret.append(env.normalize(aircraft_i["attg_1"]))
        ret.append(env.normalize(aircraft_i["attg_2"]))
        # add overload
        ret.append(env.normalize(aircraft_i["n_y"]))
        # add out border time
        ret.append(env.normalize(aircraft_i["out_border_event"]))

    if Config.rnn_switch_on:
        if k < 1:
            for i in range(env.red):
                for j in env.state_interface["AMS"][i]["SMS"]:
                    ret.append(1.0 - env.normalize(j["TGO"]))
            for i in range(env.red):
                for j in range(env.blue):
                    for n in env.state_interface["AMS"][i]["SMS_est_list"][j]:
                        ret.append(1.0 - env.normalize(n["TGO_est"]))
        else:
            for i in range(env.blue):
                for j in env.state_interface["AMS"][env.red + i]["SMS"]:
                    ret.append(1.0 - env.normalize(j["TGO"]))
            for i in range(env.blue):
                for j in range(env.red):
                    for n in env.state_interface["AMS"][env.red + i]["SMS_est_list"][j]:
                        ret.append(1.0 - env.normalize(n["TGO_est"]))
    else:
        for i in range(env.red + env.blue):
            for j in env.state_interface["AMS"][(k * env.red + i) % (env.red + env.blue)]["SMS"]:
                ret.append(1.0 - env.normalize(j["TGO"]))
    for i in range(env.red + env.blue):
        ret.append(env.normalize(env.state_interface["AMS"][(k * env.red + i) % (env.red + env.blue)]["alive"]))
    temp = []
    for j in range(env.blue + env.red):
        for t in range(j + 1, env.red + env.blue):
            env.to_list(temp, env.state_interface["AMS"][(k * env.red + j) % (env.red + env.blue)][
                "relative_observation"][(k * env.red + t) % (env.red + env.blue)])
    for k in temp:
        ret.append(env.normalize(k))
    return ret


def get_aircraft_available(env, i):
    aircraft_i = env.state_interface["AMS"][i]
    flying_missile_num = 0
    for m in aircraft_i["SMS"]:
        if abs(m["state"]["value"] - 2) < 0.1 or abs(m["state"]["value"] - 3) < 0.1:
            flying_missile_num = flying_missile_num + 1
    if env.state_interface["AMS"][i]["alive"]["value"] < 0.1 and flying_missile_num is 0:
        return False
    return True


def get_kteam_aircraft_state_for_attention(env: BattleSpace, k: int, zerolize_invalid_token=True):
    id_list = []
    origin_list = [i for i in range(env.red + env.blue)]
    for i in range(env.red):
        id_list.append(swap(origin_list, 0, i))

    ret_native = []
    ret_token = []

    for list in id_list:
        sub_ret_native = []
        # first for native observations #
        for aircraft_id in list[: env.red]:
            i = (aircraft_id + k * env.red) % (env.red + env.blue)
            aircraft_i = env.state_interface["AMS"][i]
            aircraft_available = get_aircraft_available(env, i)

            sub_ret_native.append(env.normalize(aircraft_i["AAM_remain"]) if aircraft_available else 0.0)
            # sub_ret_native.append(env.normalize(aircraft_i["FCR_locked"]))
            sub_ret_native.append(env.normalize(aircraft_i["RWR_nailed"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["RWR_spiked"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["TAS"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["h_dot"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["residual_chi"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["Xg_2"]) if aircraft_available else 0.0)
            for j in aircraft_i["FCSModel"]:
                sub_ret_native.append(env.normalize(j["FCS_available"]) if aircraft_available else 0.0)
            for j in aircraft_i["RWRModel"]:
                sub_ret_native.append(env.normalize(j["RWR_fetched"]) if aircraft_available else 0.0)
            for j in aircraft_i["RadarModel"]:
                sub_ret_native.append(env.normalize(j["FCR_locked"]) if aircraft_available else 0.0)
            for m in aircraft_i["SMS"]:
                for j in m["FCSGuide"]:
                    sub_ret_native.append(env.normalize(j["fcs_guide_info"]))
            for m in aircraft_i["SMS"]:
                for j in m["RadarGuide"]:
                    sub_ret_native.append(env.normalize(j["radar_guide_info"]))
            for m in aircraft_i["SMS"]:
                sub_ret_native.append(env.normalize(m["state"]))
                sub_ret_native.append(env.normalize(m["target_index"]))
                sub_ret_native.append(1.0 - env.normalize(m["TGO"]))
                # add msl relative states #
                if int(m["state"]["value"] + 0.1) == 2 or int(m["state"]["value"] + 0.1) == 3:  # missile flying
                    sub_ret_native.append(env.normalize(m["r_dot_m"]))
                    sub_ret_native.append(env.normalize(m["TAS_m"]))
                    sub_ret_native.append(env.normalize(m["TA_m"]))
                    sub_ret_native.append(env.normalize(m["AO_m"]))
                else:
                    sub_ret_native.append(0)  # missle mounting or escaping #
                    sub_ret_native.append(0)
                    sub_ret_native.append(0)
                    sub_ret_native.append(0)

            # add attack zone list
            for enemy_id in range(len(aircraft_i["attack_zone_list"])):
                if get_aircraft_available(env, enemy_id + (1 - k) * env.red) and get_aircraft_available(env, i):
                    # both available have DLZ else add 0 #
                    dlz_list = aircraft_i["attack_zone_list"][enemy_id]
                    sub_ret_native.append(env.normalize(dlz_list["Raero"]))
                    sub_ret_native.append(env.normalize(dlz_list["Rmax"]))
                    sub_ret_native.append(env.normalize(dlz_list["Rmin"]))
                    sub_ret_native.append(env.normalize(dlz_list["Ropt"]))
                    sub_ret_native.append(env.normalize(dlz_list["Rpi"]))
                    sub_ret_native.append(env.normalize(dlz_list["Rtr"]))
                else:
                    sub_ret_native.append(0)
                    sub_ret_native.append(0)
                    sub_ret_native.append(0)
                    sub_ret_native.append(0)
                    sub_ret_native.append(0)
                    sub_ret_native.append(0)

            # add attitude angle
            sub_ret_native.append(env.normalize(aircraft_i["attg_0"]))
            sub_ret_native.append(env.normalize(aircraft_i["attg_1"]))
            # add overload
            sub_ret_native.append(env.normalize(aircraft_i["n_y"]))
            # add out boarder time
            sub_ret_native.append(env.normalize(aircraft_i["out_of_border_time"]))
            sub_ret_native.append(env.normalize(aircraft_i["out_of_border_distance"]))

        # add agents' alive info for native
        for i in list:
            aircraft_id = (i + env.red * k) % (env.red + env.blue)
            sub_ret_native.append(env.normalize(env.state_interface["AMS"][aircraft_id]["alive"]))

        # add relative obs to state_native todo add this to native
        temp = []
        for j in range(env.blue + env.red):
            for t in range(j + 1, env.red + env.blue):
                temp_2 = []
                env.to_list(temp_2, env.state_interface["AMS"][(k * env.red + list[j]) % (env.red + env.blue)][
                    "relative_observation"][(k * env.red + list[t]) % (env.red + env.blue)])
                temp_2 = copy.deepcopy(temp_2)
                if not (get_aircraft_available(env, (k * env.red + list[j]) % (env.red + env.blue)) and get_aircraft_available(env, (k * env.red + list[t]) % (env.red + env.blue))):
                    for item in temp_2:
                        item["value"] = item["min"]
                for item in temp_2:
                    temp.append(item)
        for item in temp:
            sub_ret_native.append(env.normalize(item))

        ret_native.append(sub_ret_native)

        # add token #
        sub_ret_token = []
        relative_obs_array = []
        # # not add friend #
        # for i in range(k * env.red, env.red + k * env.blue):
        #     for j in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
        #         relative_obs_array.append([i, j])
        # add friend #
        for aircraft_id in list[: env.red]:
            i = (aircraft_id + k * env.red) % (env.red + env.blue)
            i_friend = env.red - 1 - i if i < env.red else env.red * 2 + env.blue - i - 1
            relative_obs_array.append([i, i_friend])
            for j in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
                relative_obs_array.append([i, j])
        # print(relative_obs_array)
        for array in relative_obs_array:
            # ret_i = get_token_for_attention(env, array[0], array[1])
            if zerolize_invalid_token:
                aircraft_0_available = get_aircraft_available(env, array[0])
                aircraft_1_available = get_aircraft_available(env, array[1])

                if aircraft_0_available and aircraft_1_available:   # both alive, have valid token else token will all be 0
                    ret_i = get_token_for_attention_brief(env, array[0], array[1])
                    sub_ret_token.append(ret_i)
                else:  # one of this token player dies
                    ret_i = [0] * len(get_token_for_attention_brief(env, array[0], array[1]))
                    sub_ret_token.append(ret_i)

            else:
                ret_i = get_token_for_attention_brief(env, array[0], array[1])
                sub_ret_token.append(ret_i)

        ret_token.append(sub_ret_token)

    # # todo change here #
    # if not last_time_maneuver:
    #     pass
    # else:
    #     for i in range(len(last_time_maneuver)):
    #         ret_native[i] += last_time_maneuver[i][0]
    #         ret_native[i] += last_time_maneuver[i][1]
    #
    ret = [ret_native, ret_token]
    return ret


def get_kteam_aircraft_state_for_attention_refactor(env: BattleSpace, k: int, zerolize_invalid_token=True, last_time_maneuver_one_hot=None):
    id_list = []
    origin_list = [i for i in range(env.red + env.blue)]
    for i in range(env.red):
        id_list.append(swap(origin_list, 0, i))

    ret_native = []
    ret_token = []

    for list in id_list:
        sub_ret_native = []
        # first for native observations #
        for aircraft_id in list[: env.red]:
            i = (aircraft_id + k * env.red) % (env.red + env.blue)
            aircraft_i = env.state_interface["AMS"][i]
            aircraft_available = get_aircraft_available(env, i)

            sub_ret_native.append(env.normalize(aircraft_i["AAM_remain"]) if aircraft_available else 0.0)
            # sub_ret_native.append(env.normalize(aircraft_i["FCR_locked"]))
            sub_ret_native.append(env.normalize(aircraft_i["RWR_nailed"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["RWR_spiked"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["TAS"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["h_dot"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["residual_chi"]) if aircraft_available else 0.0)
            sub_ret_native.append(env.normalize(aircraft_i["Xg_2"]) if aircraft_available else 0.0)
            for j in aircraft_i["FCSModel"]:
                sub_ret_native.append(env.normalize(j["FCS_available"]) if aircraft_available else 0.0)
            for j in aircraft_i["RWRModel"]:
                sub_ret_native.append(env.normalize(j["RWR_fetched"]) if aircraft_available else 0.0)
            for j in aircraft_i["RadarModel"]:
                sub_ret_native.append(env.normalize(j["FCR_locked"]) if aircraft_available else 0.0)
            # for m in aircraft_i["SMS"]:
            #     for j in m["FCSGuide"]:
            #         sub_ret_native.append(env.normalize(j["fcs_guide_info"]))
            # for m in aircraft_i["SMS"]:
            #     for j in m["RadarGuide"]:
            #         sub_ret_native.append(env.normalize(j["radar_guide_info"]))
            # for m in aircraft_i["SMS"]:
            #     sub_ret_native.append(env.normalize(m["state"]))
            #     sub_ret_native.append(env.normalize(m["target_index"]))
            #     sub_ret_native.append(1.0 - env.normalize(m["TGO"]))
            #     # add msl relative states #
            #     if int(m["state"]["value"] + 0.1) == 2 or int(m["state"]["value"] + 0.1) == 3:  # missile flying
            #         sub_ret_native.append(env.normalize(m["r_dot_m"]))
            #         sub_ret_native.append(env.normalize(m["TAS_m"]))
            #         sub_ret_native.append(env.normalize(m["TA_m"]))
            #         sub_ret_native.append(env.normalize(m["AO_m"]))
            #     else:
            #         sub_ret_native.append(0)  # missle mounting or escaping #
            #         sub_ret_native.append(0)
            #         sub_ret_native.append(0)
            #         sub_ret_native.append(0)

            # # add attack zone list
            # for enemy_id in range(len(aircraft_i["attack_zone_list"])):
            #     if get_aircraft_available(env, enemy_id + (1 - k) * env.red) and get_aircraft_available(env, i):
            #         # both available have DLZ else add 0 #
            #         dlz_list = aircraft_i["attack_zone_list"][enemy_id]
            #         # sub_ret_native.append(env.normalize(dlz_list["Raero"]))
            #         sub_ret_native.append(env.normalize(dlz_list["Rmax"]))
            #         # sub_ret_native.append(env.normalize(dlz_list["Rmin"]))
            #         # sub_ret_native.append(env.normalize(dlz_list["Ropt"]))
            #         sub_ret_native.append(env.normalize(dlz_list["Rpi"]))
            #         sub_ret_native.append(env.normalize(dlz_list["Rtr"]))
            #     else:
            #         # sub_ret_native.append(0)
            #         # sub_ret_native.append(0)
            #         # sub_ret_native.append(0)
            #         sub_ret_native.append(0)
            #         sub_ret_native.append(0)
            #         sub_ret_native.append(0)

            # add attitude angle
            sub_ret_native.append(env.normalize(aircraft_i["attg_0"]))
            sub_ret_native.append(env.normalize(aircraft_i["attg_1"]))
            # add overload
            sub_ret_native.append(env.normalize(aircraft_i["n_y"]))
            # add out boarder time
            sub_ret_native.append(env.normalize(aircraft_i["out_of_border_time"]))
            sub_ret_native.append(env.normalize(aircraft_i["out_of_border_distance"]))

        # add agents' alive info for native
        for i in list:
            aircraft_id = (i + env.red * k) % (env.red + env.blue)
            sub_ret_native.append(env.normalize(env.state_interface["AMS"][aircraft_id]["alive"]))

        # add relative obs to state_native todo add this to native
        temp = []
        for j in range(env.blue + env.red):
            for t in range(j + 1, env.red + env.blue):
                temp_2 = []
                env.to_list(temp_2, env.state_interface["AMS"][(k * env.red + list[j]) % (env.red + env.blue)][
                    "relative_observation"][(k * env.red + list[t]) % (env.red + env.blue)])
                temp_2 = copy.deepcopy(temp_2)
                if not (get_aircraft_available(env, (k * env.red + list[j]) % (
                        env.red + env.blue)) and get_aircraft_available(env, (k * env.red + list[t]) % (
                        env.red + env.blue))):
                    for item in temp_2:
                        item["value"] = item["min"]
                for item in temp_2:
                    temp.append(item)
        for item in temp:
            sub_ret_native.append(env.normalize(item))

        ret_native.append(sub_ret_native)

        # add token #
        sub_ret_token = []
        relative_obs_array = []
        # # not add friend #
        # for i in range(k * env.red, env.red + k * env.blue):
        #     for j in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
        #         relative_obs_array.append([i, j])
        # add friend #
        for aircraft_id in list[: env.red]:
            i = (aircraft_id + k * env.red) % (env.red + env.blue)
            i_friend = env.red - 1 - i if i < env.red else env.red * 2 + env.blue - i - 1
            relative_obs_array.append([i, i_friend])
            for j in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
                relative_obs_array.append([i, j])
        # print(relative_obs_array)
        for array in relative_obs_array:
            # ret_i = get_token_for_attention(env, array[0], array[1])
            if zerolize_invalid_token:
                aircraft_0_available = get_aircraft_available(env, array[0])
                aircraft_1_available = get_aircraft_available(env, array[1])

                if aircraft_0_available and aircraft_1_available:  # both alive, have valid token else token will all be 0
                    ret_i = get_token_for_attention_brief(env, array[0], array[1])
                    sub_ret_token.append(ret_i)
                else:  # one of this token player dies
                    ret_i = [0] * len(get_token_for_attention_brief(env, array[0], array[1]))
                    sub_ret_token.append(ret_i)

            else:
                ret_i = get_token_for_attention_brief(env, array[0], array[1])
                sub_ret_token.append(ret_i)

        ret_token.append(sub_ret_token)

    # todo change here #
    if not last_time_maneuver_one_hot:
        pass
    else:
        for i in range(len(last_time_maneuver_one_hot)):
            ret_native[i] += last_time_maneuver_one_hot[i][0]
            ret_native[i] += last_time_maneuver_one_hot[i][1]

    ret = [ret_native, ret_token]
    return ret


def get_token_for_attention_brief(env, i: int, j: int):
    # input i for self aircraft id, j for opponent id, considering 2 vs 2 #
    i_friend = (env.red - i - 1) if i < env.red else (2 * env.red + env.blue - i - 1)
    relative_i = i - env.red if i >= env.red else i
    relative_j = j - env.red if j >= env.red else j

    ret_token = []

    # step 1: add alive
    ret_token.append(env.normalize(env.state_interface["AMS"][i]["alive"]))
    ret_token.append(env.normalize(env.state_interface["AMS"][j]["alive"]))

    ret_token.append(env.normalize(env.state_interface["AMS"][i_friend]["alive"]))  # i_friend may equals j

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

    # step 5: add tgo
    # self tgo
    # for m in env.state_interface["AMS"][i]["SMS"]:
    #     if teammate:
    #         ret_token.append(0.0)  # teammate
    #     else:
    #         if int(m["state"]["value"] + 0.1) == 2 or int(m["state"]["value"] + 0.1) == 3:  # flying and seeking:
    #             if int(m["target_index"]["value"]) == relative_j:
    #                 # if missile firing this target, add tgo
    #                 ret_token.append(1.0 - env.normalize(m["TGO"]))  # valid tgo #
    #             else:
    #                 ret_token.append(0.0)
    #         else:
    #             # missile mount or escape
    #             ret_token.append(0.0)  # not considering this missile
    # # friend tgo
    # for m in env.state_interface["AMS"][i_friend]["SMS"]:
    #     if teammate:
    #         ret_token.append(0.0)
    #     else:
    #         if int(m["state"]["value"] + 0.1) == 2 or int(m["state"]["value"] + 0.1) == 3:  # flying and seeking:
    #             if int(m["target_index"]["value"]) == relative_j:
    #                 # if friend missile firing this target, add tgo
    #                 ret_token.append(1.0 - env.normalize(m["TGO"]))
    #             else:
    #                 ret_token.append(0.0)
    #         else:
    #             # missile mount or escape
    #             ret_token.append(0.0)  # not considering this missile
    # # target tgo
    # for m in env.state_interface["AMS"][j]["SMS"]:
    #     if teammate:
    #         ret_token.append(0.0)
    #     else:
    #         if int(m["state"]["value"] + 0.1) == 2 or int(m["state"]["value"] + 0.1) == 3:  # flying and seeking:
    #             if int(m["target_index"]["value"]) == relative_i:
    #                 # if friend missile firing this target, add tgo
    #                 ret_token.append(1.0 - env.normalize(m["TGO"]))
    #             else:
    #                 ret_token.append(0.0)
    #         else:
    #             # missile mount or escape
    #             ret_token.append(0.0)  # not considering this missile

    # step 5: add DLZ
    if teammate:  # for teammate, no valid DLZ info
        for _ in range(len(env.state_interface["AMS"][i]["attack_zone_list"][0]) - 1):  # - 1 for ASE circle
            ret_token.append(0.0)
    else:
        DLZ_list = env.state_interface["AMS"][i]["attack_zone_list"][relative_j]
        ret_token.append(env.normalize(DLZ_list["Raero"]))
        ret_token.append(env.normalize(DLZ_list["Rmax"]))
        ret_token.append(env.normalize(DLZ_list["Rmin"]))
        ret_token.append(env.normalize(DLZ_list["Ropt"]))
        ret_token.append(env.normalize(DLZ_list["Rpi"]))
        ret_token.append(env.normalize(DLZ_list["Rtr"]))

    return ret_token


def get_token_for_LTR_attention_brief(env: BattleSpace, i: int, j: int, last_target, last_maneuver, maneuver_remain_step):
    # use origin token for attention #
    ret_token = get_token_for_attention_brief(env, i, j)
    # add LTR part #
    # team_aircraft_num = len(last_target)
    relative_i = i if i < env.red else i - env.red
    cur_i_target = last_target[relative_i]
    cur_i_maneuver = last_maneuver[relative_i]
    cur_i_step = maneuver_remain_step[relative_i]

    this_target = False
    if abs(i - j) < env.red:
        if i < env.red:
            # i red, j red
            if cur_i_target == j:
                this_target = True
        elif i >= env.red:
            # i blue, j blue
            if cur_i_target == (j - env.red):
                this_target = True
    elif abs(i - j) >= env.red:
        if i < env.red:
            # i red j blue
            if cur_i_target == j:
                this_target = True
        elif i >= env.red:
            # i blue j red
            if cur_i_target == (j + env.red):
                this_target = True

    if this_target:
        ret_token.append(1)
        ret_token.append(cur_i_step / Config.LTR.max_time_step)
    else:
        ret_token.append(0)
        ret_token.append(0)

    for maneuver_id in range(len(env.action_interface["AMS"][0]["DiscreteManeuver"]["action_bfm_id"]["mask"])):
        if this_target:
            # focusing on this target
            if maneuver_id == cur_i_maneuver:
                ret_token.append(1)
            else:
                ret_token.append(0)
        else:
            ret_token.append(0)

    return ret_token


def get_kteam_global_ground_truth_state(env: BattleSpace, k: int):
    ret_global = []
    id_list = []
    origin_list = [i for i in range(env.red + env.blue)]
    for i in range(env.red):
        id_list.append(swap(origin_list, 0, i))

    for list in id_list:
        sub_ret_global = []
        # add aircraft native state #
        for aircraft_id in list:
            aircraft_absolute_id = (aircraft_id + k * env.red) % (env.red + env.blue)
            ret_native = get_ith_aircraft_ground_truth_native_state(env, aircraft_absolute_id)
            aircraft_available = get_aircraft_available(env, aircraft_absolute_id)
            native_available = 1.0 if aircraft_available else 0.0
            for item in ret_native:
                sub_ret_global.append(item * native_available)

        # add aircraft relative state #
        for id_i in range(env.blue + env.red):
            for id_j in range(id_i + 1, env.red + env.blue):
                i = (k * env.red + list[id_i]) % (env.red + env.blue)
                j = (k * env.red + list[id_j]) % (env.red + env.blue)
                ret_relative = get_aircraft_ground_truth_relative_state(env, i, j)
                if get_aircraft_available(env, i) and get_aircraft_available(env, j):
                    relative_valid = 1.0
                else:
                    relative_valid = 0.0
                for item in ret_relative:
                    sub_ret_global.append(item * relative_valid)

        ret_global.append(sub_ret_global)

    return ret_global


def get_ith_aircraft_ground_truth_native_state(env: BattleSpace, i):
    ret_truth = []
    aircraft_i = env.state_interface["AMS"][i]
    k = 0 if i < env.red else 1  # team id
    # 1. position relative state #
    ret_truth.append(env.normalize(aircraft_i["TAS"]))
    ret_truth.append(env.normalize(aircraft_i["Vg_0"]))
    ret_truth.append(env.normalize(aircraft_i["Vg_1"]))
    ret_truth.append(env.normalize(aircraft_i["Vg_2"]))
    ret_truth.append(env.normalize(aircraft_i["Vg_0_est"]))
    ret_truth.append(env.normalize(aircraft_i["Vg_1_est"]))
    ret_truth.append(env.normalize(aircraft_i["Vg_2_est"]))
    ret_truth.append(env.normalize(aircraft_i["Xg_0"]))
    ret_truth.append(env.normalize(aircraft_i["Xg_1"]))
    ret_truth.append(env.normalize(aircraft_i["Xg_2"]))
    ret_truth.append(env.normalize(aircraft_i["Xg_0_est"]))
    ret_truth.append(env.normalize(aircraft_i["Xg_1_est"]))
    ret_truth.append(env.normalize(aircraft_i["Xg_2_est"]))
    ret_truth.append(env.normalize(aircraft_i["attg_0"]))
    ret_truth.append(env.normalize(aircraft_i["attg_1"]))
    ret_truth.append(env.normalize(aircraft_i["attg_2"]))
    # 2. sensor relative state #
    for bandit_relative_id in range(len(aircraft_i["RWRModel"])):
        bandit_absolute_id = bandit_relative_id + (1 - k) * env.red
        if int(env.state_interface["AMS"][bandit_absolute_id]["alive"]["value"] + 0.1) == 1:  # enemy alive
            ret_truth.append(env.normalize_with_bias(aircraft_i["RWRModel"][bandit_relative_id]["RWR_fetched"]))
        else:
            ret_truth.append(0.0)

    ret_truth.append(env.normalize(aircraft_i["RWR_nailed"]))
    ret_truth.append(env.normalize(aircraft_i["RWR_spiked"]))

    for bandit_relative_id in range(len(aircraft_i["RadarModel"])):
        bandit_absolute_id = bandit_relative_id + (1 - k) * env.red
        if int(env.state_interface["AMS"][bandit_absolute_id]["alive"]["value"] + 0.1) == 1:  # enemy alive
            ret_truth.append(env.normalize_with_bias(aircraft_i["RadarModel"][bandit_relative_id]["FCR_locked"]))
        else:
            ret_truth.append(0.0)
    # 3. SMS relative state #
    # 3.1 AAM_remain #
    ret_truth.append(env.normalize(aircraft_i["AAM_remain"]))
    # 3.2 AAM_relative obs #
    for msl in aircraft_i["SMS"]:
        ret_truth.append(env.normalize(msl["state"]))
        msl_state = int(msl["state"]["value"] + 0.1)
        msl_relative_target = int(msl["target_index"]["value"] + 0.1)
        if msl_state == 2 or msl_state == 3:
            msl_absolute_target = msl_relative_target + (1 - k) * env.red
            if int(env.state_interface["AMS"][msl_absolute_target]["alive"]["value"] + 0.1) == 1:
                msl_valid = 1.0
            else:
                msl_valid = 0.0
        else:
            msl_valid = 0.0
        # msl flying and msl target alive, add msl relative obs, else and 0 #
        ret_truth.append(env.normalize(msl["AO_m"]) * msl_valid)
        ret_truth.append(env.normalize(msl["TA_m"]) * msl_valid)
        ret_truth.append(env.normalize(msl["TAS_m"]) * msl_valid)
        ret_truth.append(env.normalize(msl["Xg_m_0"]) * msl_valid)
        ret_truth.append(env.normalize(msl["Xg_m_1"]) * msl_valid)
        ret_truth.append(env.normalize(msl["Xg_m_2"]) * msl_valid)
        ret_truth.append(env.normalize(msl["r_dot_m"]) * msl_valid)
        ret_truth.append(env.normalize(msl["target_index"]) * msl_valid)
    # 3.3 DLZ info #
    for bandit_relative_id in range(len(aircraft_i["attack_zone_list"])):
        bandit_absolute_id = bandit_relative_id + (1 - k) * env.red
        if int(env.state_interface["AMS"][bandit_absolute_id]["alive"]["value"] + 0.1) == 1:  # enemy alive
            DLZ_list = aircraft_i["attack_zone_list"][bandit_relative_id]
            ret_truth.append(env.normalize(DLZ_list["Raero"]))
            ret_truth.append(env.normalize(DLZ_list["Rmax"]))
            ret_truth.append(env.normalize(DLZ_list["Rmin"]))
            ret_truth.append(env.normalize(DLZ_list["Ropt"]))
            ret_truth.append(env.normalize(DLZ_list["Rpi"]))
            ret_truth.append(env.normalize(DLZ_list["Rtr"]))
        else:
            ret_truth.append(0.0)
            ret_truth.append(0.0)
            ret_truth.append(0.0)
            ret_truth.append(0.0)
            ret_truth.append(0.0)
            ret_truth.append(0.0)
    # 4. aircraft native state #
    ret_truth.append(env.normalize(aircraft_i["alive"]))
    ret_truth.append(env.normalize(aircraft_i["n_y"]))
    ret_truth.append(env.normalize(aircraft_i["out_of_border_time"]))
    ret_truth.append(env.normalize(aircraft_i["residual_chi"]))

    return ret_truth


def get_aircraft_ground_truth_relative_state(env: BattleSpace, i, j):
    # this function for adding relative state which will cause redundancy if write in native state(like relative obs between aircrafts)
    ret_relative = []

    temp = []
    env.to_list(temp, env.state_interface["AMS"][i]["relative_observation"][j])
    for item in temp:
        ret_relative.append(env.normalize(item))

    return ret_relative


def get_msl_token(env: BattleSpace, i_aircraft, i_th_msl):
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
    ret_msl.append(env.normalize(msl["TAS_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TA_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TGO"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_0"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_1"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_2"]) * token_valid)
    ret_msl.append(env.normalize(msl["r_dot_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["target_index"]) * token_valid)

    return ret_msl


def get_kteam_msl_tokens(env: BattleSpace, k: int):
    id_list = []
    origin_list = [i for i in range(env.red + env.blue)]
    for i in range(env.red):
        id_list.append(swap(origin_list, 0, i))

    msl_token = []
    for list in id_list:
        sub_msl_token = []
        for aircraft_id in list[: env.red]:
            i = (aircraft_id + k * env.red) % (env.red + env.blue)
            aircraft = env.state_interface["AMS"][i]
            for msl_id in range(len(aircraft["SMS"])):
                sub_msl_token.append(get_msl_token(env, i, msl_id))
        msl_token.append(sub_msl_token)
    return msl_token


def swap(input_list, i, j):
    list = copy.deepcopy(input_list)
    temp = list[i]
    list[i] = list[j]
    list[j] = temp
    return list


def get_kteam_ids_one_hot_state(env: BattleSpace, team_aircraft_num: int):
    ids_one_hot_state = []
    for id in range(team_aircraft_num):
        ids_one_hot_state.append(index_to_one_hot(id, team_aircraft_num))
    return ids_one_hot_state


def get_self_msl_token(env: BattleSpace, i_aircraft, i_th_msl):
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
    ret_msl.append(env.normalize(msl["TAS_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TA_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TGO"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_0"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_1"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_2"]) * token_valid)
    ret_msl.append(env.normalize(msl["r_dot_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["target_index"]) * token_valid)

    return ret_msl


def get_bandit_msl_token(env: BattleSpace, i_aircraft, i_th_msl, self_id):
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
    ret_msl.append(env.normalize(msl["TAS_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TA_m"]) * token_valid)
    ret_msl.append(env.normalize(msl["TGO"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_0"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_1"]) * token_valid)
    ret_msl.append(env.normalize(msl["Xg_m_2"]) * token_valid)
    ret_msl.append(env.normalize(msl["r_dot_m"]) * token_valid)

    return ret_msl


def get_self_kteam_msl_tokens(env: BattleSpace, k: int):
    id_list = []
    origin_list = [i for i in range(env.red + env.blue)]
    for i in range(env.red):
        id_list.append(swap(origin_list, 0, i))

    msl_token = []
    for list in id_list:
        sub_msl_token = []
        for aircraft_id in list[:2]:
            i = (aircraft_id + k * env.red) % (env.red + env.blue)
            aircraft = env.state_interface["AMS"][i]
            for msl_id in range(len(aircraft["SMS"])):
                sub_msl_token.append(get_self_msl_token(env, i, msl_id))
        msl_token.append(sub_msl_token)
    return msl_token


def get_bandit_kteam_msl_tokens(env: BattleSpace, k: int):
    id_list = []
    origin_list = [i for i in range(env.red + env.blue)]
    for i in range(env.red):
        id_list.append(swap(origin_list, 0, i))

    msl_token = []
    for list in id_list:
        sub_msl_token = []
        self_id = (list[0] + k * env.red) % (env.red + env.blue)
        bandit_num = env.red if k else env.blue
        for aircraft_id in list[-bandit_num:]:
            i = (aircraft_id + k * env.red) % (env.red + env.blue)
            aircraft = env.state_interface["AMS"][i]
            for msl_id in range(len(aircraft["SMS"])):
                sub_msl_token.append(get_bandit_msl_token(env, i, msl_id, self_id))
        msl_token.append(sub_msl_token)
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
