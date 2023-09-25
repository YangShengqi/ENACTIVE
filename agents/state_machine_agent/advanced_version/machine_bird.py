from agents.state_machine_agent.YSQ.advanced_version.machine_config import semantic_maneuver_default, top_tactics, macro_tactics, MagicNumber
from agents.state_machine_agent.YSQ.advanced_version.combo import Abort, Evade, Intercept, Format, Banzai
from state_method import state_method_independ
from reward_method import reward_method_independ, reward_shaping
from framwork.agent_base import AgentBase
from train.config import Config
from utils.math import index_to_one_hot

from math import pi, degrees, acos
from copy import deepcopy
import numpy as np


class MachineBird(AgentBase):
    def __init__(self, reward_hyperparam_dict):
        self.side = None
        self.interval = 1
        self.maneuver_model = ["F22bot", "F22bot"]
        self.combo_executing = [None, None]
        self.threat_level = ["none", "none"]
        self.bandit_launch_range = [[], []]
        self.bandit_launch_direction = [[], []]
        self.png_target_info = [[], []]
        self.bandit_msl_valid_flag = [[], []]
        self.default_target = []
        self.default_clockwise = []
        self.combo = [{
            "defense": {
                "abort": Abort(),
                "evade": Evade()
            },
            "offense": {
                "intercept": Intercept(),
                "banzai": Banzai()
            },
            "others": {
                "format": Format()
            }
        },
        {
            "defense": {
                "abort": Abort(),
                "evade": Evade()
            },
            "offense": {
                "intercept": Intercept(),
                "banzai": Banzai()
            },
            "others": {
                "format": Format()
            }
        }]

        self.batchs = dict(state_global=[], state_native=[], state_id=[],
                           state_token=[], self_msl_token=[], bandit_msl_token=[],
                           hor=[], ver=[], shot=[], target=[], v_c=[], nn_c=[], hor_one_hots=[],
                           ver_one_hots=[], mask=[],
                           hor_masks=[], ver_masks=[], target_masks=[], shot_masks=[], mask_solo_done=[],
                           next_state_global=[], next_state_native=[], next_state_token=[], team_rewards=[],
                           shape_rewards=[], solo_rewards=[], other_mean_reward=[], final_rewards=[],
                           current_step_available=[], type=[], shot_data=[],
                           log=None)

        self.dones = None
        self.current_episode_solo_reward = None
        self.current_episode_other_mean_reward = None
        self.rewards_hyperparam_dict = reward_hyperparam_dict

    def after_reset(self, env, side):
        self.dones = [False for _ in range(env.red + env.blue)]
        self.current_episode_solo_reward = [0, 0]
        self.current_episode_other_mean_reward = [0, 0]

        if side == "red":
            self.side = 0
        elif side == "blue":
            self.side = 1

        self.combo_executing = [None, None]
        self.threat_level = ["none", "none"]
        self.bandit_launch_range = [[], []]
        self.bandit_launch_direction = [[], []]
        self.png_target_info = [[], []]
        self.bandit_msl_valid_flag = [[], []]
        self.default_target = []
        self.default_clockwise = []
        for i in range(2):
            self.combo[i]["defense"]["abort"].reset()
            self.combo[i]["defense"]["evade"].reset()
            self.combo[i]["offense"]["intercept"].reset()
            self.combo[i]["offense"]["banzai"].reset()
            self.combo[i]["others"]["format"].reset()

        team0_y = env.state_interface["AMS"][self.side * env.red]["Xg_1"]["value"]
        team1_y = env.state_interface["AMS"][self.side * env.red + 1]["Xg_1"]["value"]
        bandit0_y = env.state_interface["AMS"][(1 - self.side) * env.red]["Xg_1"]["value"]
        bandit1_y = env.state_interface["AMS"][(1 - self.side) * env.red + 1]["Xg_1"]["value"]
        if not ((team0_y <= team1_y) ^ (bandit0_y <= bandit1_y)):
            self.default_target.append((1 - self.side) * env.red)
            self.default_target.append((1 - self.side) * env.red + 1)
        else:
            self.default_target.append((1 - self.side) * env.red + 1)
            self.default_target.append((1 - self.side) * env.red)

        team0_x = env.state_interface["AMS"][self.side * env.red]["Xg_0"]["value"]
        if team0_x >= 0:
            if team0_y >= team1_y:
                self.default_clockwise.append(-1)
                self.default_clockwise.append(1)
            else:
                self.default_clockwise.append(1)
                self.default_clockwise.append(-1)
        else:
            if team0_y >= team1_y:
                self.default_clockwise.append(1)
                self.default_clockwise.append(-1)
            else:
                self.default_clockwise.append(-1)
                self.default_clockwise.append(1)

        for i in range(2):
            for m in range(4):
                self.bandit_msl_valid_flag[i].append(True)

    """功能函数"""
    def check_self_missile_work(self, env, self_id) -> bool:
        missile_set = env.state_interface["AMS"][self_id]["SMS"]
        for msl in missile_set:
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
                return True
        return False

    def check_team_missile_work(self, env) -> bool:
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            missile_set = env.state_interface["AMS"][i]["SMS"]
            for msl in missile_set:
                if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
                    return True
        return False

    def get_self_missile_work_target(self, env, self_id) -> list:
        missile_target_list = []
        missile_set = env.state_interface["AMS"][self_id]["SMS"]
        for msl in missile_set:
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
                missile_target_list.append(int(msl["target_index"]["value"]) + (1 - self.side) * env.red)
        out = list(set(missile_target_list))
        return out  # absolute index

    def get_team_missile_work_target(self, env) -> list:
        missile_target_list = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            missile_set = env.state_interface["AMS"][i]["SMS"]
            for msl in missile_set:
                if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
                    missile_target_list.append(int(msl["target_index"]["value"]) + (1 - self.side) * env.red)
        out = list(set(missile_target_list))
        return out  # absolute index

    def check_self_missile_fuck_some_target(self, env, self_id, target_id):
        fuck_flag = False
        flying_time = 0
        r_dot = 100

        missile_set = env.state_interface["AMS"][self_id]["SMS"]
        for msl in missile_set:
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
                cur_target_id = int(msl["target_index"]["value"]) + (1 - self.side) * env.red
                if cur_target_id == target_id:
                    fuck_flag = True
                    r_dot = abs(msl["r_dot_m"]["value"])
                    flying_time = msl["flying_time"]["value"]

        return fuck_flag, flying_time, r_dot

    def check_team_missile_fuck_some_target(self, env, target_id):
        fuck_flag = False
        flying_time = 0
        r_dot = 100

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            missile_set = env.state_interface["AMS"][i]["SMS"]
            for msl in missile_set:
                if msl["state"]["value"] == 2 or msl["state"]["value"] == 3:
                    cur_target_id = int(msl["target_index"]["value"]) + (1 - self.side) * env.red
                    if cur_target_id == target_id:
                        fuck_flag = True
                        r_dot = abs(msl["r_dot_m"]["value"])
                        flying_time = msl["flying_time"]["value"]

        return fuck_flag, flying_time, r_dot

    def check_target_missile_fuck_me(self, env, self_id, target_id) -> bool:
        flag = False
        missile_set = env.state_interface["AMS"][target_id]["SMS"]
        for msl in missile_set:
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
                if int(msl["target_index"]["value"] + 0.1) == self_id - self.side * env.red:
                    flag = True
        return flag

    def check_self_missile_stop(self, env, self_id, msl_id) -> bool:
        flag = False
        missile = env.state_interface["AMS"][self_id]["SMS"][msl_id]
        state = int(missile["state"]["value"] + 0.1)
        if state == 4:
            flag = True
        return flag

    def check_self_missile_stop_or_low_energy(self, env, low_ennergy_tas, self_id, msl_id) -> bool:
        flag = False
        missile = env.state_interface["AMS"][self_id]["SMS"][msl_id]
        state = int(missile["state"]["value"] + 0.1)
        tas = missile["TAS_m"]["value"]
        flying_time = missile["flying_time"]["value"]
        if state == 4 or (flying_time > 10 and tas <= low_ennergy_tas):
            flag = True
        return flag

    def get_threat_missile_range(self, env, self_id, bandit_id, msl_id):
        self_x = env.state_interface["AMS"][self_id]["Xg_0"]["value"]
        self_y = env.state_interface["AMS"][self_id]["Xg_1"]["value"]
        self_z = env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        msl_x = env.state_interface["AMS"][bandit_id]["SMS"][msl_id]["Xg_m_0"]["value"]
        msl_y = env.state_interface["AMS"][bandit_id]["SMS"][msl_id]["Xg_m_1"]["value"]
        msl_z = env.state_interface["AMS"][bandit_id]["SMS"][msl_id]["Xg_m_2"]["value"]
        msl_range = ((self_x - msl_x)**2 + (self_y - msl_y)**2 + (self_z - msl_z)**2)**0.5
        return msl_range

    def check_missile_threat(self, env, self_id):
        threat_flag = False
        min_msl_tgo = env.state_interface["AMS"][0]["SMS"][0]["TGO"]["max"]
        msl_r = 200000
        launch_range = 60000
        launch_direction = 0
        flying_time = 0
        msl_tas = 800
        delta_r = 0
        threat_source = (1 - self.side) * env.red  # TODO
        threat_msl_id = 0
        self_x = env.state_interface["AMS"][self_id]["Xg_0"]["value"]
        self_y = env.state_interface["AMS"][self_id]["Xg_1"]["value"]
        self_z = env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            bandit_aircraft = env.state_interface["AMS"][j]
            for msl_i, msl in enumerate(bandit_aircraft["SMS"]):
                if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying #
                    if int(msl["target_index"]["value"] + 0.1) == self_id - self.side * env.red:
                        threat_flag = True
                        if msl["TGO"]["value"] <= min_msl_tgo:
                            min_msl_tgo = msl["TGO"]["value"]
                            threat_source = j
                            threat_msl_id = msl_i
                            msl_r = self.get_threat_missile_range(env, self_id, j, msl_i)
                            launch_range = self.get_some_missile_launch_range(env, j, msl_i)
                            launch_direction = self.get_some_missile_launch_direction(env, j, msl_i)
                            flying_time = msl["flying_time"]["value"]
                            msl_tas = msl["TAS_m"]["value"]
                            png_target_info = self.get_some_missile_png_target_info(env, j, msl_i)
                            delta_r = ((self_x - png_target_info["x"]) ** 2 + (self_y - png_target_info["y"]) ** 2 + (
                                    self_z - png_target_info["z"]) ** 2) ** 0.5
        if delta_r > MagicNumber.min_png_delta_range and msl_tas < 650:
            self.bandit_msl_valid_flag[threat_source if self.side else threat_source - env.red][threat_msl_id] = False

        fuck_flag, _, _ = self.check_self_missile_fuck_some_target(env, self_id, threat_source)
        if threat_flag:
            if launch_range > 70000:
                if fuck_flag:
                    if flying_time <= 50:
                        threat_level = "medium"
                        advice_tactic = "snake_50_dive_25"
                    else:
                        if msl_r < 21000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "snake_50_dive_25"
                    if msl_r < 8000:
                        threat_level = "danger"
                        advice_tactic = "circle"
                else:
                    if flying_time <= 45:
                        threat_level = "medium"
                        advice_tactic = "snake_50"
                    else:
                        if msl_r < 22000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "snake_50"
                    if msl_r < 12000:
                        threat_level = "danger"
                        advice_tactic = "circle"
            elif 65000 < launch_range <= 70000:
                if fuck_flag:
                    if flying_time <= 40:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        if msl_r < 20000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "crank_50_dive_25"
                    if msl_r < 7000:
                        threat_level = "danger"
                        advice_tactic = "circle"
                else:
                    if flying_time <= 40:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        if msl_r < 23000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "crank_50_dive_25"
                    if msl_r < 13000:
                        threat_level = "danger"
                        advice_tactic = "circle"
            elif 60000 < launch_range <= 65000:
                if fuck_flag:
                    if flying_time <= 30:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        if msl_r < 22000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "crank_50_dive_25"
                    if msl_r < 7000:
                        threat_level = "danger"
                        advice_tactic = "circle"
                else:
                    if flying_time <= 30:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        if msl_r < 25000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "crank_50_dive_25"
                    if msl_r < 14000:
                        threat_level = "danger"
                        advice_tactic = "circle"
            elif 55000 < launch_range <= 60000:
                if fuck_flag:
                    if flying_time <= 25:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        if msl_r < 22000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "crank_50_dive_25"
                    if msl_r < 7500:
                        threat_level = "danger"
                        advice_tactic = "circle"
                else:
                    if flying_time <= 20:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        if msl_r < 25000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "crank_50_dive_25"
                    if msl_r < 15000:
                        threat_level = "danger"
                        advice_tactic = "circle"
            elif 50000 < launch_range <= 55000:
                if fuck_flag:
                    if flying_time <= 20:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        if msl_r < 25000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "crank_50_dive_25"
                    if msl_r < 9000:
                        threat_level = "danger"
                        advice_tactic = "circle"
                else:
                    if flying_time <= 20:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        if msl_r < 28000:
                            threat_level = "high"
                            advice_tactic = "notch_dive_25"
                        else:
                            threat_level = "medium"
                            advice_tactic = "crank_50_dive_25"
                    if msl_r < 15000:
                        threat_level = "danger"
                        advice_tactic = "circle"
            elif 45000 < launch_range <= 50000:
                if fuck_flag:
                    threat_level = "medium"
                    advice_tactic = "crank_50_dive_25"
                    if msl_r < 13000:
                        threat_level = "danger"
                        advice_tactic = "circle"
                else:
                    if flying_time <= 15:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        threat_level = "high"
                        advice_tactic = "notch_dive_25"
                    if msl_r < 18000:
                        threat_level = "danger"
                        advice_tactic = "circle"
            elif 40000 < launch_range <= 45000:
                if fuck_flag:
                    if flying_time <= 10:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        threat_level = "high"
                        advice_tactic = "notch_dive_25"
                    if msl_r < 13000:
                        threat_level = "danger"
                        advice_tactic = "circle"
                else:
                    if flying_time <= 10:
                        threat_level = "medium"
                        advice_tactic = "crank_50_dive_25"
                    else:
                        threat_level = "high"
                        advice_tactic = "notch_dive_25"
                    if msl_r < 20000:
                        threat_level = "danger"
                        advice_tactic = "circle"
            elif 30000 < launch_range <= 40000:
                if fuck_flag:
                    threat_level = "medium"
                    advice_tactic = "crank_50_dive_25"
                    if msl_r < 11000:
                        threat_level = "danger"
                        advice_tactic = "circle"
                else:
                    threat_level = "high"
                    advice_tactic = "notch_dive_25"
                    if msl_r < 20000:
                        threat_level = "danger"
                        advice_tactic = "circle"
            else:
                if fuck_flag:
                    threat_level = "medium"
                    advice_tactic = "crank_50_dive_25"
                else:
                    threat_level = "medium"
                    advice_tactic = "crank_50_dive_25"
                if msl_r < 11000:
                    threat_level = "danger"
                    advice_tactic = "circle"
            if not self.bandit_msl_valid_flag[threat_source if self.side else threat_source - env.red][threat_msl_id]:
                threat_level = "low"
                advice_tactic = "change_direction"
        else:
            threat_level = "none"
            advice_tactic = None

        return threat_flag, threat_level, advice_tactic, threat_source, launch_range, launch_direction  # threat_source: absolute index

    def check_escape_available(self, env, self_id):
        escape_flag = False
        rel_i = self_id - env.red if self.side else self_id
        threat_flag, threat_level, _, threat_source, _, _ = self.check_missile_threat(env, self_id)
        if threat_flag:
            near_bandit_i = self.threat_sort(env, self_id)[0]
            if int(env.state_interface["AMS"][near_bandit_i]["alive"]["value"] + 0.1) < 1:
                near_bandit_i = 1 - near_bandit_i if self.side else 5 - near_bandit_i
            near_r = env.state_interface["AMS"][self_id]["relative_observation"][near_bandit_i]["r"]["value"]
            near_rtr = env.state_interface["AMS"][near_bandit_i]["attack_zone_list"][rel_i]["Rtr"]["value"]
            escape_flag = (threat_level == "high" or threat_level == "danger") and (near_r > MagicNumber.min_escape_range)
        return escape_flag, threat_source

    def get_escape_available_list(self, env) -> list:
        escape_flag_list = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            escape_flag, _ = self.check_escape_available(env, i)
            escape_flag_list.append(escape_flag)
        return escape_flag_list

    def check_alive_num(self, env):
        team_alive_num = 0
        bandit_alive_num = 0
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            # missile = env.state_interface["AMS"][i]["SMS"]
            # work_missile_num = 0
            # for msl in missile["SMS"]:
            #     if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
            #         work_missile_num = work_missile_num + 1
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:  # or work_missile_num > 0:
                team_alive_num += 1
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            alive = env.state_interface["AMS"][j]["alive"]["value"]
            if int(alive + 0.1) == 1:
                bandit_alive_num += 1
        return team_alive_num, bandit_alive_num

    def get_only_alive_bandit_index(self, env):  # used for only one bandit alive condition
        only_alive_bandit_index = (1 - self.side) * env.red  # TODO
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            alive = env.state_interface["AMS"][j]["alive"]["value"]
            if int(alive + 0.1) == 1:
                only_alive_bandit_index = j

        return only_alive_bandit_index

    def range_threat_cal(self, env, i: int, j: int):
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

    def threat_sort(self, env, i: int) -> list:  # absolute index
        threat_i = []
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            threat_i.append(self.range_threat_cal(env, i, j))
        threat_i = np.array(threat_i)
        bandit_threat_absolute_idx = (np.argsort(-threat_i) + (1 - self.side) * env.red).tolist()

        return bandit_threat_absolute_idx

    def get_bandit_launch_range(self, env, bandit_id):
        rel_j = bandit_id if self.side else bandit_id - env.red
        msl_i = len(self.bandit_launch_range[rel_j])
        if msl_i < 4:
            msl = env.state_interface["AMS"][bandit_id]["SMS"][msl_i]
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
                rel_i = int(msl["target_index"]["value"] + 0.1)
                target_id = rel_i + env.red if self.side else rel_i
                r = env.state_interface["AMS"][bandit_id]["relative_observation"][target_id]["r"]["value"]
                self.bandit_launch_range[rel_j].append(r)
        else:
            pass

    def get_some_missile_launch_range(self, env, threat_source_id, threat_msl_id):
        rel_j = threat_source_id if self.side else threat_source_id - env.red
        if (threat_msl_id + 1) <= len(self.bandit_launch_range[rel_j]):
            threat_msl_launch_r = self.bandit_launch_range[rel_j][threat_msl_id]
        else:
            threat_msl_launch_r = MagicNumber.threshold_threat_msl_launch_range - 1

        return threat_msl_launch_r

    def get_bandit_launch_direction(self, env, bandit_id):
        rel_j = bandit_id if self.side else bandit_id - env.red
        msl_i = len(self.bandit_launch_direction[rel_j])
        if msl_i < 4:
            msl = env.state_interface["AMS"][bandit_id]["SMS"][msl_i]
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
                rel_i = int(msl["target_index"]["value"] + 0.1)
                target_id = rel_i + env.red if self.side else rel_i

                msl_x = msl["Xg_m_0"]["value"]
                msl_y = msl["Xg_m_1"]["value"]
                tar_x = env.state_interface["AMS"][target_id]["Xg_0"]["value"]
                tar_y = env.state_interface["AMS"][target_id]["Xg_1"]["value"]
                launch_line = [msl_x - tar_x, msl_y - tar_y]
                north_line = [1, 0]
                theta = acos(launch_line[0] / (launch_line[0] ** 2 + launch_line[1] ** 2) ** 0.5)
                out_product = north_line[0] * launch_line[1] - launch_line[0] * north_line[1]
                if out_product <= 0:
                    theta = -theta
                self.bandit_launch_direction[rel_j].append(theta)
        else:
            pass

    def get_some_missile_launch_direction(self, env, threat_source_id, threat_msl_id):
        rel_j = threat_source_id if self.side else threat_source_id - env.red
        threat_msl_launch_d = self.bandit_launch_direction[rel_j][threat_msl_id]

        return threat_msl_launch_d

    def get_png_target_info(self, env, bandit_id):
        rel_j = bandit_id if self.side else bandit_id - env.red
        add_msl_i = len(self.png_target_info[rel_j])
        if add_msl_i < 4:
            add_msl = env.state_interface["AMS"][bandit_id]["SMS"][add_msl_i]
            if int(add_msl["state"]["value"] + 0.1) == 2 or int(add_msl["state"]["value"] + 0.1) == 3:
                rel_i = int(add_msl["target_index"]["value"] + 0.1)
                target_id = rel_i + env.red if self.side else rel_i
                tar_x = env.state_interface["AMS"][target_id]["Xg_0"]["value"]
                tar_y = env.state_interface["AMS"][target_id]["Xg_1"]["value"]
                tar_z = env.state_interface["AMS"][target_id]["Xg_2"]["value"]
                tar_vx = env.state_interface["AMS"][target_id]["Vg_0"]["value"]
                tar_vy = env.state_interface["AMS"][target_id]["Vg_1"]["value"]
                tar_vz = env.state_interface["AMS"][target_id]["Vg_2"]["value"]
                add_info = {"x": tar_x, "y": tar_y, "z": tar_z, "vx": tar_vx, "vy": tar_vy, "vz": tar_vz}
                self.png_target_info[rel_j].append(add_info)

        for msl_i in range(len(self.png_target_info[rel_j])):
            msl = env.state_interface["AMS"][bandit_id]["SMS"][msl_i]
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:
                rel_i = int(msl["target_index"]["value"] + 0.1)
                target_id = rel_i + env.red if self.side else rel_i
                lost_guide_time = msl["lost_radar_guide_timer"]["value"]
                if lost_guide_time < 2:
                    tar_x = env.state_interface["AMS"][target_id]["Xg_0"]["value"]
                    tar_y = env.state_interface["AMS"][target_id]["Xg_1"]["value"]
                    tar_z = env.state_interface["AMS"][target_id]["Xg_2"]["value"]
                    tar_vx = env.state_interface["AMS"][target_id]["Vg_0"]["value"]
                    tar_vy = env.state_interface["AMS"][target_id]["Vg_1"]["value"]
                    tar_vz = env.state_interface["AMS"][target_id]["Vg_2"]["value"]
                else:
                    last_x = self.png_target_info[rel_j][msl_i]["x"]
                    last_y = self.png_target_info[rel_j][msl_i]["y"]
                    last_z = self.png_target_info[rel_j][msl_i]["z"]
                    tar_vx = self.png_target_info[rel_j][msl_i]["vx"]
                    tar_vy = self.png_target_info[rel_j][msl_i]["vy"]
                    tar_vz = self.png_target_info[rel_j][msl_i]["vz"]
                    tar_x = last_x + tar_vx * self.interval
                    tar_y = last_y + tar_vy * self.interval
                    tar_z = last_z + tar_vz * self.interval
                info = {"x": tar_x, "y": tar_y, "z": tar_z, "vx": tar_vx, "vy": tar_vy, "vz": tar_vz}
                self.png_target_info[rel_j][msl_i] = info

    def get_some_missile_png_target_info(self, env, threat_source_id, threat_msl_id) -> dict:
        rel_j = threat_source_id if self.side else threat_source_id - env.red
        threat_msl_png_target_info = self.png_target_info[rel_j][threat_msl_id]

        return threat_msl_png_target_info

    def cal_two_los_angle(self, env, target_id):
        i_0 = self.side * env.red
        i_1 = i_0 + 1
        coord_i_0 = [env.state_interface["AMS"][i_0]["Xg_0"]["value"], env.state_interface["AMS"][i_0]["Xg_1"]["value"]]
        coord_i_1 = [env.state_interface["AMS"][i_1]["Xg_0"]["value"], env.state_interface["AMS"][i_1]["Xg_1"]["value"]]
        coord_tar = [env.state_interface["AMS"][target_id]["Xg_0"]["value"], env.state_interface["AMS"][target_id]["Xg_1"]["value"]]
        los_0 = [coord_i_0[0] - coord_tar[0], coord_i_0[1] - coord_tar[1]]
        los_1 = [coord_i_1[0] - coord_tar[0], coord_i_1[1] - coord_tar[1]]
        r_0 = (los_0[0] ** 2 + los_0[1] ** 2) ** 0.5
        r_1 = (los_1[0] ** 2 + los_1[1] ** 2) ** 0.5
        cos_alpha = (los_0[0] * los_1[0] + los_0[1] * los_1[1]) / (r_0 * r_1)
        if cos_alpha > 1:
            cos_alpha = 1
        if cos_alpha < -1:
            cos_alpha = -1
        alpha = degrees(acos(cos_alpha))

        return alpha

    def cal_border_distance(self, env, self_id):
        border_dist_list = [1000000]
        x = env.state_interface["AMS"][self_id]["Xg_0"]["value"]
        y = env.state_interface["AMS"][self_id]["Xg_1"]["value"]
        cur_coord = [x, y]
        if int(env.state_interface["AMS"][self_id]["Vg_0"]["value"] + 0.1) == 0:
            k = env.state_interface["AMS"][self_id]["Vg_1"]["value"]
            inter1_coord = [x, 75000]
            inter2_coord = [x, -75000]
            dist1 = ((inter1_coord[0] - cur_coord[0]) ** 2 + (inter1_coord[1] - cur_coord[1]) ** 2) ** 0.5
            dist2 = ((inter2_coord[0] - cur_coord[0]) ** 2 + (inter2_coord[1] - cur_coord[1]) ** 2) ** 0.5
            tan1 = inter1_coord[1] - cur_coord[1]
            tan2 = inter2_coord[1] - cur_coord[1]
            direction = [k * tan1, k * tan2]
            dist = [dist1, dist2]
        elif int(env.state_interface["AMS"][self_id]["Vg_1"]["value"] + 0.1) == 0:
            k = env.state_interface["AMS"][self_id]["Vg_0"]["value"]
            inter1_coord = [75000, y]
            inter2_coord = [-75000, y]
            dist1 = ((inter1_coord[0] - cur_coord[0]) ** 2 + (inter1_coord[1] - cur_coord[1]) ** 2) ** 0.5
            dist2 = ((inter2_coord[0] - cur_coord[0]) ** 2 + (inter2_coord[1] - cur_coord[1]) ** 2) ** 0.5
            tan1 = inter1_coord[0] - cur_coord[0]
            tan2 = inter2_coord[0] - cur_coord[0]
            direction = [k * tan1, k * tan2]
            dist = [dist1, dist2]
        else:
            k = env.state_interface["AMS"][self_id]["Vg_1"]["value"] / env.state_interface["AMS"][self_id]["Vg_0"]["value"]
            inter1_coord = [75000, y + k * (75000 - x)]
            inter2_coord = [-75000, y + k * (-75000 - x)]
            inter3_coord = [x + 1 / k * (75000 - y), 75000]
            inter4_coord = [x + 1 / k * (-75000 - y), -75000]
            dist1 = ((inter1_coord[0] - cur_coord[0]) ** 2 + (inter1_coord[1] - cur_coord[1]) ** 2) ** 0.5
            dist2 = ((inter2_coord[0] - cur_coord[0]) ** 2 + (inter2_coord[1] - cur_coord[1]) ** 2) ** 0.5
            dist3 = ((inter3_coord[0] - cur_coord[0]) ** 2 + (inter3_coord[1] - cur_coord[1]) ** 2) ** 0.5
            dist4 = ((inter4_coord[0] - cur_coord[0]) ** 2 + (inter4_coord[1] - cur_coord[1]) ** 2) ** 0.5
            tan1 = (inter1_coord[1] - cur_coord[1]) / (inter1_coord[0] - cur_coord[0])
            tan2 = (inter2_coord[1] - cur_coord[1]) / (inter2_coord[0] - cur_coord[0])
            tan3 = (inter3_coord[1] - cur_coord[1]) / (inter3_coord[0] - cur_coord[0])
            tan4 = (inter4_coord[1] - cur_coord[1]) / (inter4_coord[0] - cur_coord[0])
            direction = [k * tan1, k * tan2, k * tan3, k * tan4]
            dist = [dist1, dist2, dist3, dist4]
        for i in range(len(direction)):
            if direction[i] > 0:
                border_dist_list.append(dist[i])
        border_distance = min(border_dist_list)

        return border_distance

    def cal_turn_direction_to_missile_cold(self, env, self_id):
        _, _, _, threat_source, _, _ = self.check_missile_threat(env, self_id)
        min_msl_tgo = env.state_interface["AMS"][0]["SMS"][0]["TGO"]["max"]
        msl_id = 0
        bandit_aircraft = env.state_interface["AMS"][threat_source]
        for msl_i, msl in enumerate(bandit_aircraft["SMS"]):
            if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying #
                if int(msl["target_index"]["value"] + 0.1) == self_id - self.side * env.red:
                    if msl["TGO"]["value"] <= min_msl_tgo:
                        msl_id = msl_i
        v_x = env.state_interface["AMS"][self_id]["Vg_0"]["value"]
        v_y = env.state_interface["AMS"][self_id]["Vg_1"]["value"]
        self_x = env.state_interface["AMS"][self_id]["Xg_0"]["value"]
        self_y = env.state_interface["AMS"][self_id]["Xg_1"]["value"]
        msl_x = env.state_interface["AMS"][threat_source]["SMS"][msl_id]["Xg_m_0"]["value"]
        msl_y = env.state_interface["AMS"][threat_source]["SMS"][msl_id]["Xg_m_1"]["value"]
        v = [v_x, v_y]
        rm = [msl_x - self_x, msl_y - self_y]
        out_product = v[0] * rm[1] - rm[0] * v[1]
        if out_product <= 0:
            turn_direction = 1
        else:
            turn_direction = -1

        return turn_direction

    def cal_turn_direction_to_target_hot(self, env, self_id, target_id):
        tar_vx = env.state_interface["AMS"][target_id]["Vg_0"]["value"]
        tar_vy = env.state_interface["AMS"][target_id]["Vg_1"]["value"]
        self_x = env.state_interface["AMS"][self_id]["Xg_0"]["value"]
        self_y = env.state_interface["AMS"][self_id]["Xg_1"]["value"]
        tar_x = env.state_interface["AMS"][target_id]["Xg_0"]["value"]
        tar_y = env.state_interface["AMS"][target_id]["Xg_1"]["value"]

        tarv = [tar_vx, tar_vy]
        r = [self_x - tar_x, self_y - tar_y]
        out_product = tarv[0] * r[1] - r[0] * tarv[1]
        if out_product >= 0:
            turn_direction = 1
        else:
            turn_direction = -1

        return turn_direction

    def cal_separate_direction(self, env, self_id):
        fri_id = 5 - self_id if self.side else 1 - self_id
        ban0_x = env.state_interface["AMS"][(1 - self.side) * env.red]["Xg_0"]["value"]
        ban0_y = env.state_interface["AMS"][(1 - self.side) * env.red]["Xg_1"]["value"]
        ban1_x = env.state_interface["AMS"][(1 - self.side) * env.red + 1]["Xg_0"]["value"]
        ban1_y = env.state_interface["AMS"][(1 - self.side) * env.red + 1]["Xg_1"]["value"]
        self_x = env.state_interface["AMS"][self_id]["Xg_0"]["value"]
        self_y = env.state_interface["AMS"][self_id]["Xg_1"]["value"]
        fri_x = env.state_interface["AMS"][fri_id]["Xg_0"]["value"]
        fri_y = env.state_interface["AMS"][fri_id]["Xg_1"]["value"]
        ban_center = [(ban0_x + ban1_x) / 2, (ban0_y + ban1_y) / 2]
        tea_center = [(self_x + fri_x) / 2, (self_y + fri_y) / 2]
        center_line = [ban_center[0] - tea_center[0], ban_center[1] - tea_center[1]]
        self_line = [self_x - tea_center[0], self_y - tea_center[1]]
        out_product = center_line[0] * self_line[1] - self_line[0] * center_line[1]
        if out_product <= 0:
            turn_direction = -1
        else:
            turn_direction = 1

        return turn_direction

    """running script"""
    def rule_script(self, env, semantic_maneuver_list):
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            rel_j = j if self.side else j - env.red
            if len(self.bandit_launch_range[rel_j]) < 4:
                self.get_bandit_launch_range(env, j)
            if len(self.bandit_launch_direction[rel_j]) < 4:
                self.get_bandit_launch_direction(env, j)
            self.get_png_target_info(env, j)

        select_top_tactics = self.top_rule(env)
        if select_top_tactics == "all_defense":
            maneuver_list, shoot_list, target_list = self.all_defense_rule(env, semantic_maneuver_list)
        elif select_top_tactics == "offense_and_defense":
            maneuver_list, shoot_list, target_list = self.offense_and_defense_rule(env, semantic_maneuver_list)
        else:
            maneuver_list, shoot_list, target_list = self.all_offense_rule(env, semantic_maneuver_list)

        return maneuver_list, shoot_list, target_list

    """top tactic selection"""
    def top_rule(self, env):
        all_defense_flag = self.judge_all_defense(env)
        offense_and_defense_flag = self.judge_offense_and_defense(env)
        if all_defense_flag:
            select_top_tactics = top_tactics["all_defense"]
        elif offense_and_defense_flag:
            select_top_tactics = top_tactics["offense_and_defense"]
        else:
            select_top_tactics = top_tactics["all_offense"]

        return select_top_tactics

    def judge_all_defense(self, env):
        flag = True
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) < 1:
                continue
            escape_flag, _ = self.check_escape_available(env, i)
            flag = flag and escape_flag

        return flag

    def judge_offense_and_defense(self, env):
        team_alive_num, bandit_alive_num = self.check_alive_num(env)
        escape_flag_list = self.get_escape_available_list(env)

        if team_alive_num == 2:
            flag = escape_flag_list[0] ^ escape_flag_list[1]
        # elif team_alive_num == 2 and bandit_alive_num == 2:
        #     flag = False
        #     near_bandit_r_list = []
        #     for i in range(self.side * env.red, env.red + self.side * env.blue):
        #         near_bandit_idx = self.threat_sort(env, i)[0]
        #         near_bandit_r_list.append(
        #             env.state_interface["AMS"][i]["relative_observation"][near_bandit_idx]["r"]["value"])
        #     bandit_idx = [i for i in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue)]
        #     bandit_between_r = \
        #         (env.state_interface["AMS"][bandit_idx[0]]["relative_observation"][bandit_idx[1]]["r"]["value"] +
        #          env.state_interface["AMS"][bandit_idx[1]]["relative_observation"][bandit_idx[0]]["r"]["value"]) / 2
        #     if MagicNumber.min_skate_range <= near_bandit_r_list[0] < MagicNumber.max_skate_range and \
        #        MagicNumber.min_skate_range <= near_bandit_r_list[1] < MagicNumber.max_skate_range and \
        #        bandit_between_r <= MagicNumber.max_between_bandit_range:
        #         flag = True
        #     else:
        #         flag = False
        else:
            flag = False

        return flag

    """target assignment"""
    def target_assign(self, env) -> list:
        target_list = []
        team_alive_num, bandit_alive_num = self.check_alive_num(env)
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            rel_i = i - env.red if self.side else i
            friend_id = 5 - i if self.side else 1 - i
            near_bandit_id = self.threat_sort(env, i)[0]
            if int(env.state_interface["AMS"][near_bandit_id]["alive"]["value"] + 0.1) < 1:
                near_bandit_id = 1 - near_bandit_id if self.side else 5 - near_bandit_id

            near_ta = env.state_interface["AMS"][i]["relative_observation"][near_bandit_id]["TA"]["value"]
            near_r = env.state_interface["AMS"][i]["relative_observation"][near_bandit_id]["r"]["value"]
            near_msl_remain = int(env.state_interface["AMS"][near_bandit_id]["AAM_remain"]["value"])
            near_rtr = env.state_interface["AMS"][near_bandit_id]["attack_zone_list"][rel_i]["Rtr"]["value"]

            r = env.state_interface["AMS"][i]["relative_observation"][near_bandit_id]["r"]["value"]
            r_team = env.state_interface["AMS"][i]["relative_observation"][friend_id]["r"]["value"]
            msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
            alive = env.state_interface["AMS"][i]["alive"]["value"]

            threat_flag, _, _, _, _, _ = self.check_missile_threat(env, i)
            msl_target_list = self.get_self_missile_work_target(env, i)

            if int(alive + 0.1) == 1:
                threat_flag, _, _, threat_source, _, _ = self.check_missile_threat(env, i)
                if threat_flag:
                    target = threat_source
                else:
                    if len(msl_target_list) == 1:
                        target = msl_target_list[0]
                    else:
                        target = self.default_target[rel_i]

                    if bandit_alive_num == 1:
                        target = self.get_only_alive_bandit_index(env)

                    if team_alive_num == 1 and bandit_alive_num == 2:
                        if msl_remain <= 1:
                            target = self.threat_sort(env, i)[0]
                        else:  # 一目标打一枚
                            if len(msl_target_list) == 0 or len(msl_target_list) == 2:
                                target = self.threat_sort(env, i)[0]
                            else:
                                target = 1 - msl_target_list[0] if self.side else 5 - msl_target_list[0]

                if msl_remain > 0:
                    if near_r <= MagicNumber.min_escape_range:
                        if near_msl_remain > 0:
                            if degrees(abs(near_ta)) > 120:
                                target = near_bandit_id
                    # else:
                    #     if not threat_flag:
                    #         if r > MagicNumber.min_safe_range:
                    #             if r_team < MagicNumber.min_team_range:
                    #                 if len(msl_target_list) == 0:
                    #                     target = 5 - i if self.side else 1 - i
            else:
                target = (1 - self.side) * env.red  # TODO

            target_list.append(target)

        return target_list

    """three top tactics"""
    def all_defense_rule(self, env, semantic_maneuver_list):
        target_list = self.target_assign(env)
        macro_tactics_list, shoot_list = self.all_defense_macro_tactics_assign(env)
        maneuver_list = []
        tactic_l = []

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            rel_i = i - env.red if self.side else i
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:
                shoot = shoot_list[i - env.red if self.side else i]
                target = target_list[i - env.red if self.side else i]
                macro_tactic = macro_tactics_list[i - env.red if self.side else i]
                if self.combo_executing[i - env.red if self.side else i] is not None:
                    macro_tactic = self.combo_executing[i - env.red if self.side else i]

                if macro_tactic in macro_tactics["defense"]["abort"].keys():
                    maneuver = self.abort(env, i, target, macro_tactic, semantic_maneuver_list[rel_i])
                elif macro_tactic in macro_tactics["defense"]["evade"].keys():
                    maneuver = self.evade(env, i, target, macro_tactic, semantic_maneuver_list[rel_i])
                elif macro_tactic == "intercept_cata" or macro_tactic == "intercept_level" or macro_tactic == "intercept_climb":
                    maneuver = self.intercept(env, i, shoot, macro_tactic, semantic_maneuver_list[rel_i])
                elif macro_tactic == "climb" or macro_tactic == "separate":
                    maneuver = self.format(env, i, shoot, macro_tactic, semantic_maneuver_list[rel_i])
                elif macro_tactic == "escape":
                    maneuver = self.escape(env, i, semantic_maneuver_list[rel_i])
                else:
                    maneuver = semantic_maneuver_list[rel_i]["maintain"]

                maneuver_list.append(maneuver)
                tactic_l.append(macro_tactic)
            else:
                maneuver_list.append(semantic_maneuver_list[rel_i]["maintain"])#TODO
                tactic_l.append(macro_tactics["other"]["maintain"])

        # print(tactic_l)
        return maneuver_list, shoot_list, target_list

    def offense_and_defense_rule(self, env, semantic_maneuver_list):
        target_list = self.target_assign(env)
        macro_tactics_list, shoot_list = self.offense_and_defense_macro_tactics_assign(env, target_list)
        maneuver_list = []
        tactic_l = []

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            rel_i = i - env.red if self.side else i
            shoot = shoot_list[i - env.red if self.side else i]
            target = target_list[i - env.red if self.side else i]
            macro_tactic = macro_tactics_list[i - env.red if self.side else i]
            if self.combo_executing[i - env.red if self.side else i] is not None:
                macro_tactic = self.combo_executing[i - env.red if self.side else i]

            if macro_tactic in macro_tactics["defense"]["abort"].keys():
                maneuver = self.abort(env, i, target, macro_tactic, semantic_maneuver_list[rel_i])
            elif macro_tactic in macro_tactics["defense"]["evade"].keys():
                maneuver = self.evade(env, i, target, macro_tactic, semantic_maneuver_list[rel_i])
            elif macro_tactic == "intercept_cata" or macro_tactic == "intercept_level" or macro_tactic == "intercept_climb":
                maneuver = self.intercept(env, i, shoot, macro_tactic, semantic_maneuver_list[rel_i])
            elif macro_tactic == "climb" or macro_tactic == "separate":
                maneuver = self.format(env, i, shoot, macro_tactic, semantic_maneuver_list[rel_i])
            elif macro_tactic == "banzai":
                maneuver, shoot = self.banzai(env, i, macro_tactic, semantic_maneuver_list[rel_i])
                shoot_list[i - env.red if self.side else i] = shoot
            elif macro_tactic == "escape":
                maneuver = self.escape(env, i, semantic_maneuver_list[rel_i])
            else:
                maneuver = semantic_maneuver_list[rel_i]["maintain"]

            maneuver_list.append(maneuver)
            tactic_l.append(macro_tactic)

        # print(tactic_l)
        return maneuver_list, shoot_list, target_list

    def all_offense_rule(self, env, semantic_maneuver_list):
        target_list = self.target_assign(env)
        macro_tactics_list, shoot_list = self.all_offense_macro_tactics_assign(env, target_list)
        shoot_available_list = self.shoot_available_assign(env, target_list)
        for n in range(len(shoot_list)):
            if shoot_list[n] is not None:
                shoot_list[n] = int(shoot_list[n] and shoot_available_list[n])
        maneuver_list = []
        tactic_l = []

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            rel_i = i - env.red if self.side else i
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:
                shoot = shoot_list[i - env.red if self.side else i]
                target = target_list[i - env.red if self.side else i]
                macro_tactic = macro_tactics_list[i - env.red if self.side else i]
                if self.combo_executing[i - env.red if self.side else i] is not None:
                    macro_tactic = self.combo_executing[i - env.red if self.side else i]

                if macro_tactic in macro_tactics["defense"]["abort"].keys():
                    maneuver = self.abort(env, i, target, macro_tactic, semantic_maneuver_list[rel_i])
                elif macro_tactic in macro_tactics["defense"]["evade"].keys():
                    maneuver = self.evade(env, i, target, macro_tactic, semantic_maneuver_list[rel_i])
                elif macro_tactic == "intercept_cata" or macro_tactic == "intercept_level" or macro_tactic == "intercept_climb":
                    maneuver = self.intercept(env, i, shoot, macro_tactic, semantic_maneuver_list[rel_i])
                elif macro_tactic == "climb" or macro_tactic == "separate":
                    maneuver = self.format(env, i, shoot, macro_tactic, semantic_maneuver_list[rel_i])
                elif macro_tactic == "banzai":
                    maneuver, shoot = self.banzai(env, i, macro_tactic, semantic_maneuver_list[rel_i])
                    shoot_list[i - env.red if self.side else i] = shoot
                elif macro_tactic == "escape":
                    maneuver = self.escape(env, i, semantic_maneuver_list[rel_i])
                else:
                    maneuver = semantic_maneuver_list[rel_i]["maintain"]
                maneuver_list.append(maneuver)
                tactic_l.append(macro_tactic)
            else:
                maneuver_list.append(semantic_maneuver_list[rel_i]["maintain"])#TODO
                tactic_l.append(macro_tactics["other"]["maintain"])

        # print(tactic_l)
        return maneuver_list, shoot_list, target_list

    """tactics assignment"""
    def all_defense_macro_tactics_assign(self, env):
        macro_tactics_list = []
        shoot_list = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:
                _, threat_level, advice_tactic, threat_source, _, _ = self.check_missile_threat(env, i)
                # macro_tactic, shoot = self.retreat_macro_tactic(env, i, threat_source)
                macro_tactic, shoot = self.high_defense_macro_tactic(env, i, threat_source, threat_level, advice_tactic)
                macro_tactics_list.append(macro_tactic)
                shoot_list.append(shoot)
            else:
                macro_tactics_list.append(macro_tactics["other"]["maintain"])  # TODO
                shoot_list.append(0)  # TODO

        return macro_tactics_list, shoot_list

    def offense_and_defense_macro_tactics_assign(self, env, target_list: list):
        macro_tactics_list = []
        shoot_list = []

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            target_id = target_list[i - env.red if self.side else i]
            escape_flag, _ = self.check_escape_available(env, i)
            if escape_flag:
                _, threat_level, advice_tactic, threat_source, _, _ = self.check_missile_threat(env, i)
                # macro_tactic, shoot = self.retreat_macro_tactic(env, i, threat_source)
                macro_tactic, shoot = self.high_defense_macro_tactic(env, i, threat_source, threat_level, advice_tactic)
            else:
                macro_tactic, shoot = self.general_offense_macro_tactic(env, i, target_id)
            macro_tactics_list.append(macro_tactic)
            shoot_list.append(shoot)

        return macro_tactics_list, shoot_list

    def all_offense_macro_tactics_assign(self, env, target_list: list):
        macro_tactics_list = []
        shoot_list = []
        team_alive_num, bandit_alive_num = self.check_alive_num(env)
        if team_alive_num == 1 and bandit_alive_num == 1:  # 1v1 offense
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                alive = env.state_interface["AMS"][i]["alive"]["value"]
                if int(alive + 0.1) == 1:
                    target_id = target_list[i - env.red if self.side else i]
                    macro_tactic, shoot = self.general_offense_macro_tactic(env, i, target_id)
                    macro_tactics_list.append(macro_tactic)
                    shoot_list.append(shoot)
                else:
                    macro_tactics_list.append(macro_tactics["other"]["maintain"])  # TODO
                    shoot_list.append(0)  # TODO
        elif team_alive_num == 2 and bandit_alive_num == 1:  # 2v1 offense
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                target_id = target_list[i - env.red if self.side else i]
                macro_tactic, shoot = self.general_offense_macro_tactic(env, i, target_id)
                macro_tactics_list.append(macro_tactic)
                shoot_list.append(shoot)
        elif team_alive_num == 1 and bandit_alive_num == 2:  # 1v2 offense
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                alive = env.state_interface["AMS"][i]["alive"]["value"]
                if int(alive + 0.1) == 1:
                    target_id = target_list[i - env.red if self.side else i]
                    macro_tactic, shoot = self.disadvantaged_offense_macro_tactic(env, i, target_id)
                    macro_tactics_list.append(macro_tactic)
                    shoot_list.append(shoot)
                else:
                    macro_tactics_list.append(macro_tactics["other"]["maintain"])  # TODO
                    shoot_list.append(0)  # TODO
        elif team_alive_num == 2 and bandit_alive_num == 2:  # 2v2 offense
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                target_id = target_list[i - env.red if self.side else i]
                macro_tactic, shoot = self.general_offense_macro_tactic(env, i, target_id)
                macro_tactics_list.append(macro_tactic)
                shoot_list.append(shoot)
        else:  # 2v0 1v0
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                alive = env.state_interface["AMS"][i]["alive"]["value"]
                if int(alive + 0.1) == 1:
                    macro_tactics_list.append(macro_tactics["other"]["maintain"])
                    shoot_list.append(0)
                else:
                    macro_tactics_list.append(macro_tactics["other"]["maintain"])  # TODO
                    shoot_list.append(0)  # TODO

        return macro_tactics_list, shoot_list

    def no_missile_remain_macro_tactic(self, env, self_id):
        friend_msl_remain = int(env.state_interface["AMS"][5 - self_id if self.side else 1 - self_id]["AAM_remain"]["value"])
        if friend_msl_remain > 0:
            macro_tactic = macro_tactics["defense"]["evade"]["crank_50"]
        else:
            macro_tactic = macro_tactics["defense"]["escape"]

        shoot = 0

        return macro_tactic, shoot

    def shoot_available_assign(self, env, target_list: list) -> list:
        shoot_available_flag_list = [True, True]
        if int(target_list[0]) == int(target_list[1]):
            alpha = self.cal_two_los_angle(env, target_list[0])
            if alpha >= MagicNumber.threshold_alpha_angle:
                shoot_available_flag_list = [True, True]
            else:
                i_0 = self.side * env.red
                i_1 = i_0 + 1
                r_0 = env.state_interface["AMS"][i_0]["relative_observation"][target_list[0]]["r"]["value"]
                r_1 = env.state_interface["AMS"][i_1]["relative_observation"][target_list[0]]["r"]["value"]
                if r_0 <= r_1:
                    close_i = i_0
                else:
                    close_i = i_1

                close_msl_remain = int(env.state_interface["AMS"][close_i]["AAM_remain"]["value"])
                close_ta = env.state_interface["AMS"][close_i]["relative_observation"][target_list[0]]["TA"]["value"]

                if close_msl_remain <= 1:
                    if degrees(abs(close_ta)) <= MagicNumber.threshold_launch_ta:  # far one to shoot
                        shoot_available_flag_list[close_i - self.side * env.red] = False
                    else:  # close one to shoot
                        shoot_available_flag_list[1 - (close_i - self.side * env.red)] = False
                else:  # close one to shoot
                    shoot_available_flag_list[1 - (close_i - self.side * env.red)] = False
        else:
            shoot_available_flag_list = [True, True]

        return shoot_available_flag_list

    def low_defense_macro_tactic(self, threat_level, advice_tactic):
        if threat_level == "low":
            macro_tactic = macro_tactics["defense"]["evade"]["change_direction"]
        elif threat_level == "medium":
            macro_tactic = macro_tactics["defense"]["evade"]["crank_50_dive_25"]
        else:
            macro_tactic = macro_tactics["other"]["maintain"]
        if advice_tactic is not None:
            macro_tactic = advice_tactic

        return macro_tactic

    def high_defense_macro_tactic(self, env, self_id, threat_source, threat_level, advice_tactic):
        if threat_level == "high":
            # macro_tactic, shoot = self.retreat_macro_tactic(env, self_id, threat_source)
            macro_tactic = macro_tactics["defense"]["evade"]["circle"]
        elif threat_level == "danger":
            macro_tactic = macro_tactics["defense"]["evade"]["circle"]
        else:
            macro_tactic = macro_tactics["other"]["maintain"]
        if advice_tactic is not None:
            macro_tactic = advice_tactic
        shoot = 0

        return macro_tactic, shoot

    def retreat_macro_tactic(self, env, self_id, threat_source, semantic_maneuver):
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        tas = env.state_interface["AMS"][self_id]["TAS"]["value"]
        ao = env.state_interface["AMS"][self_id]["relative_observation"][threat_source]["AO"]["value"]

        if height >= MagicNumber.mid_splits_height:
            if degrees(abs(abs(ao) - pi)) > 145:
                if 5000 <= height < 6500:
                    if tas <= 300:
                        semantic_maneuver["split_s"]["vel_cmd"] = 300
                        semantic_maneuver["split_s"]["ny_cmd"] = 5
                        macro_tactic = macro_tactics["defense"]["abort"]["split_s"]
                    elif 300 < tas <= 400:
                        semantic_maneuver["split_s"]["vel_cmd"] = 350
                        semantic_maneuver["split_s"]["ny_cmd"] = 8
                        macro_tactic = macro_tactics["defense"]["abort"]["split_s"]
                    else:
                        semantic_maneuver["abort_dive_25"]["ny_cmd"] = 8
                        macro_tactic = macro_tactics["defense"]["abort"]["abort_dive_25_2k"]
                elif 6500 <= height < 7500:
                    if tas <= 300:
                        semantic_maneuver["split_s"]["vel_cmd"] = 4000
                        semantic_maneuver["split_s"]["ny_cmd"] = 5
                        macro_tactic = macro_tactics["defense"]["abort"]["split_s"]
                    else:
                        semantic_maneuver["split_s"]["vel_cmd"] = 680
                        semantic_maneuver["split_s"]["ny_cmd"] = 8
                        macro_tactic = macro_tactics["defense"]["abort"]["split_s"]
                elif 7500 <= height < 10000:
                    if tas <= 400:
                        semantic_maneuver["split_s"]["vel_cmd"] = 680
                        semantic_maneuver["split_s"]["ny_cmd"] = 6
                        macro_tactic = macro_tactics["defense"]["abort"]["split_s"]
                    else:
                        semantic_maneuver["split_s"]["vel_cmd"] = 680
                        semantic_maneuver["split_s"]["ny_cmd"] = 8
                        macro_tactic = macro_tactics["defense"]["abort"]["split_s"]
                else:
                    semantic_maneuver["split_s"]["vel_cmd"] = 400
                    semantic_maneuver["split_s"]["ny_cmd"] = 8
                    macro_tactic = macro_tactics["defense"]["abort"]["split_s"]
            else:
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 8
                macro_tactic = macro_tactics["defense"]["abort"]["abort_dive_25_3k"]
        elif MagicNumber.min_splits_height <= height < MagicNumber.mid_splits_height:  # [4000, 5000]
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 8
                macro_tactic = macro_tactics["defense"]["abort"]["abort_dive_25_3k"]
        elif MagicNumber.min_large_abort_dive_25_height <= height < MagicNumber.min_splits_height:  # [3000, 4000]
            if tas >= 400:
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 8
            else:
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 5.5
            macro_tactic = macro_tactics["defense"]["abort"]["abort_dive_25_2k"]
        elif MagicNumber.min_small_abort_no_dive_height <= height < MagicNumber.min_large_abort_dive_25_height:  # [2000, 3000]
            if tas >= 400:
                semantic_maneuver["abort_dive_25"]["vel_cmd"] = 350
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 8
            else:
                semantic_maneuver["abort_dive_25"]["vel_cmd"] = 300
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 5.5
            macro_tactic = macro_tactics["defense"]["abort"]["abort_dive_25_1k"]
        else:
            macro_tactic = macro_tactics["defense"]["abort"]["abort"]
        shoot = 0

        return macro_tactic, shoot

    def disadvantaged_offense_macro_tactic(self, env, self_id, target_id):
        msl_remain = int(env.state_interface["AMS"][self_id]["AAM_remain"]["value"])
        if msl_remain <= 1:
            macro_tactic, shoot = self.general_offense_macro_tactic(env, self_id, target_id)
        else:
            msl_target_list = self.get_self_missile_work_target(env, self_id)
            if len(msl_target_list) == 0 or len(msl_target_list) == 2:
                macro_tactic, shoot = self.general_offense_macro_tactic(env, self_id, target_id)
            else:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
                shoot = 1
        return macro_tactic, shoot

    def general_offense_macro_tactic(self, env, self_id, target_id):  # absolute index
        team_alive_num, bandit_alive_num = self.check_alive_num(env)
        friend_id = 5 - self_id if self.side else 1 - self_id
        if not ((target_id < env.red) ^ (self_id < env.red)):
            target_id = self.threat_sort(env, self_id)[0]
        r = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["r"]["value"]
        r_team = env.state_interface["AMS"][self_id]["relative_observation"][friend_id]["r"]["value"]
        ta = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["TA"]["value"]
        ao = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["AO"]["value"]
        msl_remain = int(env.state_interface["AMS"][self_id]["AAM_remain"]["value"])
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        height_tar = -env.state_interface["AMS"][target_id]["Xg_2"]["value"]
        Rpi = env.state_interface["AMS"][self_id]["attack_zone_list"][target_id if self.side else target_id - env.red]["Rpi"]["value"]
        Rtr = env.state_interface["AMS"][self_id]["attack_zone_list"][target_id if self.side else target_id - env.red]["Rtr"]["value"]

        if msl_remain == 4:
            if (height - height_tar) >= 2000:
                threshold = 0.8 * (Rpi - Rtr) + Rtr
            elif -1500 <= (height - height_tar) <= 2000:
                threshold = 0.9 * (Rpi - Rtr) + Rtr
            else:
                threshold = 1.1 * (Rpi - Rtr) + Rtr
            if threshold <= 1.2 * Rtr:
                threshold = 1.2 * Rtr
            if r < threshold:
                macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
            else:
                macro_tactic = macro_tactics["offense"]["intercept_cata"] #crank_50
                shoot = 0

            if bandit_alive_num == 1:
                self_msl_work_flag = self.check_self_missile_work(env, self_id)
                team_msl_work_flag = self.check_team_missile_work(env)
                if self_msl_work_flag:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"]
                    shoot = 0
                else:
                    if team_msl_work_flag:
                        macro_tactic = macro_tactics["defense"]["evade"]["crank_50"] #crank_50
                        shoot = 0

            if shoot == 0:
                if height >= MagicNumber.max_offense_height:
                    macro_tactic = macro_tactics["offense"]["intercept_level"]
                else:
                    macro_tactic = macro_tactics["offense"]["intercept_climb"]

        elif msl_remain == 3:
            stop_flag = self.check_self_missile_stop(env, self_id, 4 - msl_remain - 1)
            if stop_flag:
                if (height - height_tar) >= 1500:
                    threshold = 0.8 * (Rpi - Rtr) + Rtr
                elif -1500 <= (height - height_tar) <= 1500:
                    threshold = 0.6 * (Rpi - Rtr) + Rtr
                else:
                    threshold = 1 * (Rpi - Rtr) + Rtr
                if threshold <= 1.1 * Rtr:
                    threshold = 1.1 * Rtr
                if r < threshold:
                    macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
                else:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"] #crank_50
                    shoot = 0
            else:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
                shoot = 0

            if bandit_alive_num == 1:
                self_msl_work_flag = self.check_self_missile_work(env, self_id)
                team_msl_work_flag = self.check_team_missile_work(env)
                if self_msl_work_flag:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"]
                    shoot = 0
                else:
                    if team_msl_work_flag:
                        macro_tactic = macro_tactics["defense"]["evade"]["crank_50"] #crank_50
                        shoot = 0

        elif msl_remain == 2:
            # terminal_guide_flag = self.check_self_missile_stop_or_low_energy(env, 800, self_id, 4 - msl_remain - 1)
            stop_flag = self.check_self_missile_stop(env, self_id, 4 - msl_remain - 1)
            if stop_flag:
                if degrees(abs(ta)) > 90:
                    if (height - height_tar) >= 1000:
                        threshold = 0.4 * (Rpi - Rtr) + Rtr
                    elif -1000 <= (height - height_tar) <= 1000:
                        threshold = 0.5 * (Rpi - Rtr) + Rtr
                    else:
                        threshold = 0.9 * (Rpi - Rtr) + Rtr
                    if threshold <= 0.9 * Rtr:
                        threshold = 0.9 * Rtr
                    if r < threshold:
                        macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
                    else:
                        macro_tactic = macro_tactics["offense"]["intercept_cata"]
                        shoot = 0
                else:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"]
                    shoot = 0
            else:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
                shoot = 0

            if bandit_alive_num == 1:
                self_msl_work_flag = self.check_self_missile_work(env, self_id)
                team_msl_work_flag = self.check_team_missile_work(env)
                if self_msl_work_flag:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"]
                    shoot = 0
                else:
                    if team_msl_work_flag:
                        macro_tactic = macro_tactics["defense"]["evade"]["crank_50"]
                        shoot = 0

        elif msl_remain == 1:
            # terminal_guide_flag = self.check_self_missile_stop_or_low_energy(env, 700, self_id, 4 - msl_remain - 1)
            stop_flag = self.check_self_missile_stop(env, self_id, 4 - msl_remain - 1)
            if stop_flag:
                # if (height - height_tar) >= 500:
                #     threshold = 0.4 * (Rpi - Rtr) + Rtr
                # elif -1000 <= (height - height_tar) <= 500:
                #     threshold = 0.5 * (Rpi - Rtr) + Rtr
                # else:
                #     threshold = 0.8 * (Rpi - Rtr) + Rtr
                # if threshold <= 0.7 * Rtr:
                #     threshold = 0.7 * Rtr
                if degrees(abs(ta)) > 120:  # 迎头
                    threshold = 25000
                elif degrees(abs(ta)) < 60:  # 尾追
                    threshold = 15000
                else:
                    threshold = 20000

                if r < threshold:
                    macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
                else:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"] #crank_50
                    shoot = 0

            else:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
                shoot = 0

            if bandit_alive_num == 1:
                self_msl_work_flag = self.check_self_missile_work(env, self_id)
                team_msl_work_flag = self.check_team_missile_work(env)
                if self_msl_work_flag:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"]
                    shoot = 0
                else:
                    if team_msl_work_flag:
                        macro_tactic = macro_tactics["defense"]["evade"]["crank_50"]
                        shoot = 0

        else:
            stop_flag = self.check_self_missile_stop(env, self_id, 4 - msl_remain - 1)
            if stop_flag:
                macro_tactic, shoot = self.no_missile_remain_macro_tactic(env, self_id)
            else:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
                shoot = 0

        if msl_remain > 0:  # high priority
            rel_i = self_id - env.red if self.side else self_id
            near_bandit_id = self.threat_sort(env, self_id)[0]
            if int(env.state_interface["AMS"][near_bandit_id]["alive"]["value"] + 0.1) < 1:
                near_bandit_id = 1 - near_bandit_id if self.side else 5 - near_bandit_id
            near_ta = env.state_interface["AMS"][self_id]["relative_observation"][near_bandit_id]["TA"]["value"]
            near_r = env.state_interface["AMS"][self_id]["relative_observation"][near_bandit_id]["r"]["value"]
            near_msl_remain = int(env.state_interface["AMS"][near_bandit_id]["AAM_remain"]["value"])
            near_rtr = env.state_interface["AMS"][near_bandit_id]["attack_zone_list"][rel_i]["Rtr"]["value"]

            if near_r <= MagicNumber.min_escape_range:
                if near_msl_remain > 0:
                    if degrees(abs(near_ta)) > 120:
                        macro_tactic = macro_tactics["offense"]["banzai"]
                        shoot = 0  # TODO
            else:
                threat_flag, threat_level, advice_tactic, _, _, _ = self.check_missile_threat(env, self_id)
                if threat_flag:
                    macro_tactic = self.low_defense_macro_tactic(threat_level, advice_tactic)
                else:
                    if near_r > MagicNumber.min_safe_range:
                        if r_team < MagicNumber.min_team_range:
                            # work_target_list = self.get_self_missile_work_target(env, self_id)
                            # if len(work_target_list) == 0:
                            #     macro_tactic = macro_tactics["other"]["separate"]
                            #     shoot = 0
                            pass
                        else:
                            if height < MagicNumber.min_offense_height:
                                macro_tactic = macro_tactics["offense"]["climb"]
                                shoot = 0

        return macro_tactic, shoot

    def shoot_one_target_macro_tactic(self, env, self_id, target_id):
        self_missile_work_target_list = self.get_self_missile_work_target(env, self_id)
        if target_id in self_missile_work_target_list:
            macro_tactic = macro_tactics["offense"]["intercept_cata"]
            shoot = 0
        else:
            macro_tactic = macro_tactics["offense"]["intercept_cata"]
            shoot = 1

        return macro_tactic, shoot

    def ready_to_shoot(self, ao, maneuver_when_shoot: str):
        if degrees(abs(ao)) > MagicNumber.threshold_launch_ao:
            macro_tactic = macro_tactics["offense"]["intercept_cata"]
            shoot = 0
        else:
            macro_tactic = maneuver_when_shoot
            shoot = 1

        return macro_tactic, shoot

    """macro tactics"""
    def abort(self, env, self_id, target_id, macro_tactic: str, semantic_maneuver):
        rel_i = self_id - env.red if self.side else self_id
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        ao = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["AO"]["value"]
        chi = env.state_interface["AMS"][self_id]["attg_2"]["value"]
        theta = env.state_interface["AMS"][self_id]["attg_1"]["value"]
        threat_flag, threat_level, _, _, _, launch_direction = self.check_missile_threat(env, self_id)
        msl_turn_direction = self.cal_turn_direction_to_missile_cold(env, self_id)
        maneuver, combo_complete_flag = self.combo[rel_i]["defense"]["abort"].execute(height, ao, chi, theta,
                                                                                      launch_direction,
                                                                                      msl_turn_direction,
                                                                                      threat_flag, threat_level,
                                                                                      macro_tactic, semantic_maneuver)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver

    def evade(self, env, self_id, target_id, macro_tactic: str, semantic_maneuver):
        rel_i = self_id - env.red if self.side else self_id
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        tas = env.state_interface["AMS"][self_id]["TAS"]["value"]
        ao = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["AO"]["value"]
        threat_flag, threat_level, _, _, launch_range, launch_direction = self.check_missile_threat(env, self_id)
        tar_turn_direction = self.cal_turn_direction_to_target_hot(env, self_id, target_id)
        if threat_flag:
            if macro_tactic == "circle" or macro_tactic == "circle_dive_25":
                msl_turn_direction = self.cal_turn_direction_to_missile_cold(env, self_id)
            else:
                msl_turn_direction = 0
        else:
            msl_turn_direction = 0
        maneuver, combo_complete_flag = self.combo[rel_i]["defense"]["evade"].execute(height, tas, ao, launch_range,
                                                                                      launch_direction,
                                                                                      self.default_clockwise[rel_i],
                                                                                      msl_turn_direction,
                                                                                      tar_turn_direction,
                                                                                      threat_flag,
                                                                                      threat_level, macro_tactic,
                                                                                      semantic_maneuver)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver

    def intercept(self, env, self_id, shoot, macro_tactic: str, semantic_maneuver):
        rel_i = self_id - env.red if self.side else self_id
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        threat_flag, _, _, _, _, _ = self.check_missile_threat(env, self_id)
        maneuver, combo_complete_flag = self.combo[rel_i]["offense"]["intercept"].execute(height, shoot, threat_flag,
                                                                                          macro_tactic,
                                                                                          semantic_maneuver)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver

    def format(self, env, self_id, shoot, macro_tactic: str, semantic_maneuver):
        rel_i = self_id - env.red if self.side else self_id
        friend_id = 5 - self_id if self.side else 1 - self_id
        r_team = env.state_interface["AMS"][self_id]["relative_observation"][friend_id]["r"]["value"]
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        threat_flag, _, _, _, _, _ = self.check_missile_threat(env, self_id)
        turn_direction = self.cal_separate_direction(env, self_id)
        maneuver, combo_complete_flag = self.combo[rel_i]["others"]["format"].execute(height, r_team, turn_direction,
                                                                                      shoot, threat_flag, macro_tactic,
                                                                                      semantic_maneuver)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver

    def banzai(self, env, self_id, macro_tactic: str, semantic_maneuver):
        rel_i = self_id - env.red if self.side else self_id
        near_bandit_id = self.threat_sort(env, self_id)[0]
        if int(env.state_interface["AMS"][near_bandit_id]["alive"]["value"] + 0.1) < 1:
            near_bandit_id = 1 - near_bandit_id if self.side else 5 - near_bandit_id
        _, threat_level, _, _, _, _ = self.check_missile_threat(env, self_id)
        fuck_flag, flying_time, r_dot = self.check_self_missile_fuck_some_target(env, self_id, near_bandit_id)

        maneuver, shoot, combo_complete_flag = self.combo[rel_i]["offense"]["banzai"].execute(env, self_id,
                                                                                              near_bandit_id,
                                                                                              fuck_flag, flying_time,
                                                                                              r_dot, threat_level,
                                                                                              semantic_maneuver)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver, shoot

    def escape(self, env, self_id, semantic_maneuver):
        team_msl_target_list = self.get_team_missile_work_target(env)
        if len(team_msl_target_list) > 0:
            maneuver = semantic_maneuver["intercept_cata"]
        else:
            if env.state_interface["AMS"][self_id]["out_of_border_time"]["value"] > 0:
                maneuver = semantic_maneuver["crank_50"]
            else:
                border_distance = self.cal_border_distance(env, self_id)
                if border_distance > MagicNumber.far_border_range:
                    maneuver = semantic_maneuver["out"]
                elif MagicNumber.mid_border_range <= border_distance <= MagicNumber.far_border_range:
                    maneuver = semantic_maneuver["out_150"]
                elif MagicNumber.near_border_range <= border_distance <= MagicNumber.mid_border_range:
                    maneuver = semantic_maneuver["out_120"]
                else:
                    maneuver = semantic_maneuver["crank_90"]

        return maneuver

    def before_step_for_sample(self, env):
        semantic_maneuver_list = [deepcopy(semantic_maneuver_default), deepcopy(semantic_maneuver_default)]
        maneuver_list, shoot_list, target_list = self.rule_script(env, semantic_maneuver_list)

        actions = []
        current_step_available = []
        for i in range(self.side * env.red, env.red + self.side * env.red):
            current_step_available.append(state_method_independ.get_aircraft_available(env, i))
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:
                maneuver = maneuver_list[i - env.red if self.side else i]
                shoot = shoot_list[i - env.red if self.side else i]
                target = target_list[i - env.red if self.side else i]
                # print("step: ", maneuver)
                # if i == 1:
                #     target = 3
                #     r = env.state_interface["AMS"][i]["relative_observation"][target]["r"]["value"]
                #     if 20000 < r < 80000:
                #         if int(env.state_interface["AMS"][i]["AAM_remain"]["value"] + 0.1) == 4:
                #             shoot = 1
                #         else:
                #             shoot = 0
                #     else:
                #         shoot = 0
                # else:
                #     maneuver = semantic_maneuver["maintain"]
                #     shoot = 0

                env.action_interface["AMS"][i]["BotManeuver"]["base_direction"]["value"] = 10
                env.action_interface["AMS"][i]["BotManeuver"]["combat_mode"]["value"] = 0
                env.action_interface["AMS"][i]["BotManeuver"]["flag_after_burning"]["value"] = 1
                env.action_interface["AMS"][i]["BotManeuver"]["horizontal_cmd"]["value"] = maneuver["horizontal_cmd"]
                env.action_interface["AMS"][i]["BotManeuver"]["vertical_cmd"]["value"] = maneuver["vertical_cmd"]
                env.action_interface["AMS"][i]["BotManeuver"]["vel_cmd"]["value"] = maneuver["vel_cmd"]
                env.action_interface["AMS"][i]["BotManeuver"]["ny_cmd"]["value"] = maneuver["ny_cmd"]
                env.action_interface["AMS"][i]["BotManeuver"]["clockwise_cmd"]["value"] = maneuver["clockwise_cmd"]
                env.action_interface["AMS"][i]["BotManeuver"]["base_direction"]["value"] = maneuver["base_direction"]
                env.action_interface["AMS"][i]["BotManeuver"]["maneuver_target"]["value"] = target
                env.action_interface["AMS"][i]["action_target"]["value"] = target
                # env.action_interface["AMS"][i]["action_shoot_target"]["value"] = -1
                if not ((target < env.red) ^ (i < env.red)):
                    env.action_interface["AMS"][i]["action_shoot_target"]["value"] = -1
                else:
                    if shoot:
                        env.action_interface["AMS"][i]["action_shoot_target"]["value"] = target - (1 - self.side) * env.red
                    else:
                        env.action_interface["AMS"][i]["action_shoot_target"]["value"] = -1

                hor_one_hots = index_to_one_hot(maneuver["horizontal_cmd"], 8)
                ver_one_hots = index_to_one_hot(maneuver["vertical_cmd"], 7)

                actions.append(
                    [env.action_interface["AMS"][i]["action_shoot_target"]["value"],
                     target,
                     maneuver["horizontal_cmd"],
                     maneuver["vertical_cmd"],
                     maneuver["vel_cmd"],
                     maneuver["ny_cmd"],
                     hor_one_hots,
                     ver_one_hots]
                )
            else:
                actions.append(
                    [0, 0, 0, 0, 0, 0, [0] * 8, [0] * 7]
                )

        for i in range(env.red + env.blue):
            if i < env.red:
                for j in range(env.blue):
                    env.action_interface["AMS"][i]["action_shoot_predict_list"][j]["shoot_predict"][
                        "value"] = 0
            else:
                for j in range(env.red):
                    env.action_interface["AMS"][i]["action_shoot_predict_list"][j]["shoot_predict"][
                        "value"] = 0

        self.batchs["current_step_available"].append(current_step_available)

        state_id_one_hot = state_method_independ.get_kteam_ids_one_hot_state(env, 2)
        states_global = state_method_independ.get_kteam_global_ground_truth_state(env, self.side)
        states_atten = state_method_independ.get_kteam_aircraft_state_for_attention(env, self.side)
        msl_token_self = state_method_independ.get_self_kteam_msl_tokens(env, self.side)
        msl_token_bandit = state_method_independ.get_bandit_kteam_msl_tokens(env, self.side)

        self.batchs["state_global"].append(states_global)
        self.batchs["state_native"].append(states_atten[0])
        self.batchs["state_token"].append(states_atten[1])
        self.batchs["self_msl_token"].append(msl_token_self)
        self.batchs["bandit_msl_token"].append(msl_token_bandit)
        self.batchs["state_id"].append(state_id_one_hot)

        self.batchs["hor"].append([actions[0][2], actions[1][2]])
        self.batchs["ver"].append([actions[0][3], actions[1][3]])
        self.batchs["shot"].append([actions[0][0], actions[1][0]])
        self.batchs["target"].append([actions[0][1], actions[1][1]])
        self.batchs["v_c"].append([actions[0][4], actions[1][4]])
        self.batchs["nn_c"].append([actions[0][5], actions[1][5]])

        self.batchs["hor_one_hots"].append([actions[0][6], actions[1][6]])
        self.batchs["ver_one_hots"].append([actions[0][7], actions[1][7]])

        self.batchs["type"].append([100, 100])  # state machine

        if actions[0][0] != -1 or actions[1][0] != -1:
            shot_data = [states_global, states_atten[0], states_atten[1], msl_token_self,
                         msl_token_bandit, state_id_one_hot, actions]
            self.batchs["shot_data"].append(shot_data)

        hor_masks = []
        ver_masks = []
        shot_masks = []
        target_masks = []

        hor_mask_len = len(
            env.action_interface["AMS"][self.side * env.red]["BotManeuver"]["horizontal_cmd"]["mask"])
        ver_mask_len = len(
            env.action_interface["AMS"][self.side * env.red]["BotManeuver"]["vertical_cmd"]["mask"])
        hor_mask = [1] * hor_mask_len
        ver_mask = [1] * ver_mask_len

        t_mask_len = env.red + env.blue - 1
        s_mask_len = env.blue + 1
        s_mask = [1] * s_mask_len
        t_mask = [1] * t_mask_len

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if self.dones[i]:
                hor_masks.append(hor_mask)
                ver_masks.append(ver_mask)
                shot_masks.append(s_mask)
                target_masks.append(t_mask)
            else:
                hor_masks.append(env.action_interface["AMS"][i]["BotManeuver"]["horizontal_cmd"]["mask"])
                ver_masks.append(env.action_interface["AMS"][i]["BotManeuver"]["vertical_cmd"]["mask"])
                shot_masks.append([1] + env.action_interface["AMS"][i]["action_shoot_target"]["mask"])

                cur_target_mask = []
                aircraft_i_id_mask = env.action_interface["AMS"][i]["BotManeuver"]["maneuver_target"]["mask"]
                for id in range(env.red + env.blue):
                    mask_id = (id + self.side * env.red) % (env.red + env.blue)
                    if mask_id == i:  # this aircraft
                        pass
                    else:
                        cur_target_mask.append(aircraft_i_id_mask[mask_id])
                target_masks.append(cur_target_mask)

        self.batchs["hor_masks"].append(hor_masks)
        self.batchs["ver_masks"].append(ver_masks)
        self.batchs["target_masks"].append(target_masks)
        self.batchs["shot_masks"].append(shot_masks)

    def after_step_for_sample(self, env):
        for i in range(env.red + env.blue):
            if env.state_interface["AMS"][i]["alive"]["value"] + 0.1 < 1.0:
                self.dones[i] = True

    def before_step_for_train(self, env):
        self.before_step_for_sample(env)  # add before step for sample

    def after_step_for_train(self, env):
        self.after_step_for_sample(env)

        done = env.done
        mask = 0 if done else 1
        self.batchs["mask"].append([mask, mask])

        mask_solo_done = [state_method_independ.get_aircraft_available(env, i) for i in
                          range(self.side * env.red, env.red + self.side * env.blue)]

        self.batchs["mask_solo_done"].append(mask_solo_done)

        solo_sum_rewards = [
            reward_method_independ.get_ith_aircraft_reward(env, i, self.side,
                                                           rewards_hyperparam_dict=self.rewards_hyperparam_dict) for i
            in
            range(self.side * env.red, env.red + self.side * env.blue)]
        solo_rewards = [solo_sum_rewards[i] - self.current_episode_solo_reward[i] for i in
                        range(env.blue if self.side else env.red)]
        self.current_episode_solo_reward = solo_sum_rewards

        other_sum_mean_reward = [
            reward_method_independ.get_mean_team_reward_without_ith_aircraft(env, i, self.side,
                                                                             rewards_hyperparam_dict=self.rewards_hyperparam_dict)
            for
            i in range(self.side * env.red, env.red + self.side * env.blue)]
        other_mean_reward = [other_sum_mean_reward[i] - self.current_episode_other_mean_reward[i] for i in
                             range(env.blue if self.side else env.red)]
        self.current_episode_other_mean_reward = other_sum_mean_reward

        shape_reward = [reward_shaping.get_ith_aircraft_shaped_reward(env, i, self.side, self.interval, [], [],
                                                                      rewards_hyperparam_dict=self.rewards_hyperparam_dict)
                        for i in
                        range(self.side * env.red, env.red + self.side * env.blue)]

        final_rewards = [
            (1 - Config.Independ.pro_factor_alpha) * (solo_rewards[i] + shape_reward[i]) +
            Config.Independ.pro_factor_alpha * other_mean_reward[i]
            for i in range(env.blue if self.side else env.red)]
        self.batchs["final_rewards"].append(final_rewards)

    def train(self, batchs):
        pass

    def get_batchs(self):
        return {"batchs": self.batchs}

    def get_interval(self):
        return self.interval

    def print_train_log(self):
        pass


if __name__ == "__main__":
    env = Config.env
    env.reset()

    from agents.state_machine_agent.YSQ.simple_agent import Simple
    red_agent = MachineBird(0)
    blue_agent = Simple()

    for _ in range(1000):
        red_agent.after_reset(Config.env, "red")
        blue_agent.after_reset(Config.env, "blue")
        Config.env.reset()
        for i in range(1000):
            print("step", i)

            red_agent.before_step_for_sample(Config.env)
            blue_agent.before_step_for_sample(Config.env)

            Config.env.step()

            if Config.env.done:
                break


    # def general_offense_macro_tactic(self, env, self_id, target_id):
    #     if r > MagicNumber.upper_far_offense_range:  # >80
    #         if msl_remain == 4:
    #             if height >= MagicNumber.max_offense_height:
    #                 macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #                 shoot = 0
    #             else:
    #                 macro_tactic = macro_tactics["offense"]["intercept_climb"]
    #                 shoot = 0
    #         elif 0 < msl_remain < 4:
    #             macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #             shoot = 0
    #         else:
    #             macro_tactic, shoot = self.no_missile_remain_macro_tactic(env, self_id)
    #     elif MagicNumber.lower_far_offense_range < r < MagicNumber.upper_far_offense_range:  # [60, 80]
    #         if msl_remain == 4:
    #             macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["crank_50"])
    #         elif 0 < msl_remain < 4:
    #             macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #             shoot = 0
    #         else:
    #             macro_tactic, shoot = self.no_missile_remain_macro_tactic(env, self_id)
    #     elif MagicNumber.lower_mid_offense_range < r < MagicNumber.lower_far_offense_range:  # [40, 60]
    #         team_msl_fuck_flag, flying_time, r_dot = self.check_team_missile_fuck_some_target(env, target_id)
    #         if team_msl_fuck_flag and ((flying_time > 10 and r_dot > MagicNumber.min_missile_r_dot) or flying_time <= 10):
    #             macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #             shoot = 0
    #         else:
    #             if msl_remain > 1:
    #                 if degrees(abs(ta)) >= MagicNumber.threshold_launch_ta:
    #                     macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
    #                 else:
    #                     macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #                     shoot = 0
    #             elif msl_remain == 1:
    #                 macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #                 shoot = 0
    #             else:
    #                 macro_tactic, shoot = self.no_missile_remain_macro_tactic(env, self_id)
    #     else:
    #         team_msl_fuck_flag, flying_time, r_dot = self.check_team_missile_fuck_some_target(env, target_id)
    #         if team_msl_fuck_flag and ((flying_time > 10 and r_dot > MagicNumber.min_missile_r_dot) or flying_time <= 10):
    #             macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #             shoot = 0
    #         else:
    #             if msl_remain > 1:
    #                 if degrees(abs(ta)) >= MagicNumber.threshold_launch_ta:
    #                     macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
    #                 else:
    #                     macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #                     shoot = 0
    #             elif msl_remain == 1:
    #                 if degrees(abs(ta)) > MagicNumber.threshold_face_close_offense_ta:
    #                     if r < MagicNumber.face_close_offense_range:
    #                         macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
    #                     else:
    #                         macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #                         shoot = 0
    #                 elif degrees(abs(ta)) < MagicNumber.threshold_tail_close_offense_ta:
    #                     if r < MagicNumber.tail_close_offense_range:
    #                         macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
    #                     else:
    #                         macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #                         shoot = 0
    #                 else:
    #                     if r < MagicNumber.close_offense_range:
    #                         macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
    #                     else:
    #                         macro_tactic = macro_tactics["offense"]["intercept_cata"]
    #                         shoot = 0
    #             else:
    #                 macro_tactic, shoot = self.no_missile_remain_macro_tactic(env, self_id)

    # def no_missile_remain_target_assign(self, env, self_id, bandit_alive_num, target_list: list):
    #     friend_msl_remain = int(env.state_interface["AMS"][5 - self_id if self.side else 1 - self_id]["AAM_remain"]["value"])
    #     if bandit_alive_num == 2 or bandit_alive_num == 0:
    #         target = self.threat_sort(env, self_id)[0]
    #     elif bandit_alive_num == 1:
    #         target = self.get_only_alive_bandit_index(env)
    #     else:
    #         target = (1 - self.side) * env.red#TODO
    #
    #     if friend_msl_remain > 0:
    #         team_msl_target_list = self.get_team_missile_work_target(env)
    #         if len(team_msl_target_list) == 0:
    #             target_list.append(5 - self_id if self.side else 1 - self_id)
    #         elif len(team_msl_target_list) == 1:
    #             target_list.append(team_msl_target_list[0])
    #         else:
    #             target_list.append(target)
    #     else:
    #         target_list.append(target)
    #
    #     return target_list

    # def target_assign():
    #     target_list = []
    #     team_alive_num, bandit_alive_num = self.check_alive_num(env)
    #     if top_tactics == "all_defense":
    #         for i in range(self.side * env.red, env.red + self.side * env.blue):
    #             alive = env.state_interface["AMS"][i]["alive"]["value"]
    #             if int(alive + 0.1) == 1:
    #                 _, _, _, threat_source, _ = self.check_missile_threat(env, i)
    #                 target_list.append(threat_source)
    #             else:
    #                 target_list.append((1 - self.side) * env.red)#TODO
    #     elif top_tactics == "offense_and_defense":
    #         if team_alive_num == 2 and bandit_alive_num == 1:
    #             target = self.get_only_alive_bandit_index(env)
    #             target_list = [target for _ in range(env.blue if self.side else env.red)]
    #         elif team_alive_num == 2 and (bandit_alive_num == 2 or bandit_alive_num == 0):
    #             for i in range(self.side * env.red, env.red + self.side * env.blue):
    #                 threat_flag, _, _, threat_source, _ = self.check_missile_threat(env, i)
    #                 # escape_flag, threat_source = self.check_escape_available(env, i)
    #                 if threat_flag:
    #                     target_list.append(threat_source)
    #                 else:
    #                     msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
    #                     if msl_remain == 0:  # 没弹逃或接力
    #                         target_list = self.no_missile_remain_target_assign(env, i, bandit_alive_num, target_list)
    #                     else:
    #                         target_list.append(self.threat_sort(env, i)[0])
    #         else:
    #             for i in range(self.side * env.red, env.red + self.side * env.blue):
    #                 target_list.append((1 - self.side) * env.red)#TODO
    #     else:
    #         if (team_alive_num == 1 or team_alive_num == 2) and bandit_alive_num == 1:  # 1v1 2v1
    #             target = self.get_only_alive_bandit_index(env)
    #             for i in range(self.side * env.red, env.red + self.side * env.blue):
    #                 alive = env.state_interface["AMS"][i]["alive"]["value"]
    #                 if int(alive + 0.1) == 1:
    #                     msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
    #                     if msl_remain == 0:  # 没弹逃或接力
    #                         target_list = self.no_missile_remain_target_assign(env, i, bandit_alive_num, target_list)
    #                     else:
    #                         target_list.append(target)
    #                 else:
    #                     target_list.append((1 - self.side) * env.red)#TODO
    #         elif team_alive_num == 1 and bandit_alive_num == 2:  # 1v2
    #             for i in range(self.side * env.red, env.red + self.side * env.blue):
    #                 alive = env.state_interface["AMS"][i]["alive"]["value"]
    #                 if int(alive + 0.1) == 1:
    #                     msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
    #                     if msl_remain <= 1:
    #                         target_list.append(self.threat_sort(env, i)[0])
    #                     else:  # 一目标打一枚
    #                         msl_target_list = self.get_self_missile_work_target(env, i)
    #                         if len(msl_target_list) == 0 or len(msl_target_list) == 2:
    #                             target_list.append(self.threat_sort(env, i)[0])
    #                         else:
    #                             target_list.append(1 - msl_target_list[0] if self.side else 5 - msl_target_list[0])
    #                 else:
    #                     target_list.append((1 - self.side) * env.red)#TODO
    #         elif team_alive_num == 2 and bandit_alive_num == 2:  # 2v2
    #             for i in range(self.side * env.red, env.red + self.side * env.blue):
    #                 msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
    #                 if msl_remain == 4:  # 第一枚选近的或队友目标的另一个
    #                     if len(target_list) == 0:
    #                         target_list.append(self.threat_sort(env, i)[0])
    #                     else:
    #                         if target_list[0] == (1 - self.side) * env.red or target_list[0] == env.red + (1 - self.side) * env.blue - 1:
    #                             target_list.append(1 - target_list[0] if self.side else 5 - target_list[0])
    #                         else:
    #                             _, threat_level, _, threat_source, _ = self.check_missile_threat(env, i)
    #                             if threat_level == "low" or threat_level == "medium":
    #                                 target_list.append(threat_source)
    #                             else:
    #                                 target_list.append(self.threat_sort(env, i)[0])
    #                 elif 0 < msl_remain < 4:  # 第一枚制导结束再选近的
    #                     _, threat_level, _, threat_source, _ = self.check_missile_threat(env, i)
    #                     if threat_level == "low" or threat_level == "medium":
    #                         target_list.append(threat_source)
    #                     else:
    #                         if msl_remain == 3 and len(self.get_self_missile_work_target(env, i)) != 0:
    #                             target_list.append(self.get_self_missile_work_target(env, i)[0])
    #                         else:
    #                             target_list.append(self.threat_sort(env, i)[0])
    #                 else:  # 没弹逃或接力
    #                     target_list = self.no_missile_remain_target_assign(env, i, bandit_alive_num, target_list)
    #         else:  # 1v0 2v0
    #             for i in range(self.side * env.red, env.red + self.side * env.blue):
    #                 alive = env.state_interface["AMS"][i]["alive"]["value"]
    #                 if int(alive + 0.1) == 1:
    #                     target_list.append(self.threat_sort(env, i)[0])
    #                 else:
    #                     target_list.append((1 - self.side) * env.red)#TODO
    #
    #     for i in range(self.side * env.red, env.red + self.side * env.blue):  # high priority
    #         rel_i = i - env.red if self.side else i
    #         friend_id = 5 - i if self.side else 1 - i
    #         near_bandit_id = self.threat_sort(env, i)[0]
    #         if int(env.state_interface["AMS"][near_bandit_id]["alive"]["value"] + 0.1) < 1:
    #             near_bandit_id = 1 - near_bandit_id if self.side else 5 - near_bandit_id
    #         r = env.state_interface["AMS"][i]["relative_observation"][near_bandit_id]["r"]["value"]
    #         r_team = env.state_interface["AMS"][i]["relative_observation"][friend_id]["r"]["value"]
    #         msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
    #         near_ta = env.state_interface["AMS"][i]["relative_observation"][near_bandit_id]["TA"]["value"]
    #         near_r = env.state_interface["AMS"][i]["relative_observation"][near_bandit_id]["r"]["value"]
    #         near_msl_remain = int(env.state_interface["AMS"][near_bandit_id]["AAM_remain"]["value"])
    #         near_rtr = env.state_interface["AMS"][near_bandit_id]["attack_zone_list"][rel_i]["Rtr"]["value"]
    #         threat_flag, _, _, _, _ = self.check_missile_threat(env, i)
    #         work_target_list = self.get_self_missile_work_target(env, i)
    #
    #         if msl_remain > 0:
    #             if near_r <= min(near_rtr, MagicNumber.min_escape_range):
    #                 if near_msl_remain > 0:
    #                     if degrees(abs(near_ta)) > 120:
    #                         target_list[rel_i] = near_bandit_id
    #             else:
    #                 if not threat_flag:
    #                     if r > MagicNumber.min_safe_range:
    #                         if r_team < MagicNumber.min_team_range:
    #                             if len(work_target_list) == 0:
    #                                 target_list[rel_i] = 5 - i if self.side else 1 - i
