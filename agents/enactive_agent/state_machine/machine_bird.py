from agents.single_agent.state_machine.machine_config import semantic_maneuver_default, top_tactics, macro_tactics, MagicNumber
from agents.single_agent.state_machine.combo import Abort, Evade, Intercept, Format, Banzai
from framwork.agent_base import AgentBase
from train.config import Config

from math import pi, degrees, acos
from copy import deepcopy


class MachineBird(AgentBase):
    def __init__(self):
        self.side = None
        self.interval = 1
        self.maneuver_model = ["F22bot"]
        self.combo_executing = [None]
        self.threat_level = ["none"]
        self.bandit_launch_range = [[]]
        self.bandit_launch_direction = [[]]
        self.bandit_msl_valid_flag = [[]]
        self.default_target = []
        self.default_clockwise = []
        self.combo = [
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

    def after_reset(self, env, side):
        if side == "red":
            self.side = 0
        elif side == "blue":
            self.side = 1

        self.combo_executing = [None]
        self.threat_level = ["none"]
        self.bandit_launch_range = [[]]
        self.bandit_launch_direction = [[]]
        self.bandit_msl_valid_flag = [[]]
        self.default_target = []
        self.default_clockwise = []
        for i in range(1):
            self.combo[i]["defense"]["abort"].reset()
            self.combo[i]["defense"]["evade"].reset()
            self.combo[i]["offense"]["intercept"].reset()
            self.combo[i]["offense"]["banzai"].reset()

        self.default_target.append((1 - self.side) * env.red)
        self.default_clockwise.append(-1)

        for i in range(1):
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
        threat_source = (1 - self.side) * env.red  # TODO
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            bandit_aircraft = env.state_interface["AMS"][j]
            for msl_i, msl in enumerate(bandit_aircraft["SMS"]):
                if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying #
                    if int(msl["target_index"]["value"] + 0.1) == self_id - self.side * env.red:
                        threat_flag = True
                        if msl["TGO"]["value"] <= min_msl_tgo:
                            min_msl_tgo = msl["TGO"]["value"]
                            threat_source = j
                            msl_r = self.get_threat_missile_range(env, self_id, j, msl_i)
                            launch_range = self.get_some_missile_launch_range(env, j, msl_i)
                            launch_direction = self.get_some_missile_launch_direction(env, j, msl_i)
                            flying_time = msl["flying_time"]["value"]

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
        else:
            threat_level = "none"
            advice_tactic = None

        return threat_flag, threat_level, advice_tactic, threat_source, launch_range, launch_direction  # threat_source: absolute index

    def check_escape_available(self, env, self_id):
        escape_flag = False
        rel_i = self_id - env.red if self.side else self_id
        threat_flag, threat_level, _, threat_source, _, _ = self.check_missile_threat(env, self_id)
        if threat_flag:
            near_bandit_i = 1 - self.side
            near_r = env.state_interface["AMS"][self_id]["relative_observation"][near_bandit_i]["r"]["value"]
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
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:  # or work_missile_num > 0:
                team_alive_num += 1
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            alive = env.state_interface["AMS"][j]["alive"]["value"]
            if int(alive + 0.1) == 1:
                bandit_alive_num += 1
        return team_alive_num, bandit_alive_num

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

    """running script"""
    def rule_script(self, env, semantic_maneuver_list):
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            rel_j = j if self.side else j - env.red
            if len(self.bandit_launch_range[rel_j]) < 4:
                self.get_bandit_launch_range(env, j)
            if len(self.bandit_launch_direction[rel_j]) < 4:
                self.get_bandit_launch_direction(env, j)

        select_top_tactics = self.top_rule(env)
        if select_top_tactics == "all_defense":
            maneuver_list, shoot_list, target_list = self.all_defense_rule(env, semantic_maneuver_list)
        else:
            maneuver_list, shoot_list, target_list = self.all_offense_rule(env, semantic_maneuver_list)

        return maneuver_list, shoot_list, target_list

    """top tactic selection"""
    def top_rule(self, env):
        all_defense_flag = self.judge_all_defense(env)
        if all_defense_flag:
            select_top_tactics = top_tactics["all_defense"]
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

    """three top tactics"""
    def all_defense_rule(self, env, semantic_maneuver_list):
        target_list = [1 - self.side]
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
                elif macro_tactic == "climb":
                    maneuver = self.format(env, i, shoot, macro_tactic, semantic_maneuver_list[rel_i])
                else:
                    maneuver = semantic_maneuver_list[rel_i]["maintain"]

                maneuver_list.append(maneuver)
                tactic_l.append(macro_tactic)
            else:
                maneuver_list.append(semantic_maneuver_list[rel_i]["maintain"])#TODO
                tactic_l.append(macro_tactics["other"]["maintain"])

        # print(tactic_l)
        return maneuver_list, shoot_list, target_list

    def all_offense_rule(self, env, semantic_maneuver_list):
        target_list = [1 - self.side]
        macro_tactics_list, shoot_list = self.all_offense_macro_tactics_assign(env, target_list)
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
                elif macro_tactic == "banzai":
                    maneuver, shoot = self.banzai(env, i, macro_tactic, semantic_maneuver_list[rel_i])
                    shoot_list[i - env.red if self.side else i] = shoot
                elif macro_tactic == "climb":
                    maneuver = self.format(env, i, shoot, macro_tactic, semantic_maneuver_list[rel_i])
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
                macro_tactic, shoot = self.high_defense_macro_tactic(env, i, threat_source, threat_level, advice_tactic)
                macro_tactics_list.append(macro_tactic)
                shoot_list.append(shoot)
            else:
                macro_tactics_list.append(macro_tactics["other"]["maintain"])  # TODO
                shoot_list.append(0)  # TODO

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

    def general_offense_macro_tactic(self, env, self_id, target_id):  # absolute index
        team_alive_num, bandit_alive_num = self.check_alive_num(env)
        if not ((target_id < env.red) ^ (self_id < env.red)):
            target_id = 1 - self.side
        r = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["r"]["value"]
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
            near_bandit_id = 1 - self.side
            near_ta = env.state_interface["AMS"][self_id]["relative_observation"][near_bandit_id]["TA"]["value"]
            near_r = env.state_interface["AMS"][self_id]["relative_observation"][near_bandit_id]["r"]["value"]
            near_msl_remain = int(env.state_interface["AMS"][near_bandit_id]["AAM_remain"]["value"])

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
                        if height < MagicNumber.min_offense_height:
                            macro_tactic = macro_tactics["offense"]["climb"]
                            shoot = 0

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
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        threat_flag, _, _, _, _, _ = self.check_missile_threat(env, self_id)
        maneuver, combo_complete_flag = self.combo[rel_i]["others"]["format"].execute(height, shoot, threat_flag,
                                                                                      semantic_maneuver)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver

    def banzai(self, env, self_id, macro_tactic: str, semantic_maneuver):
        rel_i = self_id - env.red if self.side else self_id
        near_bandit_id = 1 - self.side
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

    def before_step_for_sample(self, env):
        semantic_maneuver_list = [deepcopy(semantic_maneuver_default)]
        maneuver_list, shoot_list, target_list = self.rule_script(env, semantic_maneuver_list)

        for i in range(self.side * env.red, env.red + self.side * env.red):
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:
                maneuver = maneuver_list[i - env.red if self.side else i]
                shoot = shoot_list[i - env.red if self.side else i]
                target = target_list[i - env.red if self.side else i]

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
                env.action_interface["AMS"][i]["action_shoot_target"]["value"] = shoot - 1

        env.action_interface["AMS"][0]["action_shoot_predict_list"][0]["shoot_predict"]["value"] = 0
        env.action_interface["AMS"][1]["action_shoot_predict_list"][0]["shoot_predict"]["value"] = 0

    def after_step_for_sample(self, env):
        pass

    def before_step_for_train(self, env):
        self.before_step_for_sample(env)  # add before step for sample

    def after_step_for_train(self, env):
        pass

    def train(self, batchs):
        pass

    def get_batchs(self):
        pass

    def get_interval(self):
        return self.interval

    def print_train_log(self):
        pass


if __name__ == "__main__":
    env = Config.env
    env.reset()

    from agents.state_machine_agent.YSQ.simple_agent import Simple
    red_agent = MachineBird()
    blue_agent = MachineBird()

    for _ in range(1000):
        red_agent.after_reset(Config.env, "red")
        blue_agent.after_reset(Config.env, "blue")
        Config.env.reset()
        for i in range(1000):
            print("step", i)
            if i == 20:
                a=1

            red_agent.before_step_for_sample(Config.env)
            blue_agent.before_step_for_sample(Config.env)

            Config.env.step()

            if Config.env.done:
                break