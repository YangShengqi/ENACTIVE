from agents.state_machine_agent.YSQ.absolute_defense_version.machine_config import semantic_maneuver, top_tactics, macro_tactics, \
    MagicNumber
from agents.state_machine_agent.YSQ.absolute_defense_version.combo import Evade, Intercept, Crank, Climb
from state_method import state_method_independ
from reward_method import reward_method_independ, reward_shaping
from framwork.agent_base import AgentBase
from train.config import Config
from utils.math import index_to_one_hot

from math import degrees, acos
import numpy as np


class MachineBird(AgentBase):
    def __init__(self, reward_hyperparam_dict):
        self.side = None
        self.interval = 1
        self.maneuver_model = ["F22semantic", "F22semantic"]
        self.combo_executing_flag = [None, None]
        self.combo_executing = [None, None]
        self.combo = [{
            "defense": {
                "evade": Evade()
            },
            "offense": {
                "intercept": Intercept(),
                "crank": Crank(),
                "climb": Climb()
            }
        },
            {
                "defense": {
                    "evade": Evade()
                },
                "offense": {
                    "intercept": Intercept(),
                    "crank": Crank(),
                    "climb": Climb()
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
        if side == "red":
            self.side = 0
        elif side == "blue":
            self.side = 1
        for i in range(2):
            self.combo[i]["defense"]["evade"].reset()
            self.combo[i]["offense"]["intercept"].reset()
            self.combo[i]["offense"]["crank"].reset()
            self.combo[i]["offense"]["climb"].reset()

        self.dones = []
        for aircraft_i in range(env.red + env.blue):
            if int(env.state_interface["AMS"][aircraft_i]["alive"]["value"] + 0.1):
                self.dones.append(False)
            else:
                self.dones.append(True)

        self.current_episode_solo_reward = [0, 0]
        self.current_episode_other_mean_reward = [0, 0]

    """功能函数"""

    def get_self_missile_work_target(self, env, self_id) -> list:
        missile_target_list = []
        missile_set = env.state_interface["AMS"][self_id]["SMS"]
        for missile in missile_set:
            if missile["state"]["value"] == 2 or missile["state"]["value"] == 3:
                missile_target_list.append(int(missile["target_index"]["value"]) + (1 - self.side) * env.red)
        out = list(set(missile_target_list))
        return out  # absolute index

    def get_team_missile_work_target(self, env) -> list:
        missile_target_list = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            missile_set = env.state_interface["AMS"][i]["SMS"]
            for missile in missile_set:
                if missile["state"]["value"] == 2 or missile["state"]["value"] == 3:
                    missile_target_list.append(int(missile["target_index"]["value"]) + (1 - self.side) * env.red)
        out = list(set(missile_target_list))
        return out  # absolute index

    def check_team_missile_work(self, env) -> bool:
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            missile_set = env.state_interface["AMS"][i]["SMS"]
            for missile in missile_set:
                if missile["state"]["value"] == 2 or missile["state"]["value"] == 3:
                    return True
        return False

    def check_team_missile_fuck_some_target(self, env, target_id) -> bool:
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            missile_set = env.state_interface["AMS"][i]["SMS"]
            for missile in missile_set:
                if missile["state"]["value"] == 2 or missile["state"]["value"] == 3:
                    cur_target_id = int(missile["target_index"]["value"]) + (1 - self.side) * env.red
                    if cur_target_id == target_id:
                        return True
        return False

    def check_missile_threat(self, env, self_id):
        threat_flag = False
        min_threat_missile_tgo = env.state_interface["AMS"][0]["SMS"][0]["TGO"]["max"]
        threat_source = (1 - self.side) * env.red  # TODO
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            bandit_aircraft = env.state_interface["AMS"][j]
            for msl in bandit_aircraft["SMS"]:
                if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying #
                    if int(msl["target_index"]["value"] + 0.1) == self_id - self.side * env.red:
                        if msl["TGO"]["value"] <= min_threat_missile_tgo:
                            min_threat_missile_tgo = msl["TGO"]["value"]
                            threat_source = j

        if min_threat_missile_tgo < MagicNumber.min_threat_missile_tgo:
            threat_flag = True
        # if self_id == 0:
        #     print(min_threat_missile_tgo)
        return threat_flag, min_threat_missile_tgo, threat_source  # threat_source: absolute index

    def check_abort_terminate(self, env, self_id):
        terminate_flag = False
        min_threat_missile_tas = env.state_interface["AMS"][0]["SMS"][0]["TAS_m"]["min"]
        for j in range((1 - self.side) * env.red, env.red + (1 - self.side) * env.blue):
            bandit_aircraft = env.state_interface["AMS"][j]
            for msl in bandit_aircraft["SMS"]:
                if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying #
                    if int(msl["target_index"]["value"] + 0.1) == self_id - self.side * env.red:
                        if msl["TAS_m"]["value"] >= min_threat_missile_tas:
                            min_threat_missile_tas = msl["TAS_m"]["value"]

        if min_threat_missile_tas < MagicNumber.min_threat_missile_tas:
            terminate_flag = True
        return terminate_flag

    def check_escape_available(self, env, self_id):
        escape_flag = False
        threat_flag, _, threat_source = self.check_missile_threat(env, self_id)
        if threat_flag:
            threat_source_r = env.state_interface["AMS"][self_id]["relative_observation"][threat_source]["r"]["value"]
            escape_flag = threat_flag and (threat_source_r > MagicNumber.min_escape_range)
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

    def cal_two_los_angle(self, env, target_id):
        i_0 = self.side * env.red
        i_1 = i_0 + 1
        coord_i_0 = [env.state_interface["AMS"][i_0]["Xg_0"]["value"], env.state_interface["AMS"][i_0]["Xg_1"]["value"]]
        coord_i_1 = [env.state_interface["AMS"][i_1]["Xg_0"]["value"], env.state_interface["AMS"][i_1]["Xg_1"]["value"]]
        coord_tar = [env.state_interface["AMS"][target_id]["Xg_0"]["value"],
                     env.state_interface["AMS"][target_id]["Xg_1"]["value"]]
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
            k = env.state_interface["AMS"][self_id]["Vg_1"]["value"] / env.state_interface["AMS"][self_id]["Vg_0"][
                "value"]
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

    """running script"""

    def rule_script(self, env):
        select_top_tactics = self.top_rule(env)
        # maneuver_list, shoot_list, target_list = self.all_defense_rule(env)
        # maneuver_list, shoot_list, target_list = self.offense_and_defense_rule(env)
        if select_top_tactics == "all_defense":
            maneuver_list, shoot_list, target_list = self.all_defense_rule(env)
        elif select_top_tactics == "offense_and_defense":
            maneuver_list, shoot_list, target_list = self.offense_and_defense_rule(env)
        else:
            maneuver_list, shoot_list, target_list = self.all_offense_rule(env)

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
            threat_flag, _, threat_source = self.check_missile_threat(env, i)
            if not threat_flag:
                flag = False
                break
            else:
                threat_source_r = env.state_interface["AMS"][i]["relative_observation"][threat_source]["r"]["value"]
                escape_flag = threat_flag and (threat_source_r > MagicNumber.min_escape_range)
                if not escape_flag:
                    flag = False
                    break

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

    """three top tactics"""

    def all_defense_rule(self, env):
        target_list = self.target_assign(env, top_tactics["all_defense"])
        macro_tactics_list, shoot_list = self.all_defense_macro_tactics_assign(env)
        # target_list = [0, 1]
        # macro_tactics_list = [macro_tactics["defense"]["evade"], macro_tactics["defense"]["evade"]]
        maneuver_list = []

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:
                target = target_list[i - env.red if self.side else i]
                macro_tactic = macro_tactics_list[i - env.red if self.side else i]
                if self.combo_executing[i - env.red if self.side else i] is not None:
                    macro_tactic = self.combo_executing[i - env.red if self.side else i]

                if macro_tactic in macro_tactics["defense"].keys() and macro_tactic != "escape":
                    maneuver = self.evade(env, i, target, macro_tactic)
                elif macro_tactic == "intercept_cata" or macro_tactic == "intercept_level" or macro_tactic == "intercept_climb":
                    maneuver = self.intercept(env, i, macro_tactic)
                elif macro_tactic == "crank_50":
                    maneuver = self.crank(env, i, macro_tactic)
                elif macro_tactic == "climb":
                    maneuver = self.climb(env, i, macro_tactic)
                elif macro_tactic == "escape":
                    maneuver = self.escape(env, i)
                else:
                    maneuver = semantic_maneuver["maintain"]

                maneuver_list.append(maneuver)
            else:
                maneuver_list.append(semantic_maneuver["maintain"])  # TODO

        return maneuver_list, shoot_list, target_list

    def offense_and_defense_rule(self, env):
        target_list = self.target_assign(env, top_tactics["offense_and_defense"])
        macro_tactics_list, shoot_list = self.offense_and_defense_macro_tactics_assign(env, target_list)
        # target_list = [0, 1]
        # macro_tactics_list = [macro_tactics["offense"]["intercept_cata"], macro_tactics["offense"]["intercept_cata"]]
        maneuver_list = []

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            target = target_list[i - env.red if self.side else i]
            macro_tactic = macro_tactics_list[i - env.red if self.side else i]
            if self.combo_executing[i - env.red if self.side else i] is not None:
                macro_tactic = self.combo_executing[i - env.red if self.side else i]

            if macro_tactic in macro_tactics["defense"].keys() and macro_tactic != "escape":
                maneuver = self.evade(env, i, target, macro_tactic)
            elif macro_tactic == "intercept_cata" or macro_tactic == "intercept_level" or macro_tactic == "intercept_climb":
                maneuver = self.intercept(env, i, macro_tactic)
            elif macro_tactic == "crank_50":
                maneuver = self.crank(env, i, macro_tactic)
            elif macro_tactic == "climb":
                maneuver = self.climb(env, i, macro_tactic)
            elif macro_tactic == "escape":
                maneuver = self.escape(env, i)
            else:
                maneuver = semantic_maneuver["maintain"]

            maneuver_list.append(maneuver)

        return maneuver_list, shoot_list, target_list

    def all_offense_rule(self, env):
        target_list = self.target_assign(env, top_tactics["all_offense"])
        macro_tactics_list, shoot_list = self.all_offense_macro_tactics_assign(env, target_list)
        shoot_available_list = self.shoot_available_assign(env, target_list)
        for n in range(len(shoot_list)):
            if shoot_list[n] is not None:
                shoot_list[n] = int(shoot_list[n] and shoot_available_list[n])
        maneuver_list = []

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:
                target = target_list[i - env.red if self.side else i]
                macro_tactic = macro_tactics_list[i - env.red if self.side else i]
                if self.combo_executing[i - env.red if self.side else i] is not None:
                    macro_tactic = self.combo_executing[i - env.red if self.side else i]

                if macro_tactic in macro_tactics["defense"].keys() and macro_tactic != "escape":
                    maneuver = self.evade(env, i, target, macro_tactic)
                elif macro_tactic == "intercept_cata" or macro_tactic == "intercept_level" or macro_tactic == "intercept_climb":
                    maneuver = self.intercept(env, i, macro_tactic)
                elif macro_tactic == "crank_50":
                    maneuver = self.crank(env, i, macro_tactic)
                elif macro_tactic == "climb":
                    maneuver = self.climb(env, i, macro_tactic)
                elif macro_tactic == "escape":
                    maneuver = self.escape(env, i)
                else:
                    maneuver = semantic_maneuver["maintain"]
                maneuver_list.append(maneuver)
            else:
                maneuver_list.append(semantic_maneuver["maintain"])  # TODO

        return maneuver_list, shoot_list, target_list

    """target assignment"""

    def target_assign(self, env, top_tactics: str) -> list:
        target_list = []
        team_alive_num, bandit_alive_num = self.check_alive_num(env)
        if top_tactics == "all_defense":
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                alive = env.state_interface["AMS"][i]["alive"]["value"]
                if int(alive + 0.1) == 1:
                    _, _, threat_source = self.check_missile_threat(env, i)
                    target_list.append(threat_source)
                else:
                    target_list.append((1 - self.side) * env.red)  # TODO
        elif top_tactics == "offense_and_defense":
            if team_alive_num == 2 and bandit_alive_num == 1:
                target = self.get_only_alive_bandit_index(env)
                target_list = [target for _ in range(env.blue if self.side else env.red)]
            elif team_alive_num == 2 and (bandit_alive_num == 2 or bandit_alive_num == 0):
                for i in range(self.side * env.red, env.red + self.side * env.blue):
                    escape_flag, threat_source = self.check_escape_available(env, i)
                    if escape_flag:
                        target_list.append(threat_source)
                    else:
                        msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
                        if msl_remain == 0:  # 没弹逃或接力
                            target_list = self.no_missile_remain_target_assign(env, i, bandit_alive_num, target_list)
                        else:
                            target_list.append(self.threat_sort(env, i)[0])
            else:
                for i in range(self.side * env.red, env.red + self.side * env.blue):
                    target_list.append((1 - self.side) * env.red)  # TODO
        else:
            if (team_alive_num == 1 or team_alive_num == 2) and bandit_alive_num == 1:  # 1v1 2v1
                target = self.get_only_alive_bandit_index(env)
                for i in range(self.side * env.red, env.red + self.side * env.blue):
                    alive = env.state_interface["AMS"][i]["alive"]["value"]
                    if int(alive + 0.1) == 1:
                        msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
                        if msl_remain == 0:  # 没弹逃或接力
                            target_list = self.no_missile_remain_target_assign(env, i, bandit_alive_num, target_list)
                        else:
                            target_list.append(target)
                    else:
                        target_list.append((1 - self.side) * env.red)  # TODO
            elif team_alive_num == 1 and bandit_alive_num == 2:  # 1v2
                for i in range(self.side * env.red, env.red + self.side * env.blue):
                    alive = env.state_interface["AMS"][i]["alive"]["value"]
                    if int(alive + 0.1) == 1:
                        msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
                        if msl_remain <= 1:
                            target_list.append(self.threat_sort(env, i)[0])
                        else:  # 一目标打一枚
                            msl_target_list = self.get_self_missile_work_target(env, i)
                            if len(msl_target_list) == 0 or len(msl_target_list) == 2:
                                target_list.append(self.threat_sort(env, i)[0])
                            else:
                                target_list.append(1 - msl_target_list[0] if self.side else 5 - msl_target_list[0])
                    else:
                        target_list.append((1 - self.side) * env.red)  # TODO
            elif team_alive_num == 2 and bandit_alive_num == 2:  # 2v2
                for i in range(self.side * env.red, env.red + self.side * env.blue):
                    msl_remain = int(env.state_interface["AMS"][i]["AAM_remain"]["value"])
                    if msl_remain == 4:  # 第一枚选近的或队友目标的另一个
                        if len(target_list) == 0:
                            target_list.append(self.threat_sort(env, i)[0])
                        else:
                            if target_list[0] == (1 - self.side) * env.red or target_list[0] == env.red + (
                                    1 - self.side) * env.blue - 1:
                                target_list.append(1 - target_list[0] if self.side else 5 - target_list[0])
                            else:
                                target_list.append(self.threat_sort(env, i)[0])
                    elif 0 < msl_remain < 4:  # 第一枚制导结束再选近的
                        if msl_remain == 3 and len(self.get_self_missile_work_target(env, i)) != 0:
                            target_list.append(self.get_self_missile_work_target(env, i)[0])
                        else:
                            target_list.append(self.threat_sort(env, i)[0])
                    else:  # 没弹逃或接力
                        target_list = self.no_missile_remain_target_assign(env, i, bandit_alive_num, target_list)
            else:  # 1v0 2v0
                for i in range(self.side * env.red, env.red + self.side * env.blue):
                    alive = env.state_interface["AMS"][i]["alive"]["value"]
                    if int(alive + 0.1) == 1:
                        target_list.append(self.threat_sort(env, i)[0])
                    else:
                        target_list.append((1 - self.side) * env.red)  # TODO

        return target_list

    def no_missile_remain_target_assign(self, env, self_id, bandit_alive_num, target_list: list):
        friend_msl_remain = int(
            env.state_interface["AMS"][5 - self_id if self.side else 1 - self_id]["AAM_remain"]["value"])
        if bandit_alive_num == 2 or bandit_alive_num == 0:
            target = self.threat_sort(env, self_id)[0]
        elif bandit_alive_num == 1:
            target = self.get_only_alive_bandit_index(env)
        else:
            target = (1 - self.side) * env.red  # TODO

        if friend_msl_remain > 0:
            team_msl_target_list = self.get_team_missile_work_target(env)
            if len(team_msl_target_list) == 0:
                target_list.append(5 - self_id if self.side else 1 - self_id)
            elif len(team_msl_target_list) == 1:
                target_list.append(team_msl_target_list[0])
            else:
                target_list.append(target)
        else:
            target_list.append(target)

        return target_list

    """tactics assignment"""

    def all_defense_macro_tactics_assign(self, env):
        macro_tactics_list = []
        shoot_list = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:
                _, min_threat_msl_tgo, _ = self.check_missile_threat(env, i)
                macro_tactic, shoot = self.defense_macro_tactic(env, i, min_threat_msl_tgo)
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
                _, min_threat_msl_tgo, _ = self.check_missile_threat(env, i)
                macro_tactic, shoot = self.defense_macro_tactic(env, i, min_threat_msl_tgo)
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
        friend_msl_remain = int(
            env.state_interface["AMS"][5 - self_id if self.side else 1 - self_id]["AAM_remain"]["value"])
        if friend_msl_remain > 0:
            team_msl_target_list = self.get_team_missile_work_target(env)
            if len(team_msl_target_list) == 0:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
            elif len(team_msl_target_list) == 1:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
            else:
                macro_tactic = macro_tactics["defense"]["escape"]
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

    def defense_macro_tactic(self, env, self_id, min_threat_msl_tgo):  # absolute index
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        tas = env.state_interface["AMS"][self_id]["TAS"]["value"]
        if height >= MagicNumber.mid_splits_height:
            # if min_threat_msl_tgo <= MagicNumber.min_threat_missile_splits_tgo:
            if 5000 <= height < 6500:
                if tas <= 300:
                    semantic_maneuver["split_s"]["vel_cmd"] = 300
                    semantic_maneuver["split_s"]["ny_cmd"] = 5
                    macro_tactic = macro_tactics["defense"]["split_s"]
                elif 300 < tas <= 400:
                    semantic_maneuver["split_s"]["vel_cmd"] = 350
                    semantic_maneuver["split_s"]["ny_cmd"] = 8
                    macro_tactic = macro_tactics["defense"]["split_s"]
                else:
                    semantic_maneuver["abort_dive_25"]["ny_cmd"] = 8
                    macro_tactic = macro_tactics["defense"]["abort_dive_25_2k"]
            elif 6500 <= height < 7500:
                if tas <= 300:
                    semantic_maneuver["split_s"]["vel_cmd"] = 400
                    semantic_maneuver["split_s"]["ny_cmd"] = 5
                    macro_tactic = macro_tactics["defense"]["split_s"]
                else:
                    semantic_maneuver["split_s"]["vel_cmd"] = 400
                    semantic_maneuver["split_s"]["ny_cmd"] = 8
                    macro_tactic = macro_tactics["defense"]["split_s"]
            elif 7500 <= height < 10000:
                if tas <= 400:
                    semantic_maneuver["split_s"]["vel_cmd"] = 400
                    semantic_maneuver["split_s"]["ny_cmd"] = 6
                    macro_tactic = macro_tactics["defense"]["split_s"]
                else:
                    semantic_maneuver["split_s"]["vel_cmd"] = 400
                    semantic_maneuver["split_s"]["ny_cmd"] = 8
                    macro_tactic = macro_tactics["defense"]["split_s"]
            else:
                semantic_maneuver["split_s"]["vel_cmd"] = 400
                semantic_maneuver["split_s"]["ny_cmd"] = 8
                macro_tactic = macro_tactics["defense"]["split_s"]
            # else:
            #     if tas >= 400:
            #         semantic_maneuver["abort_dive_25"]["ny_cmd"] = 7
            #     else:
            #         semantic_maneuver["abort_dive_25"]["ny_cmd"] = 5.5
            #     macro_tactic = macro_tactics["defense"]["abort_dive_25_2k"]
        elif MagicNumber.min_splits_height <= height < MagicNumber.mid_splits_height:  # [4000, 5000]
            if tas <= 350:
                semantic_maneuver["split_s"]["vel_cmd"] = 300
                semantic_maneuver["split_s"]["ny_cmd"] = 8
                macro_tactic = macro_tactics["defense"]["split_s"]
            else:
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 8
                macro_tactic = macro_tactics["defense"]["abort_dive_25_2k"]
        elif MagicNumber.min_large_abort_dive_25_height <= height < MagicNumber.min_splits_height:  # [3000, 4000]
            if tas >= 400:
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 8
            else:
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 5.5
            macro_tactic = macro_tactics["defense"]["abort_dive_25_2k"]
        elif MagicNumber.min_small_abort_no_dive_height <= height < MagicNumber.min_large_abort_dive_25_height:  # [2000, 3000]
            if tas >= 400:
                semantic_maneuver["abort_dive_25"]["vel_cmd"] = 350
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 8
            else:
                semantic_maneuver["abort_dive_25"]["vel_cmd"] = 300
                semantic_maneuver["abort_dive_25"]["ny_cmd"] = 5.5
            macro_tactic = macro_tactics["defense"]["abort_dive_25_1k"]
        else:
            macro_tactic = macro_tactics["defense"]["abort_no_dive"]
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
        r = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["r"]["value"]
        ta = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["TA"]["value"]
        ao = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["AO"]["value"]
        msl_remain = int(env.state_interface["AMS"][self_id]["AAM_remain"]["value"])
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        # Rpi = env.state_interface["AMS"][self_id]["attack_zone_list"][target_id if self.side else target_id - env.red]["Rpi"]["value"]
        # Rtr = env.state_interface["AMS"][self_id]["attack_zone_list"][target_id if self.side else target_id - env.red]["Rtr"]["value"]
        if r > MagicNumber.upper_far_offense_range:  # >80
            if msl_remain == 4:
                if height >= MagicNumber.max_offense_height:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"]
                    shoot = 0
                else:
                    macro_tactic = macro_tactics["offense"]["intercept_climb"]
                    shoot = 0
            elif 0 < msl_remain < 4:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
                shoot = 0
            else:
                macro_tactic, shoot = self.no_missile_remain_macro_tactic(env, self_id)
        elif MagicNumber.lower_far_offense_range < r < MagicNumber.upper_far_offense_range:  # [55, 80]
            if msl_remain == 4:
                macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["crank_50"])
            elif 0 < msl_remain < 4:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
                shoot = 0
            else:
                macro_tactic, shoot = self.no_missile_remain_macro_tactic(env, self_id)
        elif MagicNumber.lower_mid_offense_range < r < MagicNumber.lower_far_offense_range:  # [40, 55]
            team_msl_fuck_cur_target_flag = self.check_team_missile_fuck_some_target(env, target_id)
            if team_msl_fuck_cur_target_flag:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
                shoot = 0
            else:
                if msl_remain > 1:
                    if degrees(abs(ta)) >= MagicNumber.threshold_launch_ta:
                        macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
                    else:
                        macro_tactic = macro_tactics["offense"]["intercept_cata"]
                        shoot = 0
                elif msl_remain == 1:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"]
                    shoot = 0
                else:
                    macro_tactic, shoot = self.no_missile_remain_macro_tactic(env, self_id)
        else:
            team_msl_fuck_cur_target_flag = self.check_team_missile_fuck_some_target(env, target_id)
            if team_msl_fuck_cur_target_flag:
                macro_tactic = macro_tactics["offense"]["intercept_cata"]
                shoot = 0
            else:
                if msl_remain > 1:
                    if degrees(abs(ta)) >= MagicNumber.threshold_launch_ta:
                        macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
                    else:
                        macro_tactic = macro_tactics["offense"]["intercept_cata"]
                        shoot = 0
                elif msl_remain == 1:
                    if degrees(abs(ta)) > MagicNumber.threshold_face_close_offense_ta:
                        if r < MagicNumber.face_close_offense_range:
                            macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
                        else:
                            macro_tactic = macro_tactics["offense"]["intercept_cata"]
                            shoot = 0
                    elif degrees(abs(ta)) < MagicNumber.threshold_tail_close_offense_ta:
                        if r < MagicNumber.tail_close_offense_range:
                            macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
                        else:
                            macro_tactic = macro_tactics["offense"]["intercept_cata"]
                            shoot = 0
                    else:
                        if r < MagicNumber.close_offense_range:
                            macro_tactic, shoot = self.ready_to_shoot(ao, macro_tactics["offense"]["intercept_cata"])
                        else:
                            macro_tactic = macro_tactics["offense"]["intercept_cata"]
                            shoot = 0
                else:
                    macro_tactic, shoot = self.no_missile_remain_macro_tactic(env, self_id)

        if msl_remain > 0:
            threat_flag, _, _ = self.check_missile_threat(env, self_id)
            if threat_flag and r < MagicNumber.min_escape_range:  # high priority
                self_missile_work_target_list = self.get_self_missile_work_target(env, self_id)
                if target_id in self_missile_work_target_list:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"]
                    shoot = 0
                else:
                    macro_tactic = macro_tactics["offense"]["intercept_cata"]
                    shoot = 1
            else:
                if r > MagicNumber.lower_mid_offense_range and height < MagicNumber.min_offense_height:
                    macro_tactic = macro_tactics["offense"]["climb"]
                    shoot = 0

        return macro_tactic, shoot

    """macro tactics"""

    def evade(self, env, self_id, target_id, macro_tactic: str):
        rel_i = self_id - env.red if self.side else self_id
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        ao = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["AO"]["value"]
        chi = env.state_interface["AMS"][self_id]["attg_2"]["value"]
        theta = env.state_interface["AMS"][self_id]["attg_1"]["value"]
        terminate_flag = self.check_abort_terminate(env, self_id)
        maneuver, combo_complete_flag = self.combo[rel_i]["defense"]["evade"].execute(height, ao, chi, theta,
                                                                                      terminate_flag, macro_tactic)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver

    def intercept(self, env, self_id, macro_tactic: str):
        rel_i = self_id - env.red if self.side else self_id
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        threat_flag, _, _ = self.check_missile_threat(env, self_id)
        maneuver, combo_complete_flag = self.combo[rel_i]["offense"]["intercept"].execute(height, threat_flag,
                                                                                          macro_tactic)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver

    def crank(self, env, self_id, macro_tactic: str):
        rel_i = self_id - env.red if self.side else self_id
        threat_flag, _, _ = self.check_missile_threat(env, self_id)
        maneuver, combo_complete_flag = self.combo[rel_i]["offense"]["crank"].execute(threat_flag)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver

    def climb(self, env, self_id, macro_tactic: str):
        rel_i = self_id - env.red if self.side else self_id
        height = -env.state_interface["AMS"][self_id]["Xg_2"]["value"]
        threat_flag, _, _ = self.check_missile_threat(env, self_id)
        maneuver, combo_complete_flag = self.combo[rel_i]["offense"]["climb"].execute(height, threat_flag)
        if not combo_complete_flag:
            self.combo_executing[rel_i] = macro_tactic
        else:
            self.combo_executing[rel_i] = None

        return maneuver

    def escape(self, env, self_id):
        team_msl_target_list = self.get_team_missile_work_target(env)
        if len(team_msl_target_list) > 0:
            maneuver = semantic_maneuver["intercept_cata"]
        else:
            if env.state_interface["AMS"][self_id]["out_of_border_time"]["value"] > 0:
                maneuver = semantic_maneuver["crank_50"]
            else:
                border_distance = self.cal_border_distance(env, self_id)
                if border_distance > MagicNumber.border_range:
                    maneuver = semantic_maneuver["out"]
                else:
                    maneuver = semantic_maneuver["notch"]

        return maneuver

    def ready_to_shoot(self, ao, maneuver_when_shoot: str):
        if degrees(abs(ao)) > MagicNumber.threshold_launch_ao:
            macro_tactic = macro_tactics["offense"]["intercept_cata"]
            shoot = 0
        else:
            macro_tactic = maneuver_when_shoot
            shoot = 1

        return macro_tactic, shoot

    def before_step_for_sample(self, env):
        maneuver_list, shoot_list, target_list = self.rule_script(env)

        actions = []
        current_step_available = []
        for i in range(self.side * env.red, env.red + self.side * env.red):
            current_step_available.append(state_method_independ.get_aircraft_available(env, i))
            alive = env.state_interface["AMS"][i]["alive"]["value"]
            if int(alive + 0.1) == 1:
                maneuver = maneuver_list[i - env.red if self.side else i]
                shoot = shoot_list[i - env.red if self.side else i]
                target = target_list[i - env.red if self.side else i]

                env.action_interface["AMS"][i]["SemanticManeuver"]["combat_mode"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["flag_after_burning"]["value"] = 1
                env.action_interface["AMS"][i]["SemanticManeuver"]["horizontal_cmd"]["value"] = maneuver[
                    "horizontal_cmd"]
                env.action_interface["AMS"][i]["SemanticManeuver"]["vertical_cmd"]["value"] = maneuver["vertical_cmd"]
                env.action_interface["AMS"][i]["SemanticManeuver"]["vel_cmd"]["value"] = maneuver["vel_cmd"]
                env.action_interface["AMS"][i]["SemanticManeuver"]["ny_cmd"]["value"] = maneuver["ny_cmd"]
                env.action_interface["AMS"][i]["SemanticManeuver"]["maneuver_target"]["value"] = target
                env.action_interface["AMS"][i]["action_target"]["value"] = target
                env.action_interface["AMS"][i]["SemanticManeuver"]["clockwise_cmd"] = 0
                if not ((target < env.red) ^ (i < env.red)):
                    env.action_interface["AMS"][i]["action_shoot_target"]["value"] = -1
                else:
                    if shoot:
                        env.action_interface["AMS"][i]["action_shoot_target"]["value"] = target - (
                                1 - self.side) * env.red
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
                env.action_interface["AMS"][i]["SemanticManeuver"]["combat_mode"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["flag_after_burning"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["horizontal_cmd"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["vertical_cmd"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["vel_cmd"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["ny_cmd"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["maneuver_target"]["value"] = (1 - self.side) * env.red
                env.action_interface["AMS"][i]["action_target"]["value"] = (1 - self.side) * env.red
                env.action_interface["AMS"][i]["action_shoot_target"]["value"] = -1
                env.action_interface["AMS"][i]["SemanticManeuver"]["clockwise_cmd"] = 0

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
            env.action_interface["AMS"][self.side * env.red]["SemanticManeuver"]["horizontal_cmd"]["mask"])
        ver_mask_len = len(
            env.action_interface["AMS"][self.side * env.red]["SemanticManeuver"]["vertical_cmd"]["mask"])
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
                hor_masks.append(env.action_interface["AMS"][i]["SemanticManeuver"]["horizontal_cmd"]["mask"])
                ver_masks.append(env.action_interface["AMS"][i]["SemanticManeuver"]["vertical_cmd"]["mask"])
                shot_masks.append([1] + env.action_interface["AMS"][i]["action_shoot_target"]["mask"])

                cur_target_mask = []
                aircraft_i_id_mask = env.action_interface["AMS"][i]["SemanticManeuver"]["maneuver_target"]["mask"]
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

        # done = env.done
        # mask = 0 if done else 1
        # self.batchs["mask"].append([mask, mask])
        #
        # mask_solo_done = [state_method_independ.get_aircraft_available(env, i) for i in
        #                   range(self.side * env.red, env.red + self.side * env.blue)]
        #
        # self.batchs["mask_solo_done"].append(mask_solo_done)
        #
        # solo_sum_rewards = [
        #     reward_method_independ.get_ith_aircraft_reward(env, i, self.side, rewards_hyperparam_dict=self.rewards_hyperparam_dict) for i in
        #     range(self.side * env.red, env.red + self.side * env.blue)]
        # solo_rewards = [solo_sum_rewards[i] - self.current_episode_solo_reward[i] for i in
        #                 range(env.blue if self.side else env.red)]
        # self.current_episode_solo_reward = solo_sum_rewards
        #
        # other_sum_mean_reward = [
        #     reward_method_independ.get_mean_team_reward_without_ith_aircraft(env, i, self.side,
        #                                                                      rewards_hyperparam_dict=self.rewards_hyperparam_dict) for
        #     i in range(self.side * env.red, env.red + self.side * env.blue)]
        # other_mean_reward = [other_sum_mean_reward[i] - self.current_episode_other_mean_reward[i] for i in
        #                      range(env.blue if self.side else env.red)]
        # self.current_episode_other_mean_reward = other_sum_mean_reward
        #
        # shape_reward = [reward_shaping.get_ith_aircraft_shaped_reward(env, i, self.side, self.interval, [], [],
        #                                                               rewards_hyperparam_dict=self.rewards_hyperparam_dict) for i in
        #                 range(self.side * env.red, env.red + self.side * env.blue)]
        #
        # final_rewards = [
        #     (1 - Config.Independ.pro_factor_alpha) * (solo_rewards[i] + shape_reward[i]) +
        #     Config.Independ.pro_factor_alpha * other_mean_reward[i]
        #     for i in range(env.blue if self.side else env.red)]
        # self.batchs["final_rewards"].append(final_rewards)

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
    # env.reset()

    red_agent = MachineBird(None)
    blue_agent = MachineBird(None)

    for _ in range(100):
        red_agent.after_reset(Config.env, "red")
        blue_agent.after_reset(Config.env, "blue")
        # Config.env.random_init()
        # Config.env.rsi_init()
        Config.env.reset()
        for i in range(1000):
            print("step", i)

            red_agent.before_step_for_train(Config.env)
            blue_agent.before_step_for_train(Config.env)

            Config.env.step()

            red_agent.after_step_for_train(Config.env)
            blue_agent.after_step_for_train(Config.env)

            if Config.env.done:
                break
