from framwork.agent_base import AgentBase
from agents.state_machine_agent.YSQ.advanced_version.machine_config import semantic_maneuver_default, MagicNumber
import math


class Simple(AgentBase):
    def __init__(self):
        self.side = None
        self.interval = 1
        self.maneuver_model = ["F22semantic", "F22semantic"]

    def after_reset(self, env, side):
        if side == "red":
            self.side = 0
        elif side == "blue":
            self.side = 1

    def read_state(self, env, self_id, target_id):  # absolute index
        if self_id < env.red:
            side = 0
        else:
            side = 1
        cur_state_set = env.state_interface["AMS"][self_id]

        cur_r = cur_state_set["relative_observation"][target_id]["r"]["value"]
        cur_ao = cur_state_set["relative_observation"][target_id]["AO"]["value"]
        cur_ta = cur_state_set["relative_observation"][target_id]["TA"]["value"]
        cur_alive = cur_state_set["alive"]["value"]
        cur_rmax = cur_state_set["attack_zone_list"][target_id - (1 - side) * env.red]["Rmax"]["value"]
        cur_rpi = cur_state_set["attack_zone_list"][target_id - (1 - side) * env.red]["Rpi"]["value"]
        cur_rtr = cur_state_set["attack_zone_list"][target_id - (1 - side) * env.red]["Rtr"]["value"]
        cur_missile = [{"state": cur_state_set["SMS"][i]["state"]["value"]} for i in range(len(cur_state_set["SMS"]))]
        cur_rwr = cur_state_set["RWR_nailed"]["value"]
        cur_h = -cur_state_set["Xg_2"]["value"]
        cur_msl = cur_state_set["AAM_remain"]["value"]

        cur_state = {"r": cur_r, "AO": cur_ao, "TA": cur_ta, "alive": cur_alive, "Rmax": cur_rmax, "Rpi": cur_rpi,
                     "Rtr": cur_rtr, "missile": cur_missile, "RWR": cur_rwr, "h": cur_h, "msl_remain": cur_msl}

        return cur_state

    def check_missile_work(self, missile_set):
        for missile in missile_set:
            if missile["state"] == 2 or missile["state"] == 3:
                return True
        return False

    def check_missile_threat(self, env, self_id, k):
        threat_flag = False
        min_threat_missile_tgo = env.state_interface["AMS"][0]["SMS"][0]["TGO"]["max"]
        threat_source = None
        for j in range((1 - k) * env.red, env.red + (1 - k) * env.blue):
            bandit_aircraft = env.state_interface["AMS"][j]
            for msl in bandit_aircraft["SMS"]:
                if int(msl["state"]["value"] + 0.1) == 2 or int(msl["state"]["value"] + 0.1) == 3:  # msl flying #
                    if int(msl["target_index"]["value"] + 0.1) == self_id - k * env.red:
                        threat_flag = True
                        if msl["TGO"]["value"] <= min_threat_missile_tgo:
                            min_threat_missile_tgo = msl["TGO"]["value"]
                            threat_source = j

        return threat_flag, min_threat_missile_tgo, threat_source  # threat_source: absolute index

    def judge_defense(self, env, self_id, k):  # 判断防御条件
        defense_flag = False
        threat_flag, min_threat_missile_tgo, threat_source = self.check_missile_threat(env, self_id, k)
        if threat_flag:
            if min_threat_missile_tgo <= MagicNumber.mid_threat_level_msl_tgo:
                defense_flag = True
        return defense_flag, threat_source

    def rule_defense(self, threat_source, cur_state):  # 防御规则
        target = threat_source
        shoot = 0
        if cur_state["h"] <= 1500:
            maneuver = semantic_maneuver_default["out"]
        else:
            maneuver = semantic_maneuver_default["abort_dive_25"]

        return maneuver, shoot, target

    def rule_attack(self, cur_state):  # 进攻规则
        if cur_state["r"] > cur_state["Rmax"]:
            maneuver = semantic_maneuver_default["intercept_climb_20"]
            shoot = 0
        elif cur_state["Rtr"] < cur_state["r"] <= cur_state["Rmax"]:
            if abs(math.degrees(cur_state["TA"])) < MagicNumber.threshold_launch_ta or self.check_missile_work(cur_state["missile"]):
                maneuver = semantic_maneuver_default["intercept_cata"]
                shoot = 0
            else:
                maneuver = semantic_maneuver_default["intercept_cata"]
                if cur_state["msl_remain"] == 4:
                    shoot = 1
                else:
                    shoot = 0
        else:
            if self.check_missile_work(cur_state["missile"]):
                maneuver = semantic_maneuver_default["intercept_cata"]
                shoot = 0
            else:
                maneuver = semantic_maneuver_default["intercept_cata"]
                shoot = 1
                # if cur_state["msl_remain"] == 4:
                #     shoot = 1
                # else:
                #     shoot = 0

        return maneuver, shoot

    def rule_one_on_one(self, env, self_id, k, cur_state):  # self_id: absolute index
        defense_flag, threat_source = self.judge_defense(env, self_id, k)

        if defense_flag:
            maneuver, shoot, target = self.rule_defense(threat_source, cur_state)
        else:
            maneuver, shoot = self.rule_attack(cur_state)
            target = None

        return maneuver, shoot, target

    def before_step_for_sample(self, env):
        for i in range(self.side * env.red, env.red + self.side * env.red):
            if i == 0:
                target_id = 3
            elif i == 1:
                target_id = 2
            elif i == 2:
                target_id = 0
            else:
                target_id = 1
            if not env.state_interface["AMS"][target_id]["alive"]["value"]:
                if i == 0:
                    target_id = 2
                elif i == 1:
                    target_id = 3
                elif i == 2:
                    target_id = 1
                else:
                    target_id = 0

            cur_state = self.read_state(env, i, target_id)
            maneuver, shoot, target = self.rule_one_on_one(env, i, self.side, cur_state)
            if target is not None:
                target_id = target

            # if i == 3:
            #     r = env.state_interface["AMS"][i]["relative_observation"][target_id]["r"]["value"]
            #     if 20000 < r < 70000:
            #         if int(env.state_interface["AMS"][i]["AAM_remain"]["value"] + 0.1) == 4:
            #             shoot = 1
            #         else:
            #             shoot = 0
            #     else:
            #         shoot = 0
            # else:
            #     maneuver = semantic_maneuver_default["maintain"]
            #     shoot = 0

            env.action_interface["AMS"][i]["SemanticManeuver"]["combat_mode"]["value"] = 0
            env.action_interface["AMS"][i]["SemanticManeuver"]["flag_after_burning"]["value"] = 1
            env.action_interface["AMS"][i]["SemanticManeuver"]["horizontal_cmd"]["value"] = maneuver["horizontal_cmd"]
            env.action_interface["AMS"][i]["SemanticManeuver"]["vertical_cmd"]["value"] = maneuver["vertical_cmd"]
            env.action_interface["AMS"][i]["SemanticManeuver"]["vel_cmd"]["value"] = maneuver["vel_cmd"]
            env.action_interface["AMS"][i]["SemanticManeuver"]["ny_cmd"]["value"] = maneuver["ny_cmd"]
            env.action_interface["AMS"][i]["SemanticManeuver"]["clockwise_cmd"]["value"] = 0
            env.action_interface["AMS"][i]["SemanticManeuver"]["maneuver_target"]["value"] = target_id
            env.action_interface["AMS"][i]["action_target"]["value"] = target_id

            # env.action_interface["AMS"][i]["action_shoot_target"]["value"] = -1
            if shoot:
                env.action_interface["AMS"][i]["action_shoot_target"]["value"] = target_id - (1 - self.side) * env.red
            else:
                env.action_interface["AMS"][i]["action_shoot_target"]["value"] = -1

        for i in range(env.red + env.blue):
            if i < env.red:
                for j in range(env.blue):
                    env.action_interface["AMS"][i]["action_shoot_predict_list"][j]["shoot_predict"][
                        "value"] = 0
            else:
                for j in range(env.red):
                    env.action_interface["AMS"][i]["action_shoot_predict_list"][j]["shoot_predict"][
                        "value"] = 0

    def after_step_for_sample(self, env):
        pass

    def before_step_for_train(self, env):
        self.before_step_for_sample(env)  # add before step for sample
        pass

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