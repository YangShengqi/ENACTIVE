from agents.state_machine_agent.YSQ.absolute_defense_version.machine_config import semantic_maneuver, MagicNumber
from math import pi, degrees


class MacroTactic:
    def __init__(self):
        self.pointer = 0
        self.step = 0
        self.combo_complete_flag = False

    def execute(self, *args):
        raise NotImplementedError("Please Implement this method")

    def reset(self):
        self.combo_complete_flag = False
        self.pointer = 0
        self.step = 0


class Evade(MacroTactic):
    def __init__(self):
        super().__init__()
        self.primitive_action_num = {
            "abort_no_dive": 2,
            "abort_dive_25_1k": 3,
            "abort_dive_25_2k": 3,
            "split_s": 2
        }
        self.pointer = 0
        self.step = 0
        self.init_height = None
        self.init_chi = None
        self.init_theta = None
        self.combo_complete_flag = False
        self.primitive_action = {
            "abort_no_dive": [semantic_maneuver["abort_no_dive"], semantic_maneuver["level_accelerate_escape"]],
            "abort_dive_25_1k": [semantic_maneuver["abort_dive_25"], semantic_maneuver["out"],
                                 semantic_maneuver["level_accelerate_escape"]],
            "abort_dive_25_2k": [semantic_maneuver["abort_dive_25"], semantic_maneuver["out"],
                                 semantic_maneuver["level_accelerate_escape"]],
            "split_s": [semantic_maneuver["split_s"], semantic_maneuver["level_accelerate_escape"]]
        }

    def execute(self, height, ao, chi, theta, terminate_flag: bool, maneuver_style: str):
        complete_flag = self.combo_complete_flag
        if maneuver_style == "abort_dive_25_1k":
            if self.pointer == 0 and self.step == 0:
                self.init_height = height
            if self.pointer == 0:
                delta_height = abs(self.init_height - height)
                if delta_height < MagicNumber.small_abort_dive_25_delta_height:
                    maneuver = self.primitive_action["abort_dive_25_1k"][self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                delta_ao = degrees(abs(abs(ao) - pi))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    maneuver = self.primitive_action["abort_dive_25_1k"][self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 2:
                maneuver = self.primitive_action["abort_dive_25_1k"][self.pointer]
                self.step += 1
            if terminate_flag:
                complete_flag = True
                self.reset()
        elif maneuver_style == "abort_dive_25_2k":
            if self.pointer == 0 and self.step == 0:
                self.init_height = height
            if self.pointer == 0:
                delta_height = abs(self.init_height - height)
                if delta_height < MagicNumber.large_abort_dive_25_delta_height:
                    maneuver = self.primitive_action["abort_dive_25_2k"][self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                delta_ao = degrees(abs(abs(ao) - pi))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    maneuver = self.primitive_action["abort_dive_25_2k"][self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 2:
                maneuver = self.primitive_action["abort_dive_25_2k"][self.pointer]
                self.step += 1
            if terminate_flag:
                complete_flag = True
                self.reset()
        elif maneuver_style == "abort_no_dive":
            if self.pointer == 0:
                delta_ao = degrees(abs(abs(ao) - pi))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    maneuver = self.primitive_action["abort_no_dive"][self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                maneuver = self.primitive_action["abort_no_dive"][self.pointer]
                self.step += 1
            if terminate_flag:
                complete_flag = True
                self.reset()
        elif maneuver_style == "split_s":
            if self.pointer == 0 and self.step == 0:
                self.init_chi = chi
                self.init_theta = theta
            if self.pointer == 0:
                delta_chi = degrees(abs(abs(self.init_chi) - abs(chi)))
                theta = degrees(abs(theta))
                if delta_chi < MagicNumber.threshold_splits_delta_chi or theta > MagicNumber.threshold_splits_theta:
                    maneuver = self.primitive_action["split_s"][self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                maneuver = self.primitive_action["split_s"][self.pointer]
                self.step += 1
            if terminate_flag:
                complete_flag = True
                self.reset()
        else:
            print("there's no ", maneuver_style, " in Evade maneuver style.")

        return maneuver, complete_flag


class Intercept(MacroTactic):
    def __init__(self):
        super().__init__()
        self.primitive_action_num = {
            "cata": 1,
            "level": 1,
            "climb": 2
        }
        self.pointer = 0
        self.step = 0
        self.init_height = None
        self.combo_complete_flag = False
        self.primitive_action = {
            "cata": [semantic_maneuver["intercept_cata"]],
            "level": [semantic_maneuver["intercept_level"]],
            "climb": [semantic_maneuver["intercept_climb_20"], semantic_maneuver["maintain"]]
        }

    def execute(self, height, threat_flag: bool, maneuver_style: str):
        complete_flag = self.combo_complete_flag
        if maneuver_style == "intercept_cata":
            maneuver = self.primitive_action["cata"][self.pointer]
            self.step += 1
            self.pointer += 1
            if self.pointer == self.primitive_action_num["cata"]:
                complete_flag = True
                self.reset()
        elif maneuver_style == "intercept_level":
            maneuver = self.primitive_action["level"][self.pointer]
            self.step += 1
            self.pointer += 1
            if self.pointer == self.primitive_action_num["level"]:
                complete_flag = True
                self.reset()
        elif maneuver_style == "intercept_climb":
            if self.pointer == 0 and self.step == 0:
                self.init_height = height
            if self.pointer == 0:
                delta_height = abs(height - self.init_height)
                if delta_height < MagicNumber.min_intercept_climb_20_delta_height and height < MagicNumber.max_offense_height:
                    maneuver = self.primitive_action["climb"][self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                maneuver = self.primitive_action["climb"][self.pointer]
                self.step += 1
                self.pointer += 1
            if self.pointer == self.primitive_action_num["climb"] or threat_flag:
                complete_flag = True
                self.reset()
        else:
            print("there's no ", maneuver_style, " in abort maneuver style.")

        return maneuver, complete_flag


class Crank(MacroTactic):
    def __init__(self):
        super().__init__()
        self.primitive_action_num = 1
        self.pointer = 0
        self.step = 0
        self.combo_complete_flag = False
        self.primitive_action = {
            "crank": [semantic_maneuver["crank_50"]]
        }

    def execute(self, threat_flag: bool):
        complete_flag = self.combo_complete_flag
        maneuver = self.primitive_action["crank"][self.pointer]
        if self.step < MagicNumber.min_crank_time:
            self.step += 1
        else:
            self.pointer += 1
        if self.pointer == self.primitive_action_num or threat_flag:
            complete_flag = True
            self.reset()

        return maneuver, complete_flag


class Climb(MacroTactic):
    def __init__(self):
        super().__init__()
        self.primitive_action_num = 2
        self.pointer = 0
        self.step = 0
        self.combo_complete_flag = False
        self.init_height = None
        self.primitive_action = {
            "climb": [semantic_maneuver["climb_20"], semantic_maneuver["in"]]
        }

    def execute(self, height, threat_flag: bool):
        complete_flag = self.combo_complete_flag
        if self.pointer == 0 and self.step == 0:
            self.init_height = height
        if self.pointer == 0:
            delta_height = abs(height - self.init_height)
            if delta_height < MagicNumber.min_climb_20_delta_height and height < MagicNumber.max_offense_height:
                maneuver = self.primitive_action["climb"][self.pointer]
                self.step += 1
            else:
                self.pointer += 1
        if self.pointer == 1:
            maneuver = self.primitive_action["climb"][self.pointer]
            self.step += 1
            self.pointer += 1
        if self.pointer == self.primitive_action_num or threat_flag:
            complete_flag = True
            self.reset()

        return maneuver, complete_flag

