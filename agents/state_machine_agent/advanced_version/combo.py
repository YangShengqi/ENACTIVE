from agents.state_machine_agent.YSQ.advanced_version.machine_config import MagicNumber
from math import pi, degrees, radians
import random


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


class Abort(MacroTactic):
    def __init__(self):
        super().__init__()
        self.pointer = 0
        self.step = 0
        self.init_height = None
        self.init_chi = None
        self.init_theta = None
        self.clockwise = -1
        self.turn_direction = None
        self.combo_complete_flag = False

    def execute(self, height, ao, chi, theta, launch_direction, msl_turn_direction, threat_flag: bool,
                threat_level: str, maneuver_style: str, semantic_maneuver):
        complete_flag = self.combo_complete_flag
        if maneuver_style == "abort":
            self.primitive_action = [semantic_maneuver["out"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 2
            if self.pointer == 0:
                delta_ao = degrees(abs(abs(ao) - pi))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
            if (not threat_flag) or threat_level == "low" or threat_level == "danger":
                complete_flag = True
                self.reset()
        elif maneuver_style == "abort_dive_25_1k":
            self.primitive_action = [semantic_maneuver["out"], semantic_maneuver["out"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 3
            if self.pointer == 0 and self.step == 0:
                self.init_height = height
            if self.pointer == 0:
                delta_height = abs(self.init_height - height)
                if delta_height < MagicNumber.abort_dive_25_1k_delta_height:
                    self.primitive_action[self.pointer]["vertical_cmd"] = 3
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                delta_ao = degrees(abs(abs(ao) - pi))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 2:
                self.primitive_action[self.pointer]["base_direction"] = 10
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
            if (not threat_flag) or threat_level == "low" or threat_level == "danger":
                complete_flag = True
                self.reset()
        elif maneuver_style == "abort_dive_25_2k":
            self.primitive_action = [semantic_maneuver["out"], semantic_maneuver["out"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 3
            if self.pointer == 0 and self.step == 0:
                self.init_height = height
            if self.pointer == 0:
                delta_height = abs(self.init_height - height)
                if delta_height < MagicNumber.abort_dive_25_2k_delta_height:
                    self.primitive_action[self.pointer]["vertical_cmd"] = 3
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                delta_ao = degrees(abs(abs(ao) - pi))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 2:
                self.primitive_action[self.pointer]["base_direction"] = 10
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
            if (not threat_flag) or threat_level == "low" or threat_level == "danger":
                complete_flag = True
                self.reset()
        elif maneuver_style == "abort_dive_25_3k":
            self.primitive_action = [semantic_maneuver["out"], semantic_maneuver["out"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 3
            if self.pointer == 0 and self.step == 0:
                self.init_height = height
                self.turn_direction = msl_turn_direction
            if self.pointer == 0:
                delta_height = abs(self.init_height - height)
                if delta_height < MagicNumber.abort_dive_25_3k_delta_height:
                    if self.turn_direction > 0:  # turn right
                        self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                    else:
                        self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                    self.primitive_action[self.pointer]["vertical_cmd"] = 3
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                delta_ao = degrees(abs(abs(ao) - pi))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if self.turn_direction > 0:  # turn right
                        self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                    else:
                        self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                    self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 2:
                self.primitive_action[self.pointer]["base_direction"] = 10
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
            if (not threat_flag) or threat_level == "low" or threat_level == "danger":
                complete_flag = True
                self.reset()
        elif maneuver_style == "split_s":
            self.primitive_action = [semantic_maneuver["split_s"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                self.init_chi = chi
                self.init_theta = theta
            if self.pointer == 0:
                delta_chi = degrees(abs(abs(self.init_chi) - abs(chi)))
                theta = degrees(abs(theta))
                if delta_chi < MagicNumber.threshold_splits_delta_chi or theta > MagicNumber.threshold_splits_theta:
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                self.primitive_action[self.pointer]["base_direction"] = 10
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
            if (not threat_flag) or threat_level == "low" or threat_level == "danger":
                complete_flag = True
                self.reset()
        else:
            print("there's no ", maneuver_style, " in Evade maneuver style.")

        return maneuver, complete_flag


class Evade(MacroTactic):
    def __init__(self):
        super().__init__()
        self.pointer = 0
        self.step = 0
        self.init_height = None
        self.init_ao = None
        self.threshold_ao_0 = None
        self.threshold_ao_1 = None
        self.threshold_ao_2 = None
        self.combo_complete_flag = False
        self.clockwise = 0
        self.turn_direction = None
        self.snake_maintain_counter = 0
        self.snake_turn_counter = 0
        self.snake_clockwise_counter = 0

    def execute(self, height, tas, ao, launch_range, launch_direction, default_clockwise, msl_turn_direction,
                tar_turn_direction, threat_flag: bool, threat_level: str, maneuver_style: str, semantic_maneuver):
        complete_flag = self.combo_complete_flag
        if maneuver_style == "change_direction":
            self.primitive_action = [semantic_maneuver["crank_50"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                self.init_ao = ao
            if self.pointer == 0:
                if self.step < 15:
                    if self.init_ao > 0:
                        self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                    else:
                        self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                    if height < MagicNumber.max_offense_height:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 5
                    else:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
                self.pointer += 1
            if self.pointer == self.primitive_action_num or (not threat_flag):
                complete_flag = True
                self.reset()
        elif maneuver_style == "crank_30":
            self.primitive_action = [semantic_maneuver["crank_30"]]
            self.primitive_action_num = 1
            self.clockwise = random.choice([-1, 1])
            # self.clockwise = default_clockwise
            self.primitive_action[self.pointer]["clockwise_cmd"] = self.clockwise
            maneuver = self.primitive_action[self.pointer]
            self.step += 1
            self.pointer += 1
            if self.pointer == self.primitive_action_num:
                complete_flag = True
                self.reset()
        elif maneuver_style == "crank_50":
            self.primitive_action = [semantic_maneuver["crank_50"]]
            self.primitive_action_num = 1
            self.clockwise = random.choice([-1, 1])
            # self.clockwise = default_clockwise
            self.primitive_action[self.pointer]["clockwise_cmd"] = self.clockwise
            maneuver = self.primitive_action[self.pointer]
            self.step += 1
            self.pointer += 1
            if self.pointer == self.primitive_action_num:
                complete_flag = True
                self.reset()
        elif maneuver_style == "crank_50_dive_25":
            self.primitive_action = [semantic_maneuver["crank_50"], semantic_maneuver["crank_50"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                # self.clockwise = random.choice([-1, 1])
                if self.turn_direction is None:
                    if launch_range <= 45000:
                        self.turn_direction = msl_turn_direction
                    else:
                        self.turn_direction = tar_turn_direction
            if self.pointer == 0:
                delta_ao = degrees(abs(abs(ao) - radians(50)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if height >= 1500:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 3
                    else:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    if self.turn_direction > 0:  # turn right
                        self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                    else:
                        self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                if height >= 1500:
                    self.primitive_action[self.pointer]["vertical_cmd"] = 3
                else:
                    self.primitive_action[self.pointer]["vertical_cmd"] = 0
                if self.turn_direction > 0:  # turn right
                    self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                else:
                    self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                self.primitive_action[self.pointer]["base_direction"] = launch_direction
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
            if (not threat_flag) or threat_level == "low" or threat_level == "high" or threat_level == "danger":
                complete_flag = True
                self.turn_direction = None
                self.reset()
        elif maneuver_style == "notch_dive_25":
            self.primitive_action = [semantic_maneuver["crank_90"], semantic_maneuver["crank_90"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                # self.clockwise = random.choice([-1, 1])
                if self.turn_direction is None:
                    self.turn_direction = msl_turn_direction
            if self.pointer == 0:
                # delta_ao = degrees(abs(abs(ao) - radians(90)))
                # if delta_ao > MagicNumber.threshold_abort_delta_ao:
                if self.step < MagicNumber.min_notch_time:
                    if height >= 1000:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 3
                    else:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    if self.turn_direction > 0:  # turn right
                        self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                    else:
                        self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                    if tas > 380:
                        self.primitive_action[self.pointer]["ny_cmd"] = 8
                    else:
                        self.primitive_action[self.pointer]["ny_cmd"] = 6
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                if height >= 1000:
                    self.primitive_action[self.pointer]["vertical_cmd"] = 3
                else:
                    self.primitive_action[self.pointer]["vertical_cmd"] = 0
                if self.turn_direction > 0:  # turn right
                    self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                else:
                    self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                if tas > 380:
                    self.primitive_action[self.pointer]["ny_cmd"] = 8
                else:
                    self.primitive_action[self.pointer]["ny_cmd"] = 6
                self.primitive_action[self.pointer]["base_direction"] = launch_direction
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
            if (not threat_flag) or threat_level == "low" or threat_level == "danger":
                complete_flag = True
                self.turn_direction = None
                self.reset()
        elif maneuver_style == "circle":
            self.primitive_action_num = 4
            if self.pointer == 0 and self.step == 0:
                if self.turn_direction is None:
                    self.turn_direction = msl_turn_direction
                if self.turn_direction > 0:  # turn right
                    if 0 <= degrees(ao) < 120:
                        self.primitive_action = [semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = 0
                        self.threshold_ao_1 = -120
                        self.threshold_ao_2 = 120
                    elif -120 <= degrees(ao) < 0:
                        self.primitive_action = [semantic_maneuver["out_120"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = -120
                        self.threshold_ao_1 = 120
                        self.threshold_ao_2 = 0
                    else:
                        self.primitive_action = [semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = 120
                        self.threshold_ao_1 = 0
                        self.threshold_ao_2 = -120
                else:
                    if 0 <= degrees(ao) < 120:
                        self.primitive_action = [semantic_maneuver["out_120"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = 120
                        self.threshold_ao_1 = -120
                        self.threshold_ao_2 = 0
                    elif -120 <= degrees(ao) < 0:
                        self.primitive_action = [semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = 0
                        self.threshold_ao_1 = 120
                        self.threshold_ao_2 = -120
                    else:
                        self.primitive_action = [semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = -120
                        self.threshold_ao_1 = 0
                        self.threshold_ao_2 = 120

            if self.pointer == 0:
                delta_ao = degrees(abs(ao - radians(self.threshold_ao_0)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if self.turn_direction > 0:  # turn right
                        if 0 <= degrees(ao) < 120:
                            pass
                        elif -120 <= degrees(ao) < 0:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                        else:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                    else:
                        if 0 <= degrees(ao) < 120:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                        elif -120 <= degrees(ao) < 0:
                            pass
                        else:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                    if tas > 380:
                        self.primitive_action[self.pointer]["ny_cmd"] = 8
                    else:
                        self.primitive_action[self.pointer]["ny_cmd"] = 6
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                delta_ao = degrees(abs(ao - radians(self.threshold_ao_1)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if self.turn_direction > 0:  # turn right
                        if 0 <= degrees(ao) < 120:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                        elif -120 <= degrees(ao) < 0:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                        else:
                            pass
                    else:
                        if 0 <= degrees(ao) < 120:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                        elif -120 <= degrees(ao) < 0:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                        else:
                            pass
                    if tas > 380:
                        self.primitive_action[self.pointer]["ny_cmd"] = 8
                    else:
                        self.primitive_action[self.pointer]["ny_cmd"] = 6
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 2:
                delta_ao = degrees(abs(ao - radians(self.threshold_ao_2)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if self.turn_direction > 0:  # turn right
                        if 0 <= degrees(ao) < 120:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                        elif -120 <= degrees(ao) < 0:
                            pass
                        else:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                    else:
                        if 0 <= degrees(ao) < 120:
                            pass
                        elif -120 <= degrees(ao) < 0:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                        else:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                    if tas > 360:
                        self.primitive_action[self.pointer]["ny_cmd"] = 8
                    else:
                        self.primitive_action[self.pointer]["ny_cmd"] = 6
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 3:
                self.primitive_action[self.pointer]["ny_cmd"] = 6
                self.primitive_action[self.pointer]["base_direction"] = launch_direction
                maneuver = self.primitive_action[self.pointer]
                self.pointer += 1
                self.step += 1
            if self.pointer == self.primitive_action_num or (not threat_flag):
                complete_flag = True
                self.turn_direction = None
                self.reset()
        elif maneuver_style == "circle_dive_25":
            self.primitive_action_num = 4
            if self.pointer == 0 and self.step == 0:
                if self.turn_direction is None:
                    self.turn_direction = msl_turn_direction
                if self.turn_direction > 0:  # turn right
                    if 0 <= degrees(ao) < 120:
                        self.primitive_action = [semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = 0
                        self.threshold_ao_1 = -120
                        self.threshold_ao_2 = 120
                    elif -120 <= degrees(ao) < 0:
                        self.primitive_action = [semantic_maneuver["out_120"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = -120
                        self.threshold_ao_1 = 120
                        self.threshold_ao_2 = 0
                    else:
                        self.primitive_action = [semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = 120
                        self.threshold_ao_1 = 0
                        self.threshold_ao_2 = -120
                else:
                    if 0 <= degrees(ao) < 120:
                        self.primitive_action = [semantic_maneuver["out_120"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = 120
                        self.threshold_ao_1 = -120
                        self.threshold_ao_2 = 0
                    elif -120 <= degrees(ao) < 0:
                        self.primitive_action = [semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = 0
                        self.threshold_ao_1 = 120
                        self.threshold_ao_2 = -120
                    else:
                        self.primitive_action = [semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_level"],
                                                 semantic_maneuver["out_120"],
                                                 semantic_maneuver["intercept_cata"]]
                        self.threshold_ao_0 = -120
                        self.threshold_ao_1 = 0
                        self.threshold_ao_2 = 120

            if self.pointer == 0:
                delta_ao = degrees(abs(ao - radians(self.threshold_ao_0)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if height >= 3000:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 3
                    else:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    if self.turn_direction > 0:  # turn right
                        if 0 <= degrees(ao) < 120:
                            pass
                        elif -120 <= degrees(ao) < 0:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                        else:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                    else:
                        if 0 <= degrees(ao) < 120:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                        elif -120 <= degrees(ao) < 0:
                            pass
                        else:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                    if tas > 380:
                        self.primitive_action[self.pointer]["ny_cmd"] = 8
                    else:
                        self.primitive_action[self.pointer]["ny_cmd"] = 6
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                delta_ao = degrees(abs(ao - radians(self.threshold_ao_1)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if height >= 3000:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 3
                    else:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    if self.turn_direction > 0:  # turn right
                        if 0 <= degrees(ao) < 120:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                        elif -120 <= degrees(ao) < 0:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                        else:
                            pass
                    else:
                        if 0 <= degrees(ao) < 120:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                        elif -120 <= degrees(ao) < 0:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                        else:
                            pass
                    if tas > 380:
                        self.primitive_action[self.pointer]["ny_cmd"] = 8
                    else:
                        self.primitive_action[self.pointer]["ny_cmd"] = 6
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 2:
                delta_ao = degrees(abs(ao - radians(self.threshold_ao_2)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if height >= 3000:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 3
                    else:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    if self.turn_direction > 0:  # turn right
                        if 0 <= degrees(ao) < 120:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                        elif -120 <= degrees(ao) < 0:
                            pass
                        else:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                    else:
                        if 0 <= degrees(ao) < 120:
                            pass
                        elif -120 <= degrees(ao) < 0:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = 1
                        else:
                            self.primitive_action[self.pointer]["clockwise_cmd"] = -1
                    if tas > 380:
                        self.primitive_action[self.pointer]["ny_cmd"] = 8
                    else:
                        self.primitive_action[self.pointer]["ny_cmd"] = 6
                    self.primitive_action[self.pointer]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 3:
                self.primitive_action[self.pointer]["ny_cmd"] = 5
                self.primitive_action[self.pointer]["base_direction"] = launch_direction
                maneuver = self.primitive_action[self.pointer]
                self.pointer += 1
                self.step += 1
            if self.pointer == self.primitive_action_num or (not threat_flag):
                complete_flag = True
                self.turn_direction = None
                self.reset()
        elif maneuver_style == "snake_50":
            self.primitive_action = [semantic_maneuver["crank_50"], semantic_maneuver["crank_50"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                # self.clockwise = random.choice([-1, 1])
                self.clockwise = default_clockwise
            if self.pointer % 2 == 0:
                if self.snake_clockwise_counter % 2 == 0:
                    goal_ao = -1 * self.clockwise * 55
                else:
                    goal_ao = -1 * self.clockwise * (-55)
                delta_ao = degrees(abs(ao - radians(goal_ao)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if self.snake_clockwise_counter % 2 == 0:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = self.clockwise
                    else:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = -1 * self.clockwise
                    self.primitive_action[self.pointer % 2]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer % 2]
                    self.step += 1
                    self.snake_turn_counter += 1
                else:
                    self.pointer += 1
                    self.snake_maintain_counter = 0
                    self.snake_turn_counter = 0
            if self.pointer % 2 == 1:
                if self.snake_maintain_counter < MagicNumber.snake_50_maintain_time:
                    if self.snake_clockwise_counter % 2 == 0:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = self.clockwise
                    else:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = -1 * self.clockwise
                    self.primitive_action[self.pointer % 2]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer % 2]
                    self.step += 1
                    self.snake_maintain_counter += 1
                else:
                    maneuver = semantic_maneuver["maintain"]
                    self.pointer += 1
                    self.snake_clockwise_counter += 1
            # print(maneuver)
            # print(self.clockwise)
            if (self.snake_clockwise_counter != 0 and
                self.snake_clockwise_counter % MagicNumber.snake_clockwise_counter == 0) or \
                    threat_level == "low" or threat_level == "high" or threat_level == "danger" or (not threat_flag):
                complete_flag = True
                self.snake_clockwise_counter = 0
                self.snake_maintain_counter = 0
                self.snake_turn_counter = 0
                self.reset()
        elif maneuver_style == "snake_50_dive_25":
            self.primitive_action = [semantic_maneuver["crank_50"], semantic_maneuver["crank_50"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                # self.clockwise = random.choice([-1, 1])
                self.clockwise = default_clockwise
            if self.pointer % 2 == 0:
                if self.snake_clockwise_counter % 2 == 0:
                    goal_ao = -1 * self.clockwise * 55
                else:
                    goal_ao = -1 * self.clockwise * (-55)
                delta_ao = degrees(abs(ao - radians(goal_ao)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if self.snake_clockwise_counter % 2 == 0:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = self.clockwise
                    else:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = -1 * self.clockwise
                    if height >= 1000:
                        self.primitive_action[self.pointer % 2]["vertical_cmd"] = 3
                    else:
                        self.primitive_action[self.pointer % 2]["vertical_cmd"] = 0
                    self.primitive_action[self.pointer % 2]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer % 2]
                    self.step += 1
                    self.snake_turn_counter += 1
                else:
                    self.pointer += 1
                    self.snake_maintain_counter = 0
                    self.snake_turn_counter = 0
            if self.pointer % 2 == 1:
                if self.snake_maintain_counter < MagicNumber.snake_50_maintain_time:
                    if self.snake_clockwise_counter % 2 == 0:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = self.clockwise
                    else:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = -1 * self.clockwise
                    if height >= 1000:
                        self.primitive_action[self.pointer % 2]["vertical_cmd"] = 3
                    else:
                        self.primitive_action[self.pointer % 2]["vertical_cmd"] = 0
                    self.primitive_action[self.pointer % 2]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer % 2]
                    self.step += 1
                    self.snake_maintain_counter += 1
                else:
                    maneuver = semantic_maneuver["maintain"]
                    self.pointer += 1
                    self.snake_clockwise_counter += 1
            if (self.snake_clockwise_counter != 0 and
                self.snake_clockwise_counter % MagicNumber.snake_clockwise_counter == 0) or \
                    threat_level == "low" or threat_level == "high" or threat_level == "danger" or (not threat_flag):
                complete_flag = True
                self.snake_clockwise_counter = 0
                self.snake_maintain_counter = 0
                self.snake_turn_counter = 0
                self.reset()
        elif maneuver_style == "snake_90":
            self.primitive_action = [semantic_maneuver["crank_90"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                # self.clockwise = random.choice([-1, 1])
                self.clockwise = default_clockwise
            if self.pointer % 2 == 0:
                if self.snake_clockwise_counter % 2 == 0:
                    goal_ao = -1 * self.clockwise * 85
                else:
                    goal_ao = -1 * self.clockwise * (-85)
                delta_ao = degrees(abs(ao - radians(goal_ao)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if self.snake_clockwise_counter % 2 == 0:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = self.clockwise
                    else:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = -1 * self.clockwise
                    self.primitive_action[self.pointer % 2]["ny_cmd"] = 6
                    self.primitive_action[self.pointer % 2]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer % 2]
                    self.step += 1
                else:
                    self.pointer += 1
                    self.snake_maintain_counter = 0
            if self.pointer % 2 == 1:
                if self.snake_clockwise_counter % 2 == 0:
                    snake_maintain = MagicNumber.snake_90_maintain_time_0
                else:
                    snake_maintain = MagicNumber.snake_90_maintain_time_1
                if self.snake_maintain_counter < snake_maintain:
                    if self.snake_clockwise_counter % 2 == 0:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = self.clockwise
                    else:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = -1 * self.clockwise
                    self.primitive_action[self.pointer % 2]["ny_cmd"] = 6
                    self.primitive_action[self.pointer % 2]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer % 2]
                    self.step += 1
                    self.snake_maintain_counter += 1
                else:
                    maneuver = semantic_maneuver["maintain"]
                    self.pointer += 1
                    self.snake_clockwise_counter += 1
            if (self.snake_clockwise_counter != 0 and
                self.snake_clockwise_counter % MagicNumber.snake_clockwise_counter == 0) \
                    or threat_level == "low" or threat_level == "high" or (not threat_flag):
                complete_flag = True
                self.snake_clockwise_counter = 0
                self.snake_maintain_counter = 0
                self.reset()
        elif maneuver_style == "snake_90_dive_25":
            self.primitive_action = [semantic_maneuver["crank_90"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                # self.clockwise = random.choice([-1, 1])
                self.clockwise = default_clockwise
            if self.pointer % 2 == 0:
                if self.snake_clockwise_counter % 2 == 0:
                    goal_ao = -1 * self.clockwise * 85
                else:
                    goal_ao = -1 * self.clockwise * (-85)
                delta_ao = degrees(abs(ao - radians(goal_ao)))
                if delta_ao > MagicNumber.threshold_abort_delta_ao:
                    if self.snake_clockwise_counter % 2 == 0:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = self.clockwise
                    else:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = -1 * self.clockwise
                    if height >= 1000:
                        self.primitive_action[self.pointer % 2]["vertical_cmd"] = 3
                    else:
                        self.primitive_action[self.pointer % 2]["vertical_cmd"] = 0
                    self.primitive_action[self.pointer % 2]["ny_cmd"] = 6
                    self.primitive_action[self.pointer % 2]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer % 2]
                    self.step += 1
                else:
                    self.pointer += 1
                    self.snake_maintain_counter = 0
            if self.pointer % 2 == 1:
                if self.snake_clockwise_counter % 2 == 0:
                    snake_maintain = MagicNumber.snake_90_maintain_time_0
                else:
                    snake_maintain = MagicNumber.snake_90_maintain_time_1
                if self.snake_maintain_counter < snake_maintain:
                    if self.snake_clockwise_counter % 2 == 0:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = self.clockwise
                    else:
                        self.primitive_action[self.pointer % 2]["clockwise_cmd"] = -1 * self.clockwise
                    if height >= 1000:
                        self.primitive_action[self.pointer % 2]["vertical_cmd"] = 3
                    else:
                        self.primitive_action[self.pointer % 2]["vertical_cmd"] = 0
                    self.primitive_action[self.pointer % 2]["ny_cmd"] = 6
                    self.primitive_action[self.pointer % 2]["base_direction"] = launch_direction
                    maneuver = self.primitive_action[self.pointer % 2]
                    self.step += 1
                    self.snake_maintain_counter += 1
                else:
                    maneuver = semantic_maneuver["maintain"]
                    self.pointer += 1
                    self.snake_clockwise_counter += 1
            if (self.snake_clockwise_counter != 0 and
                self.snake_clockwise_counter % MagicNumber.snake_clockwise_counter == 0) or threat_level == "low" or (not threat_flag):
                complete_flag = True
                self.snake_clockwise_counter = 0
                self.snake_maintain_counter = 0
                self.reset()
        else:
            print("there's no ", maneuver_style, " in Evade maneuver style.")

        return maneuver, complete_flag


class Intercept(MacroTactic):
    def __init__(self):
        super().__init__()
        self.pointer = 0
        self.step = 0
        self.init_height = None
        self.combo_complete_flag = False

    def execute(self, height, shoot, threat_flag: bool, maneuver_style: str, semantic_maneuver):
        complete_flag = self.combo_complete_flag
        if maneuver_style == "intercept_cata":
            self.primitive_action = [semantic_maneuver["intercept_cata"]]
            self.primitive_action_num = 1
            maneuver = self.primitive_action[self.pointer]
            self.step += 1
            self.pointer += 1
            if self.pointer == self.primitive_action_num:
                complete_flag = True
                self.reset()
        elif maneuver_style == "intercept_level":
            self.primitive_action = [semantic_maneuver["intercept_level"]]
            self.primitive_action_num = 1
            maneuver = self.primitive_action[self.pointer]
            self.step += 1
            self.pointer += 1
            if self.pointer == self.primitive_action_num:
                complete_flag = True
                self.reset()
        elif maneuver_style == "intercept_climb":
            self.primitive_action = [semantic_maneuver["intercept_climb_20"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                self.init_height = height
            if self.pointer == 0:
                delta_height = abs(height - self.init_height)
                if delta_height < MagicNumber.min_intercept_climb_20_delta_height and height < MagicNumber.max_offense_height:
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
                self.pointer += 1
            if self.pointer == self.primitive_action_num or shoot or threat_flag:
                complete_flag = True
                self.reset()
        else:
            print("there's no ", maneuver_style, " in abort maneuver style.")

        return maneuver, complete_flag


class Format(MacroTactic):
    def __init__(self):
        super().__init__()
        self.pointer = 0
        self.step = 0
        self.combo_complete_flag = False
        self.init_height = None
        self.turn_direction = 0

    def execute(self, height, r_team, turn_direction, shoot, threat_flag: bool, maneuver_style: str, semantic_maneuver):
        complete_flag = self.combo_complete_flag
        if maneuver_style == "climb":
            self.primitive_action = [semantic_maneuver["climb_20"], semantic_maneuver["in"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                self.init_height = height
            if self.pointer == 0:
                delta_height = abs(height - self.init_height)
                if delta_height < MagicNumber.min_climb_20_delta_height and height < MagicNumber.max_offense_height:
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
                self.pointer += 1
            if self.pointer == self.primitive_action_num or shoot or threat_flag:
                complete_flag = True
                self.reset()
        elif maneuver_style == "separate":
            self.primitive_action = [semantic_maneuver["out_150"], semantic_maneuver["maintain"]]
            self.primitive_action_num = 2
            if self.pointer == 0 and self.step == 0:
                self.turn_direction = turn_direction
            if self.pointer == 0:
                if r_team < MagicNumber.min_team_range:
                    if height < MagicNumber.min_offense_height:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 5
                    else:
                        self.primitive_action[self.pointer]["vertical_cmd"] = 0
                    self.primitive_action[self.pointer]["clockwise_cmd"] = self.turn_direction
                    maneuver = self.primitive_action[self.pointer]
                    self.step += 1
                else:
                    self.pointer += 1
            if self.pointer == 1:
                maneuver = self.primitive_action[self.pointer]
                self.step += 1
                self.pointer += 1
            if self.pointer == self.primitive_action_num or r_team > MagicNumber.min_team_range or threat_flag:
                complete_flag = True
                self.turn_direction = 0
                self.reset()

        return maneuver, complete_flag


class Banzai(MacroTactic):
    def __init__(self):
        super().__init__()
        self.primitive_action_num = 1
        self.pointer = 0
        self.step = 0
        self.combo_complete_flag = False

    def execute(self, env, self_id, target_id, fuck_flag, flying_time, r_dot, threat_level: str, semantic_maneuver):
        complete_flag = self.combo_complete_flag
        ao = env.state_interface["AMS"][self_id]["relative_observation"][target_id]["AO"]["value"]
        target_alive = int(env.state_interface["AMS"][target_id]["alive"]["value"])
        msl_remain = int(env.state_interface["AMS"][self_id]["AAM_remain"]["value"])

        if degrees(abs(ao)) > 40:
            semantic_maneuver["intercept_cata"]["ny_cmd"] = 7
            maneuver = semantic_maneuver["intercept_cata"]
            shoot = 0
        else:
            maneuver = semantic_maneuver["intercept_cata"]
            if fuck_flag:
                # if flying_time > 10:
                #     if r_dot < MagicNumber.min_missile_r_dot:
                #         shoot = 1
                #     else:
                #         shoot = 0
                # else:
                #     shoot = 0
                shoot = 0
            else:
                shoot = 1

        if shoot == 1:
            complete_flag = True
            self.reset()

        return maneuver, shoot, complete_flag