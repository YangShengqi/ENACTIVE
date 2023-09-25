from framwork.agent_base import AgentBase
from models.stick import Stick


class StickAgent(AgentBase):
    def __init__(self):
        self.side = None
        self.stick = Stick()
        self.interval = 0.03

    def after_reset(self, env, side):
        if side == "red":
            self.side = 0
        elif side == "blue":
            self.side = 1

    def before_step_for_sample(self, env):
        actions = self.stick.sample()
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            #print("sample before")

            relative_index = i - self.side * env.red
            env.action_interface["AMS"][i]["F22Stick"]["action_Tc"]["value"] = 1
            env.action_interface["AMS"][i]["F22Stick"]["action_nyc"]["value"] = \
            actions[relative_index if relative_index < len(actions) else -1]["axis1"]
            env.action_interface["AMS"][i]["F22Stick"]["action_wxc"]["value"] = \
            actions[relative_index if relative_index < len(actions) else -1]["axis0"]
            #print("axis0",env.action_interface["AMS"][i]["F22Stick"]["action_wxc"]["value"])
            #print("axis1", env.action_interface["AMS"][i]["F22Stick"]["action_nyc"]["value"])
            # env["AWS"][i]["action_shoot_predict_list"][0]["shoot_predict"]["value"] = 0
            # action["action_shoot_predict_list"][1]["shoot_predict"]["value"] = 0
            env.action_interface["AMS"][i]["action_shoot"]["value"] = actions[relative_index if relative_index < len(actions) else -1][
                "shoot"]
            target_aircraft = env.action_interface["AMS"][i]["target_aircraft"]["value"]
            target_aircraft = (self.side*env.red + env.blue) % (env.red+env.blue) if target_aircraft is None else target_aircraft
            target_aircraft = env.resort(target_aircraft,actions[relative_index if relative_index < len(actions) else -1]["target"])
            env.action_interface["AMS"][i]["target_aircraft"]["value"] = target_aircraft

            if i < env.red:
                for j in range(env.blue):
                    env.action_interface["AMS"][i]["action_shoot_predict_list"][j]["shoot_predict"]["value"] = 0
            else:
                for j in range(env.red):
                    env.action_interface["AMS"][i]["action_shoot_predict_list"][j]["shoot_predict"]["value"] = 0

    def after_step_for_sample(self, env):
        pass

    def before_step_for_train(self, env):
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
