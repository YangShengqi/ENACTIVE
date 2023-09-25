from environment.battlespace import BattleSpace
from reward_method.reward_hyperparam_dict import origin_reward_parameters
from train.config import Config


def get_ith_aircraft_reward(env: BattleSpace, i: int, rewards_hyperparam_dict):

    # rewards_hyperparam_dict = {"crash_extra_reward": -100, "be_shoot_down_extra_reward": -0,
    #                            "stall_extra_reward": -100, "death_event": -400,
    #                            "in_border_reward": 50, "out_border_reward": -50,
    #                            "shoot_down_reward": 600, "be_shoot_down_extra_reward": -0, "death_event":-400,
    #                            "fire_event": -50, "all_shoot_down_event_reward": 200}

    death_event_reward = rewards_hyperparam_dict["death_event"]
    be_shot_down_event_reward = rewards_hyperparam_dict["be_shot_down_event"]
    crash_event_reward = rewards_hyperparam_dict["crash_event"]
    shoot_down_event_reward = rewards_hyperparam_dict["shoot_down_event"]
    fire_event_reward = rewards_hyperparam_dict["fire_event"]
    all_shoot_down_event_reward = rewards_hyperparam_dict["all_shoot_down_event_reward"]

    ret = 0.0
    aircraft_i = env.state_interface["AMS"][i]

    ret = ret - death_event_reward * aircraft_i["death_event"]["value"]

    # death event is /crash event/be shot down event/stall event/out border event? TODO
    # ret = ret - crash_event_reward * aircraft_i["crash_event"]["value"]
    # ret = ret - be_shot_down_event_reward * aircraft_i["be_shot_down_event"]["value"]

    ret = ret + be_shot_down_event_reward * aircraft_i["be_shot_down_event"]["value"]
    ret = ret + crash_event_reward * aircraft_i["crash_event"]["value"]  # if crash, add extra penalty(-200)
    ret = ret + shoot_down_event_reward * aircraft_i["shoot_down_event"]["value"]
    ret = ret - 50 * aircraft_i["out_border_event"]["value"]
    ret = ret + 50 * aircraft_i["in_border_event"]["value"]
    # for missile in aircraft_i["SMS"]:
    #     ret = ret - 25.0*missile["fire_event"]["value"]

    missile_launch = 0
    for missile in aircraft_i["SMS"]:
        missile_launch += missile["fire_event"]["value"]

    ret = ret - fire_event_reward * (missile_launch * (missile_launch + 1)) / 2  # missile value increase with AMS remain

    # check reward problem #
    if ret > 950 or ret < -2000:
        print("ret", ret)
        print(aircraft_i["death_event"]["value"],
              aircraft_i["shoot_down_event"]["value"],
              aircraft_i["out_border_event"]["value"],
              aircraft_i["in_border_event"]["value"],
              )
        print(missile_launch)

    return ret


# rewrite on 2020/09/01, compatible of rsi init #
def get_kteam_aircraft_reward(env: BattleSpace, k: int, rewards_hyperparam_dict):
    # get each event reward from rewards_hyperparam_dict #

    # radical_reward_parameters = dict(crash_extra_reward=-100, be_shoot_down_extra_reward=0,
    #                                  stall_extra_reward=-100, death_reward=-400,
    #                                  in_border_reward=50, out_border_reward=-50,
    #                                  shoot_down_reward=600, fire_reward=-50, accumulate_fire_reward=-30,
    #                                  accumulate_shoot_down_reward=200,
    #                                  accumulate_death_reward=-200,
    #                                  all_shoot_down_event_reward=500)

    # shoot down reward #
    shoot_down_reward = rewards_hyperparam_dict["shoot_down_reward"]
    accumulate_shoot_down_reward = rewards_hyperparam_dict["accumulate_shoot_down_reward"]

    # death reward and extra reward #
    death_reward = rewards_hyperparam_dict["death_reward"]
    be_shoot_down_extra_reward = rewards_hyperparam_dict["be_shoot_down_extra_reward"]  # not used #
    stall_extra_reward = rewards_hyperparam_dict["stall_extra_reward"]
    crash_extra_reward = rewards_hyperparam_dict["crash_extra_reward"]
    accumulate_death_reward = rewards_hyperparam_dict["accumulate_death_reward"]

    # border relative reward #
    in_border_reward = rewards_hyperparam_dict["in_border_reward"]
    out_border_reward = rewards_hyperparam_dict["out_border_reward"]

    # fire reward #
    fire_reward = rewards_hyperparam_dict["fire_reward"]
    accumulate_fire_reward = rewards_hyperparam_dict["accumulate_fire_reward"]

    # all shoot down reward #
    all_shoot_down_event_reward = rewards_hyperparam_dict["all_shoot_down_event_reward"]

    # add shaped reward for training # give team reward #
    ret = 0.0
    team_kill = 0
    team_die = 0
    bandit_die = 0
    bandit_remain = 0
    team_stall = 0
    team_crash = 0
    team_launch = []
    team_in_border = 0
    team_out_border = 0

    for i in range(k * env.red, env.red + k * env.blue):
        aircraft_i = env.state_interface["AMS"][i]
        aircraft_i_msl_num = len(aircraft_i["SMS"])

        team_kill += aircraft_i["shoot_down_event"]["value"]

        # team_die += aircraft_i["death_event"]["value"]
        team_die += (1 - aircraft_i["alive"]["value"])  # already die #

        team_stall += aircraft_i["stall_event"]["value"]
        team_crash += aircraft_i["crash_event"]["value"]
        team_in_border += aircraft_i["in_border_event"]["value"]
        team_out_border += aircraft_i["out_border_event"]["value"]

        # missile_launch = 0
        # for missile in aircraft_i["SMS"]:
        #     missile_launch += missile["fire_event"]["value"]
        missile_launch = aircraft_i_msl_num - aircraft_i["AAM_remain"]["value"]
        team_launch.append(missile_launch)

    for i in range((1-k) * env.red, env.red + (1-k) * env.blue):
        aircraft_i = env.state_interface["AMS"][i]
        # bandit_die += aircraft_i["death_event"]["value"]
        bandit_remain += aircraft_i["alive"]["value"]

    if k == 0:
        bandit_die = env.blue - bandit_remain
    elif k == 1:
        bandit_die = env.red - bandit_remain

    ret += team_kill * shoot_down_reward  # shoot down target add reward
    # ret += team_kill * (team_kill - 1) / 2 * accumulate_shoot_down_reward
    ret += bandit_die * (bandit_die - 1) / 2 * accumulate_shoot_down_reward

    ret += team_die * death_reward  # dead event reward
    ret += team_stall * stall_extra_reward
    ret += team_crash * crash_extra_reward
    ret += team_kill * (team_kill - 1) / 2 * accumulate_death_reward

    ret += team_in_border * in_border_reward  # border relative reward
    ret += team_out_border * out_border_reward

    for missile_launch in team_launch:  # missile launch reward
        ret += missile_launch * fire_reward
        ret += missile_launch * (missile_launch - 1) / 2 * accumulate_fire_reward

    if (k == 0 and bandit_die == env.blue) or (k == 1 and bandit_die == env.red):  # all kill reward
        ret += all_shoot_down_event_reward

    ret_min, ret_max = get_kteam_reward_range(env, k, rewards_hyperparam_dict)
    # ret = (ret - ret_min) / (ret_max - ret_min) - 0.5  # normalize #
    ret = (ret - ret_min) / (ret_max - ret_min)  # normalize #

    return ret


def get_kteam_reward_range(env: BattleSpace, k: int, rewards_hyperparam_dict: dict):
    # radical_reward_parameters = dict(crash_extra_reward=-100, be_shoot_down_extra_reward=0,
    #                                  stall_extra_reward=-100, death_reward=-400,
    #                                  in_border_reward=50, out_border_reward=-50,
    #                                  shoot_down_reward=600, fire_reward=-50, accumulate_fire_reward=-30,
    #                                  accumulate_shoot_down_reward=200,
    #                                  accumulate_death_reward=-200,
    #                                  all_shoot_down_event_reward=500)

    # shoot down reward #
    shoot_down_reward = rewards_hyperparam_dict["shoot_down_reward"]
    accumulate_shoot_down_reward = rewards_hyperparam_dict["accumulate_shoot_down_reward"]

    # death reward and extra reward #
    death_reward = rewards_hyperparam_dict["death_reward"]
    be_shoot_down_extra_reward = rewards_hyperparam_dict["be_shoot_down_extra_reward"]  # not used #
    stall_extra_reward = rewards_hyperparam_dict["stall_extra_reward"]
    crash_extra_reward = rewards_hyperparam_dict["crash_extra_reward"]
    accumulate_death_reward = rewards_hyperparam_dict["accumulate_death_reward"]

    # border relative reward #
    in_border_reward = rewards_hyperparam_dict["in_border_reward"]
    out_border_reward = rewards_hyperparam_dict["out_border_reward"]

    # fire reward #
    fire_reward = rewards_hyperparam_dict["fire_reward"]
    accumulate_fire_reward = rewards_hyperparam_dict["accumulate_fire_reward"]

    # all shoot down reward #
    all_shoot_down_event_reward = rewards_hyperparam_dict["all_shoot_down_event_reward"]

    max_team_kill = env.blue if k == 0 else env.red
    max_team_death = env.red if k == 1 else env.blue
    missile_num = len(env.state_interface["AMS"][k * env.red]["SMS"])

    ret_max = max_team_kill * shoot_down_reward + \
              max_team_kill * (max_team_death - 1) / 2 * accumulate_shoot_down_reward + \
              0 * (death_reward + stall_extra_reward + crash_extra_reward) + \
              0 * accumulate_death_reward + \
              fire_reward * max_team_kill + \
              all_shoot_down_event_reward

    ret_min = 0 * shoot_down_reward + \
              0 * accumulate_shoot_down_reward + \
              max_team_death * (death_reward + stall_extra_reward + crash_extra_reward) + \
              max_team_death * (max_team_death - 1) / 2 * accumulate_death_reward + \
              fire_reward * missile_num * max_team_death + \
              accumulate_fire_reward * missile_num * (missile_num - 1) / 2 * max_team_death + \
              out_border_reward * max_team_death

    return ret_min, ret_max


if __name__ == "__main__":
    # test code #
    env = Config.env
    env.reset()
    ret_min, ret_max = get_kteam_reward_range(env, 1, origin_reward_parameters)
    print("min: ", ret_min)
    print("max: ", ret_max)
