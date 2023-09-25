from environment.battlespace import BattleSpace


def get_gcd(a: int, b: int):  # greatest common divisor
    if a < b:
        small = a
    else:
        small = b
    gcd = 1
    for i in range(1, int(small + 1)):
        if a % i == 0 and b % i == 0:
            gcd = i
    return gcd


def generate_env_list(env_list, env_config_list, agent_list):
    for i in range(len(agent_list)):
        for j in range(i, len(agent_list)):
            m_model_0 = agent_list[i].maneuver_model
            m_model_1 = agent_list[j].maneuver_model
            interval_0 = agent_list[i].interval
            interval_1 = agent_list[j].interval
            maneuver_model_0 = [m_model_0, m_model_1]
            maneuver_model_1 = [m_model_1, m_model_0]

            interval = get_gcd(interval_0, interval_1)
            config_key_0 = [maneuver_model_0, interval]
            config_key_1 = [maneuver_model_1, interval]
            if config_key_0 in env_config_list:
                pass
            else:
                env_config_list.append(config_key_0)
                env_list.append(BattleSpace(maneuver_list=maneuver_model_0, interval=interval))
            if config_key_1 in env_config_list:
                pass
            else:
                env_config_list.append(config_key_1)
                env_list.append(BattleSpace(maneuver_list=maneuver_model_1, interval=interval))

    for env in env_list:
        env.random_init()
        env.reset()

    # for config in env_config_list:
    #     print(config)
    # print(env_config_list)


def generate_env_list_config(env_list, env_config_list, agent_config_list):
    for i in range(len(agent_config_list)):
        for j in range(i, len(agent_config_list)):
            m_model_0 = agent_config_list[i][0]
            m_model_1 = agent_config_list[j][0]
            interval_0 = agent_config_list[i][1]
            interval_1 = agent_config_list[j][1]
            maneuver_model_0 = [m_model_0, m_model_1]
            maneuver_model_1 = [m_model_1, m_model_0]

            interval = get_gcd(interval_0, interval_1)
            config_key_0 = [maneuver_model_0, interval]
            config_key_1 = [maneuver_model_1, interval]
            if config_key_0 in env_config_list:
                pass
            else:
                env_config_list.append(config_key_0)
                env_list.append(BattleSpace(maneuver_list=maneuver_model_0, interval=interval))
            if config_key_1 in env_config_list:
                pass
            else:
                env_config_list.append(config_key_1)
                env_list.append(BattleSpace(maneuver_list=maneuver_model_1, interval=interval))

    for env in env_list:
        env.random_init()
        env.reset()
