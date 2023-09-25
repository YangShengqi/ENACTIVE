import torch
import os
import datetime
from environment.battlespace import BattleSpace
import multiprocessing as mp
import socket
import json
import math


class Config:

    """
    1.file operation must use abspath
    2.import module must from second dir or general module
    3.boot_script must exist sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../....../..'))) at first line
    """
    ############################
    # not frequently used part #
    ############################
    # os and sys configuration #
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # env relative, env established sampler node #
    env_list = []
    env_config_list = []
    env = BattleSpace()
    env.random_init()
    env.reset()

    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../environment/config.json'), "r")
    config = f.read()
    config = json.loads(config)
    whole_time = float(config["time_out"])

    # protect args #
    multinomial_protect = 1e-10
    log_protect = 1e-10
    devide_protect = 1e-10

    # host/address relative #
    docker = False
    if docker:
        scheduler_address = socket.gethostbyname("schedule")
    else:
        scheduler_address = "192.168.100.2"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((scheduler_address, 80))
    local_address = s.getsockname()[0]

    # model establish args #
    rnn_switch_on = False
    hybrid_v_c = [340, 680]  # available when using hybrid and semantic #
    hybrid_nn_c = [5, 7]

    ltt_interval = 3  # available when use ltt method
    delib_cost = 0.2

    # model train args #
    TD_step = 0  # when computing TD error
    gamma = 0.99  # for reward discount
    tau = 0.95  # for advantage transition discount
    lam = 0.95  # lam for gae transition
    l2_reg = 0

    # sample args #
    SC_cpu = False  # SC_cpu: super computer cpu
    if SC_cpu:
        # use may set cpu_core_num or use "cpu_core_num = max(int((mp.cpu_count() - 2) / 2), 1)"
        cpu_core_num = 30
        # cpu_core_num = max(int((mp.cpu_count() - 2) / 2), 1)
    else:
        cpu_core_num = mp.cpu_count()

    ########################
    # frequently used part #
    ########################
    mini_batch_size = 3000  # when using self play pbt, config of batch size in SidelessPBT
    optim_batch_size = 300
    epochs = 5
    assign_sample_num_for_node = False  # if this is true, sample num is set to node, else sample num is set to process
    dynamic_env_method = False
    single_process_batch_size = 200
    single_node_batch_size = 2400
    group_game_num = 200
    single_process_game_num = 5
    single_node_game_num = 40

    # random state initialization trick.
    # for preventing agent from overfitting to battle initial phase.
    rsi_ratio = 0

    # log path #
    evaluation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation/" + str(datetime.datetime.now()) + "/")
    pbt_eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pbt_eval_res/" + str(datetime.datetime.now()) + "/")
    replay_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay/" + str(datetime.datetime.now()) + "/")
    tensorboard_logging = False
    tensorboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb/" + str(datetime.datetime.now()) + "/")

    class SimpleCentralScheduler:
        scheduler_save_iteration = 5
        red_agent = None
        blue_agent = None

    class SingleAgentTrainScheduler:
        scheduler_save_iteration = 5
        red_agent = None
        blue_agent = None

    class PPO:
        clip_epsilon = 0.2
        clip_c = 1.2

    class Battle:
        red_agent = None
        blue_agent = None
        times = None
        replay = False
        terminal_mode = False
        god_eye_address = ["0.0.0.0"]

    class SidelessPBT:
        self_play_mode = True
        control_task_num = True
        # control_sample_rate = False

        agent_num = 16  # total agent num in population
        static_agent_num = 8  #
        agent_start_elo = 1500
        elo_update_K = 32
        agent_array = []
        state_machine_array = []

        trainer_evoking_overtime = 500
        use_parallel_sample_train_method = False  # when this is true, set SC_cpu_num to fit computing power

        # pbt sample relative parameters #
        num_of_agent_per_group = 4
        if self_play_mode:
            agent_num = 2
            num_of_agent_per_group = 2

        if self_play_mode:
            sample_rate_inside_group = 0.6
            sample_rate_outside_group = 0
            sample_rate_with_history = 0.2
            history_agent_in_sample = 8
            sample_rate_with_statemachine = 0.2
            statemachine_agent_in_sample = 2
        else:
            # following rate may change by pbt function #
            sample_rate_inside_group = 0.4
            sample_rate_outside_group = 0.2
            sample_rate_with_history = 0.2
            history_agent_in_sample = 8  # todo consider redis capacity #
            sample_rate_with_statemachine = 0.2
            statemachine_agent_in_sample = 4

        eliminate_rate = 0.25
        pbt_agent_sample_rate = 0.5
        alph = 0.1
        mutation_prob = 0.1
        mutation_perturbe = 0.1

        # tournament
        history_agent_num_in_tournament = 4
        state_machine_num_in_tournament = 2

        # history_agent_array = []
        # program structure parameters #
        init_training = False
        init_training_iteration = 500
        pbt_training_iteration = 1
        agent_save_iteration = 10  # also used in self play mode
        evaluation_iteration = 500

        train_and_tournament_iter_num_between_pbt = 5
        log_dir_path = None

    class Independ:
        pro_factor_alpha = 0.1  # other agents mean reward weight
        beta = 0  # default 0, team_reward weight











