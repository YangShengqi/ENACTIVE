import multiprocessing
from utils.replay_memory import Memory
from utils.math import *
from utils.torch import *
import math
import time

import copy
import pickle
from io import BytesIO
import multiprocessing as mp

import grpc
import train.distributed.grpc_proto_pb2 as r_p
import train.distributed.grpc_proto_pb2_grpc as r_p_g

from train.args_init import args, grpc_available_server_host

import redis
import socket


def merge_log(log_list, aircraft_num):
    log = dict()
    log['total_reward'] = [0]*aircraft_num
    log['num_episodes'] = [0]*aircraft_num
    log['num_steps'] = [0]*aircraft_num
    log['avg_reward'] = [0]*aircraft_num
    log['max_reward'] = [-1e6]*aircraft_num
    log['min_reward'] = [1e6]*aircraft_num
    log['avg_positive_reward'] = [0]*aircraft_num
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = [0]*aircraft_num
        log['avg_c_reward'] = [0]*aircraft_num
        log['max_c_reward'] = [-1e6]*aircraft_num
        log['min_c_reward'] = [1e6]*aircraft_num
    for i in range(aircraft_num):
        log['total_reward'][i] = sum([x['total_reward'][i] for x in log_list])
        log['num_episodes'][i] = sum([x['num_episodes'][i] for x in log_list])
        log['num_steps'][i] = sum([x['num_steps'][i] for x in log_list])
        log['avg_reward'][i] = log['total_reward'][i] / log['num_episodes'][i]
        log['max_reward'][i] = max([x['max_reward'][i] for x in log_list])
        log['min_reward'][i] = min([x['min_reward'][i] for x in log_list])

        # log['avg_positive_reward'][i] = sum([x['avg_positive_reward'][i] for x in log_list]) / len(log_list)  # have some problem here #
        if len(log_list) == 1:  # first merge
            log['avg_positive_reward'][i] = sum([x['avg_positive_reward'][i] for x in log_list]) / len(log_list)
        else:
            if log_list[1]['avg_positive_reward'][i] > 0:
                log['avg_positive_reward'][i] = sum([x['avg_positive_reward'][i] for x in log_list]) / len(log_list)
            else:
                log['avg_positive_reward'][i] = log_list[0]['avg_positive_reward'][i]

        if 'total_c_reward' in log_list[0]:
            log['total_c_reward'][i] = sum([x['total_c_reward'][i] for x in log_list])
            log['avg_c_reward'][i] = log['total_c_reward'][i] / log['num_steps'][i]
            log['max_c_reward'][i] = max([x['max_c_reward'][i] for x in log_list])
            log['min_c_reward'][i] = min([x['min_c_reward'][i] for x in log_list])

    return log


class Agent:
    def __init__(self, env, multihead_nets, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1, coop_factor=1.0):
        self.env = env
        self.multihead_nets = multihead_nets
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads
        self.coop_factor = coop_factor
        self.available_server_host = []
        self.available_server_host_check = []
        self.origin_available_server_host = grpc_available_server_host
        self.available_server_num_check = len(self.origin_available_server_host)
        self.learner_redis = redis.Redis(host=args.learing_redis_host, port=args.learing_redis_port)

    def check_conn(self):  # check # using subprocess

        self.learner_redis.delete("available_host")  # clear redis hosts key

        grpc_port = args.local_grpc_port

        for host_test in self.origin_available_server_host:
            conn = grpc.insecure_channel(host_test + ':' + grpc_port)  # connection
            client_request = r_p_g.testStub(channel=conn)
            try:
                response = client_request.rpc_test_server(r_p.send_tag())
                self.learner_redis.lpush("available_host", host_test)
            except:
                # self.available_server_host_check.remove(host_test)
                print("server on host {} down".format(host_test))

    def check_server_available(self):

        self.available_server_host_check = copy.deepcopy(self.origin_available_server_host)
        self.available_server_num_check = len(self.origin_available_server_host)

        process = mp.Process(target=self.check_conn, args=())
        process.start()
        process.join()

        host_available_iteration_byte = self.learner_redis.lrange("available_host", 0, -1)

        host_available_iteration = []
        for elem in host_available_iteration_byte:
            host_available_iteration.append(elem.decode())

        print("available actor server list : ", host_available_iteration)

        self.available_server_num_check = len(host_available_iteration)

        return host_available_iteration

    def collect_samples(self):

        self.available_server_num_check = 0

        while True:  # break when at least one actor is ready
            if args.docker:
                _, _, addr = socket.gethostbyname_ex("sample")
                print("sample addr ", addr)
                self.origin_available_server_host = addr
            available_server_host_now = self.check_server_available()  # test available server

            if self.available_server_num_check == 0:
                print("no actor is ready, please start actor")
                time.sleep(5)
                continue
            else:
                print("{} actor(s) is available, start sampling".format(self.available_server_num_check))
                break

        self.learner_redis.set("sample_enough", "no")

        t_start = time.time()
        to_device(torch.device('cpu'), *self.multihead_nets)#TODO
        team_num = 2

        process_array = []
        for host_now in available_server_host_now:
            process_array.append(mp.Process(target=self.grpc_collect_sample_client_start, args=(host_now, args.local_grpc_port)))
            process_array[-1].start()

        memorys = [[] for _ in range(team_num)]
        log = dict()

        redis_sample_pack = 0
        while True:
            sample_data_j = self.learner_redis.brpop("sample")
            f = BytesIO()
            f.write(sample_data_j[1])
            f.seek(0)
            data = pickle.load(f)
            f.close()

            redis_sample_pack += 1
            if redis_sample_pack % 50 == 0:
                print("get {} groups of sample from redis".format(redis_sample_pack))

            for i in range(team_num):
                memorys[i].extend(data[0][i])
            if redis_sample_pack == 1:
                log = merge_log([data[1]], team_num)
            else:
                log = merge_log([log, data[1]], team_num)

            memorys_len = []
            for i in range(team_num):  # get memory len
                memorys_len.append(len(memorys[i]))
            min_memory_len = min(memorys_len)

            if min_memory_len > args.min_batch_size:
                #print("aircraft memory length: ", len(memorys[0]), len(memorys[1]), len(memorys[2]), len(memorys[3]))
                print("sample enough")
                self.learner_redis.set("sample_enough", "yes")
                break

            self.learner_redis.delete("sample")  # clear samples remain in redis

        batchs = [dict(state=[], maneuver=[], steer=[], shot=[], target=[],
                       mask=[], next_state=[], reward=[]) for _ in range(team_num)]

        for i in range(team_num):
            for m in memorys[i]:
                for k, v in m.items():
                    batchs[i][k].append(v)
        to_device(self.device, *self.multihead_nets)
        t_end = time.time()
        log['sample_time'] = t_end - t_start

        return batchs, log

    def grpc_collect_sample_client_start(self, host_now, port_now):
        # grpc sample process
        conn = grpc.insecure_channel(host_now + ':' + port_now)  # connection
        client_request = r_p_g.sampleStub(channel=conn)
        print("connection on host {} established".format(host_now))

        packed_nets = [self.multihead_nets]
        file = BytesIO()
        pickle.dump(packed_nets, file)
        data_to_send = file.getvalue()
        file.close()

        response = client_request.rpc_start_samples(r_p.data_send(packed_data=data_to_send))

        return response.done
