from train.config import Config
from framwork.utils import save_replay
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn
from threading import Thread
from copy import deepcopy
import json
from urllib.parse import parse_qs
import multiprocessing as mp
from environment.battlespace import BattleSpace
import time
import socket
import os
import sys
from environment.dynamic_env_establish import get_gcd


class Battle:

    state = dict()
    action = dict()
    port = 20601

    def __init__(self):

        __class__.ThreadingHttpServer.allow_reuse_address = True
        web_server = __class__.ThreadingHttpServer(("0.0.0.0", __class__.port), __class__.MyServer)
        p = Thread(target=web_server.serve_forever)
        p.start()

        self.red_agent = None
        self.blue_agent = None
        self.dog = IntervalDog(0.03)
        self.episode = -1
        self.env = Config.env
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../environment/"))

        self.times = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,200000000)
        # sudo sysctl -w net.core.wmem_max = 200000000
        self.q1 = mp.Queue()
        self.q2 = mp.Queue()
        if Config.Battle.replay and not os.path.exists(Config.replay_path):
            os.makedirs(Config.replay_path, exist_ok=True)

    def run(self):
        self.red_agent = Config.Battle.red_agent
        self.blue_agent = Config.Battle.blue_agent
        self.times = Config.Battle.times
        file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../environment/sim_out.json"), "w+")
        p = mp.Process(target=self.send)
        p.start()

        if Config.dynamic_env_method:  # use as pbt sample
            red_maneuver_model = self.red_agent.maneuver_model
            blue_maneuver_model = self.blue_agent.maneuver_model
            maneuver_model_list = [red_maneuver_model, blue_maneuver_model]
            red_interval = self.red_agent.interval
            blue_interval = self.blue_agent.interval
            env_interval = get_gcd(red_interval, blue_interval)
            Config.env = BattleSpace(maneuver_model_list, env_interval)

        self.env = Config.env
        episode = -1

        while True:
            self.q2.get()
            line_count = 0
            file.seek(0)
            steps = 0
            begin_time = time.time()
            red_interval_step = self.get_interval_step(self.env, self.red_agent)
            blue_interval_step = self.get_interval_step(self.env, self.blue_agent)
            self.env.random_init()
            self.env.reset(True)
            self.episode = self.episode + 1
            self.red_agent.after_reset(self.env, "red")
            self.blue_agent.after_reset(self.env, "blue")
            while True:
                data = file.readlines()
                line_count = line_count + len(data)
                self.q1.put(line_count)
                self.dog.step()
                if steps != 0:
                    if steps % red_interval_step == 0 or self.env.done:  # same as sample_base, very important
                        self.red_agent.after_step_for_sample(self.env)
                    if steps % blue_interval_step == 0 or self.env.done:
                        self.blue_agent.after_step_for_sample(self.env)
                if self.env.done:
                    if Config.Battle.replay:
                        save_replay(self.episode)
                    if Config.Battle.terminal_mode:
                        while True:
                            command = sys.stdin.readline()
                            if command == "Fangs Out\n":
                                print("Good Luck")
                                break
                    print("over")
                    break
                if steps % red_interval_step == 0:
                    self.red_agent.before_step_for_sample(self.env)
                if steps % blue_interval_step == 0:
                    self.blue_agent.before_step_for_sample(self.env)
                self.update()
                interval = time.time() - begin_time
                while (line_count - 1 - (int)(interval/0.03)*self.times) > self.times:
                    time.sleep(0.03)
                    interval = interval + 0.03
                self.env.step()
                steps = steps + 1

            self.q1.put(-1)
        self.q1.put(-2)

    def get_interval_step(self, env, agent):
        if agent.get_interval() < env.interval:
            return 1
        else:
            return int(round(float(agent.get_interval())/float(env.interval)))

    def send(self):
        file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../environment/sim_out.json"), "w+")
        line_count = 0
        self.q2.put(0)
        while True:
            temp = self.q1.get()
            if temp == -1:
                file.seek(0)
                line_count = 0
                self.q2.put(0)
                continue
            elif temp == -2:
                break
            while line_count < temp:
                #print(line_count, temp)
                line = file.readline()
                line_count = line_count + 1
                if (line_count - 1) % self.times is 0:
                    for addr in Config.Battle.god_eye_address:
                        self.sock.sendto(line.encode("ascii"), (addr, 10601))
                    self.dog.step()
                else:
                    continue

    class ThreadingHttpServer(ThreadingMixIn, TCPServer):
        pass

    class MyServer(BaseHTTPRequestHandler):

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            #print(self.headers)
            self.wfile.write(bytes(str(Battle.state).replace("\'", "\"").encode(encoding="ascii")))

        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            length = int(self.headers['content-length'])
            data = json.loads(parse_qs(self.rfile.read(length).decode("utf8"))["data"][0])
            Battle.action[data["id"]] = data
            self.wfile.write(bytes(str(Battle.state).replace("\'", "\"").encode(encoding="ascii")))

        def log_request(self, code=None, size=None):
            pass

        def log_message(self, format, *args):
            pass

    def update(self):
        state = dict()
        state["data"] = deepcopy(
            self.env.state_interface)  # must deepcopy, because env.state_interface is not threading safe
        state["red"] = self.env.red
        state["blue"] = self.env.blue
        state["episode"] = self.episode
        state["action"] = __class__.action
        __class__.state = state

    def get_pilot_action(self, env, episode):
        state = dict()
        state["data"] = deepcopy(env.state_interface) # must deepcopy, because env.state_interface is not threading safe
        state["red"] = env.red
        state["blue"] = env.blue
        state["episode"] = episode
        state["action"] = __class__.action
        __class__.state = state
        return __class__.action


class IntervalDog:
    def __init__(self, interval: float):
        self.interval = interval
        self.step_count = 0
        self.last_time = 0

    def step(self):
        if self.step_count is not 0:
            interval = self.interval - (time.time() - self.last_time)
            if interval > 0:
                time.sleep(interval)
        self.step_count = self.step_count + 1
        self.last_time = time.time()
