from io import BytesIO
import pickle
import os
from shutil import copyfile
from train.config import Config


def serialize(data):
    f = BytesIO()
    pickle.dump(data, f)
    serialized_data = f.getvalue()
    f.close()
    return serialized_data


def deserialize(serialized_data):
    f = BytesIO()
    f.write(serialized_data)
    f.seek(0)
    data = pickle.load(f)
    f.close()
    return data


def save_obj_to_file(obj, name):
    obj_path = os.path.join(Config.evaluation_path, str(name))
    pickle.dump(obj, open(obj_path, 'wb'))


def save_obj_to_file_for_pbt(obj, name):
    obj_path = os.path.join(Config.pbt_eval_path, str(name))
    pickle.dump(obj, open(obj_path, 'wb'))


def save_replay(name):
    replay_path = os.path.join(Config.replay_path, str(name))
    copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../environment/sim_out.json"), replay_path)


def load_obj_from_file():
    obj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../train/mrlittle.p")
    return pickle.load(open(obj_path, "rb"))
