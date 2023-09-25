from os import path


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))

def get_ip( ethcard ):
    info = psutil.net_if_addrs()
    for k,v in info.items():
        for item in v:
            if k==ethcard:
                return item[1]
    return 0
