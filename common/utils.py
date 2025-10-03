import numpy as np
import torch
from collections import defaultdict
import requests
import subprocess
import os
import signal
import json
import time


def stack_dict(data: dict, axis=0):
    for key in data:
        data[key] = torch.from_numpy(np.stack(data[key], axis=axis))
    return data


def collate_to_device(batch, device="cuda"):
    unpacked = list(zip(*batch))
    for i in range(len(unpacked)):
        unpacked[i] = torch.stack(unpacked[i]).to(device)
    return unpacked


def list_dict_extend(lst: list) -> dict:
    data = defaultdict(list)
    for dct in lst:
        for key in dct:
            data[key].extend(dct[key])
    return data


def reinforce_loss(pred: torch.Tensor, rewards: torch.Tensor):
    return -(pred.log_softmax(dim=-1) * (rewards)).mean()


def is_port_running(port):
    port = 8501
    url = f"http://127.0.0.1:{port}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return True
    except requests.ConnectionError:
        pass
    return False


def kill_process_on_port(port):
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Убил процесс PID={pid} на порту {port}")
    except Exception as e:
        print(f"Ошибка: {e}")


def add_meta(entry):
    with open("meta.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    data.append(entry)

    with open("meta.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def rand_by_time(a=0, b=100):
    t = int(time.time_ns())
    return a + (t % (b - a + 1))


def restart_tensorboard(port):
    with open("meta.json", "r", encoding="utf-8") as f:
        runs_meta = json.load(f)

    run_with_alias = ""
    tmp = []
    for run in runs_meta:
        if run['visible']:
            tmp.append(f"{run['alias']}:{run['run_id']}/")
    run_with_alias = ",".join(tmp)

    if is_port_running(port):
        kill_process_on_port(port)
    try:
        # Запуск TensorBoard в subprocess
        subprocess.Popen(
            ["tensorboard", "--logdir_spec", run_with_alias, "--port", str(port)],
            # "--logdir_spec exp1:/path/to/run1,exp2:/path/to/run2"
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return f"TensorBoard запущен на порту {port}", False
    except Exception as e:
        return f"Ошибка запуска TensorBoard: {e}", True