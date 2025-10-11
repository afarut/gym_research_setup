import numpy as np
import torch
from collections import defaultdict
import requests
import subprocess
import os
import signal
import json
import time
import torch.nn as nn


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
    url = f"http://127.0.0.1:{port}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return True
    except requests.ConnectionError:
        pass
    return False


def kill_process_on_port(port: int):
    try:
        result = subprocess.check_output(
            ["lsof", "-nP", "-i", f"TCP:{port}", "+c", "15"]
        ).decode()
        for line in result.splitlines()[1:]:
            cols = line.split()
            pid = int(cols[1])
            cmd = cols[0]
            if "tensorboard" in cmd.lower() or "python" in cmd.lower() or "streamlit" in cmd.lower():
                print(f"Killing {cmd} (PID {pid})...")
                os.kill(pid, signal.SIGKILL)
            else:
                print(f"Skipping {cmd}")
    except subprocess.CalledProcessError:
        print(f"No process found on port {port}.")


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

    kill_process_on_port(port)
    try:
        # Запуск TensorBoard в subprocess
        subprocess.Popen(
            ["tensorboard", "--logdir_spec", run_with_alias, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return f"TensorBoard запущен на порту {port}", False
    except Exception as e:
        return f"Ошибка запуска TensorBoard: {e}", True


def build_mlp(input_dim, hidden_dim, depth, output_dim=None, activation="ReLU", normalization="LayerNorm"):
    layers = []
    act = getattr(nn, activation)
    norm = getattr(nn, normalization)
    prev_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act())
        layers.append(norm(hidden_dim))
        prev_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)