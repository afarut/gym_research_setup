import streamlit as st
import json
import os, sys
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils import restart_tensorboard


META_FILE = "meta.json"

if not os.path.exists(META_FILE):
    with open(META_FILE, "w") as f:
        json.dump([], f)

with open(META_FILE, "r") as f:
    try:
        runs_meta = json.load(f)
    except json.decoder.JSONDecodeError:
        runs_meta = []

st.title("Run Alias & Visibility Manager")

for run in runs_meta:
    cols = st.columns([3, 1])

    run['alias'] = cols[0].text_input(
        f"Alias for {run['run_id']}", 
        value=run.get('alias', ''), 
        key=f"alias_{run['run_id']}"
    )

    run['visible'] = cols[1].checkbox(
        "Visible", 
        value=run.get('visible', False), 
        key=f"visible_{run['run_id']}"
    )

if st.button("Сохранить"):
    with open(META_FILE, "w") as f:
        json.dump(runs_meta, f, indent=2)
    st.success("Alias и Visible обновлены!")

    logdir = f"logs/{run['run_id']}"  # Путь к логам TensorBoard
    cfg = OmegaConf.load("config/train.yaml")
    port = cfg["tensorboard_port"]
    
    message, error = restart_tensorboard(port)
    if error:
        st.error(message)
    else:
        st.success(message)