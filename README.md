# Gym Research Setup
This project is needed for quick and convenient testing of hypothesis at the gymnasium.
## Quick start
### Important
Before launch this project make sure that true ports are indicate for TensorBoard and Streamlit in `train.yaml`.
- If something runs in this ports, then it will be kills
- Config name for this ports are `tensorboard_port` and `streamlit_port`
### Install requirements
```
pip install -r requirements.txt
```
- Better will be use with virtual env
### Run inference
```
python run_inference.py
```
- You can inference from checkpoint with `python run_inference.py +checkpoint_path=outputs/DATE/TIME`
  - Where `DATE` and `TIME` your path to launch
### Run train
```
python run_train.py
```
- For mo details go to link
## Streamlit and TensorBoard
You can open localhost with `tensorboard_port` and `streamlit_port` to see that
- checkbox with `visible` in streamlit is responsible for visible in TensorBoard
- Input with `alias` is responsible for alias in TensorBoard

## More details
### How to clear all launchs?
```
python run_clear.py
```
### How to run train from checkpoint?
```
python run_train.py +checkpoint_path=outputs/DATE/TIME
```
- You can drop past information from tensorboard with `+checkpoint_drop_past=True`
  - If you use this flag, then will be use `best` model, else `last`
### How to run train without saving model? 
```
python run_train.py +mode=debug
```
