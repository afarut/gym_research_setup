# Gym Research Setup
This project is needed for quick and convenient testing of hypothesis at the gymnasium (and IsaacGym).
## Quick start
### Important
Before launch this project make sure that true ports are indicate for TensorBoard and Streamlit in `train.yaml`.
- If something runs in this ports, then it will be kills
- Config name for this ports are `tensorboard_port` and `streamlit_port`
### Setting up the environment
Download and unzip the [archive](https://developer.nvidia.com/isaac-gym/download) into `isaacgym/`
#### IsaacGym Dockerfile
Add this command at the end `isaacgym/docker/Dockerfile`
- `RUN cd isaacgym/IsaacGymEnvs && pip install -q -e .`
- `
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt`
#### IsaacGym run.sh
Edit your `isaacgym/docker/run.sh` so it looks something like this:
```bash
#!/bin/bash
set -e
set -u


if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host --gpus=all --name=isaacgym_container isaacgym /bin/bash
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/opt/isaacgym -e DISPLAY=$DISPLAY --network=host --gpus=all -p 6006:6006 -p 8501:8501 --name=isaacgym_container isaacgym /bin/bash
	xhost -
fi
```
#### IsaacGym build.sh
Edit your `isaacgym/docker/build.sh` so it looks something like this:
```bash
#!/bin/bash
set -e
set -u

if [ ! -d "./isaacgym/IsaacGymEnvs" ]; then
	git clone https://github.com/isaac-sim/IsaacGymEnvs ./isaacgym/IsaacGymEnvs
fi

docker build --network host -t isaacgym -f isaacgym/docker/Dockerfile .
```
### Build env
```
sh isaacgym/docker/build.sh
```
- It must to do only once
### Connect to env
```
sh isaacgym/docker/run.sh :0
```

### Run inference in container
```
python run_inference.py +setup=LunarLander
```
- You can inference from checkpoint with `python run_inference.py +checkpoint_path=outputs/DATE/TIME`
  - Where `DATE` and `TIME` your path to launch
### Run train in container
```
python run_train.py +setup=LunarLander
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
python run_train.py +setup=LunarLander +checkpoint_path=outputs/DATE/TIME
```
- You can drop past information from tensorboard with `+checkpoint_drop_past=True`
  - If you use this flag, then will be use `best` model, else `last`
### How to run train without saving model? 
```
python run_train.py +setup=LunarLander +mode=debug
```
