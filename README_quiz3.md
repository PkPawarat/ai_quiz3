# Assignment 3: Reinforcement Learning

In this assignment you will apply the RL algorithms you learnt from the tutorials to a simulated robot car in a pybullet environment.

You will be asked to (percentages are allocation of assignment marks):

- Train the robot to drive to the green goal marker which spawns at random locations (60%)
- Modify the epsilon-greedy function to incorporate prior knowledge (20%)
- Modify the reward function (10%)
- Add obstacles to the environment (10%)

It is highly recommended to install pybullet and run your code locally since things will run much faster. It will also make editing the gym environment code easier.

## Simple Car Environment

---

![simple_car.gif](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExODU0NmVlMzU1MGU1YzJjMjA5ODE5NjM0MTg0MTU1ZmM1OTA1NzRkNCZjdD1n/VI3OuvQShK3gzENiVz/giphy.gif)

_(code for this tutorial adapted from: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e_)

---

This is a simple car environment with a continuous state space and discrete action space with the goal of driving towards a green marker. Driving within 1.5 metres of the green marker causes the episode to end or if a certain amount of time has passed.

We can instantiate the environment as follows:

```
env = gym.make('SimpleDriving-v0', apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
```

### Action Space

- 0: Reverse-Left
- 1: Reverse
- 2: Reverse-Right
- 3: Steer-Left (no throttle)
- 4: No throttle and no steering
- 5: Steer-Right (no throttle)
- 6: Forward-right
- 7: Forward
- 8: Forward-left

### Observation Space

Two dimensional array with distance in (x, y) from goal position.

###Rewards
Negative euclidean distance from the goal.

### Interacting with the Environment

We can sample actions randomly, get the agent to perform that action and then observe how the environment state changes:

```
state = env.reset()  # this needs to be called once at the start before sending any actions
action = env.action_space.sample()
state, reward, done, _, info = env.step(action)
```

---

## Installing and Modifying Gym Environment Code

For installing in collab you would have already been familiar with using the following command:

```
pip install git+https://github.com/fredsukkar/simple-car-env-template
```

To edit the gym environment first create a github account and then go to https://github.com/fredsukkar/simple-car-env-template and create a new repository using the repository as a template as follows:
![sdlfk](https://i.ibb.co/MMsLv1G/github-template.jpg)

Once you have your own copy of the repository you can then edit the files in the browser via github or alternatively (recommended) you can [clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) and modify the code locally.

To install the package from github you can use the usual:

```
pip install git+https://github.com/<your repo>
```

To install the package locally:

```
cd /path/to/cloned_repo
python setup.py install
```

Note that for both methods you will need to install the package again after you've made any changes for them to take any effect.

The main file you will be modifying is: https://github.com/fredsukkar/Gym-Medium-Post/blob/main/simple_driving/envs/simple_driving_env.py.

There are four main functions that you have been calling via the gym environment object:

```
class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass
```

Parts 3 and 4 of the assignment will ask you to modify one of these functions.

---

Before we can execute any code we first need to install the following packages:

!apt update
!pip install gym==0.26.2 pyvirtualdisplay pygame torch > /dev/null 2>&1
!pip install git+https://github.com/fredsukkar/simple-car-env-template > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!apt-get install -y xvfb

Now import the necessary packages and following helper functions (you don't need the `display_video` function if running locally):

import gym
import simple_driving

# import pybullet_envs

import pybullet as p
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from collections import defaultdict
import pickle
from IPython.display import clear_output
import torch
import random

display = Display(visible=0, size=(400, 300))
display.start()

def display_video(frames, framerate=30):
"""Generates video from `frames`.

Args:
frames (ndarray): Array of shape (n_frames, height, width, 3).
framerate (int): Frame rate in units of Hz.

Returns:
Display object.
"""
height, width, \_ = frames[0].shape
dpi = 70
orig_backend = matplotlib.get_backend()
matplotlib.use('Agg') # Switch to headless 'Agg' to inhibit figure rendering.
fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
matplotlib.use(orig_backend) # Switch back to the original backend.
ax.set_axis_off()
ax.set_aspect('equal')
ax.set_position([0, 0, 1, 1])
im = ax.imshow(frames[0])
def update(frame):
im.set_data(frame)
return [im]
interval = 1000/framerate
anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
interval=interval, blit=True, repeat=False)
return HTML(anim.to_html5_video())

## Part 1

Train the robot to drive to the green goal marker. Use any of the RL algorithms you learnt in the tutorials.

You can save the model after training to save you having to retrain everytime you open colab:

```
from google.colab import drive
drive.mount('/content/drive')
torch.save(model.state_dict(), "/content/drive/My Drive/Colab Notebooks/simple_driving_qlearning.pkl")  # this will save to folder "Colab Notebooks" on your google drive
```

You can then load the model:

```
model.load_state_dict(torch.load("/content/drive/My Drive/Colab Notebooks/simple_driving_qlearning.pkl"))
```

Once loaded you can even continue your training again using the learnt weights, that way you don't have to start from scratch again if you decide you want to train for longer. But keep in mind the epsilon greedy function will start from completely random again so you might want to set epsilon starting value appropriately.

## Part 2

Incorporate prior knowledge into the epsilon-greedy function by choosing a non-uniform distribution to sample from when performing exploration. For example, for training flappy bird we used the following to sample flapping actions less often to avoid flying off into the sky during early stages of training:

```
return np.random.choice(np.array(range(2)), p=[0.9,0.1])
```

Note that you will need to change the parameters to suit the car's action space and also choose a suitable distribution.

## Part 3

Modify the reward to give a bonus of 50 if the goal is reached. You can do this either in the `simulate` function or directly by modifying the `step` function in the gym environment code.

## Part 4

Add obstacles to the environment. You can do this by modifying the `reset` function in the gym environment code. For example you can add objects as follows:

```
self.obstacle = self._p.loadURDF(fileName=<path to urdf file here>,
                   basePosition=[0, 0, 0])
```

An example urdf file: https://github.com/fredsukkar/simple-car-env-template/blob/main/simple_driving/resources/simplegoal.urdf

**Note:** you will need to add features to your state so that the agent learns to avoid obstacles. For example, you could add the x, y distance from the agent to the closest obstacle in the environment. Then your state would become: `[x_goal, y_goal, x_obstacle, y_obstacle]`.

---
#   a i I n r o b o t i c _ q u i z 3  
 