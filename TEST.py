import torch
mlp = torch.load('/home/hchonglondon/working_space/policy-gradient/TRAINED_MODEL')

import gym
import numpy as np
import matplotlib.pylot as plt

import pandas as pd

env = gym.make('gym_datacenter:datacenter-v0')
observation = env.reset()

Tout = []
Tz1 = []
Tz2 = []
for i in range(4*24*7):
	ob = np.array(observation, dtype=np.float32)
	x = torch.from_numpy(ob).view(-1,6)
	y = mlp(x)
	[action] = y.cpu().detach.numpy()
    observation, reward, done, info = env.step(action)
	Tout.append(observation[0])
	Tz1.append(observation[1])
	Tz2.append(observation[2])

plt.xlabel('time step / every 15 minutes')
plt.ylabel('centigrade degrees, ten kilowatts')
plt.plot(Tout)
plt.plot(Tz1)
plt.plot(Tz2)
plt.save('test.png')
