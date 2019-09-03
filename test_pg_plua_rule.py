unit = 0.01

import torch
mlp = torch.load('/home/hchonglondon/working_space/rl-testbed-for-energyplus/TRAINED_MODEL')

import gym
import numpy as np
import matplotlib.pyplot as plt
# add for remote server
plt.switch_backend('agg') 

env = gym.make('gym_energyplus:EnergyPlus-v0')
observation = env.reset()

plt.rcParams['figure.figsize'] = [18, 7]
plt.xlabel('time step (every 15 minutes)')
plt.ylabel('centigrade degrees, 1e4 watts')


w = []
e = []
p = []

itp = []
out = []

cw = 0
ce = 0

observation = env.reset()
day = 31
for i in range(96*day):
	ot = observation[0]/unit
	wt = observation[1]/unit
	et = observation[2]/unit
	it = observation[4]/unit
	hvac = observation[5]/unit

	if wt>20 and wt<24: cw +=1
	if et>20 and et<24: ce +=1

	ob = np.array(observation, dtype=np.float32)
	x = torch.from_numpy(ob).view(-1,6)
	y = mlp(x)
	action = y.data.cpu().numpy()[0]
	
	if wt > 25: 
		action[0] = -0.8
		action[2] = 1.0
	if et > 25: 
		action[1] = -0.8
		action[3] = 1.0

	if wt < 20:
		action[0] = -0.2
		action[2] = 0.5
	if et < 20:
		action[1] = -0.2
		action[3] = 0.5
		
	print(action)
	
	observation, reward, done, info = env.step(action)
	
	w.append(wt)
	e.append(et)
	p.append(hvac)
	itp.append(it)
	out.append(ot)


plt.plot(w, label='west zone temp')
plt.plot(e, label='east zone temp')
plt.plot(p, label='HVAC power consumption')


plt.plot(itp, label='IT equip power consumption', linestyle=':')
plt.plot(out, label='outdoor temp')

plt.legend()
name = 'test_pg.png'
plt.savefig(name)

print('save simulation result as : '+name)
print('HVAC power consumption per day is : ', sum(p)/float(day))
print('west zone temp well controlled : ',cw/(96.0*day))
print('east zone temp well controlled : ',ce/(96.0*day))

##/home/hchonglondon//working_space/rl-testbed-for-energyplus/test_pg.png
