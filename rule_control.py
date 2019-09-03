unit = 0.01
import gym
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

env_id = 'gym_energyplus:EnergyPlus-v0'
env = gym.make(env_id)

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
	
	w.append(wt)
	e.append(et)
	p.append(hvac)
	itp.append(it)
	out.append(ot)
	
	action = np.array([-0.2,-0.2,0.5,0.5])
	
	if wt > 20: 
		action[0] = -0.3
		action[2] = 0.6
	if wt > 22: 
		action[0] = -0.4
		action[2] = 0.7	
	if wt > 24: 
		action[0] = -0.5
		action[2] = 0.8	
	if wt > 26: 
		action[0] = -0.6
		action[2] = 0.9
	if wt > 28: 
		action[0] = -0.8
		action[2] = 0.95
	if wt > 30: 
		action[0] = -1.0
		action[2] = 1.0
		
	if et > 20: 
		action[1] = -0.3
		action[3] = 0.6
	if et > 22: 
		action[1] = -0.4
		action[3] = 0.7	
	if et > 24: 
		action[1] = -0.5
		action[3] = 0.8	
	if et > 26: 
		action[1] = -0.6
		action[3] = 0.9
	if et > 28: 
		action[1] = -0.8
		action[3] = 0.95
	if et > 30: 
		action[1] = -1.0
		action[3] = 1.0
		
	if it > 8:
		action += np.array([-0.05,-0.05, 0.05, 0.05])
	if it > 10:
		action += np.array([-0.03,-0.03, 0.03, 0.03])
		
	if ot > 10:
		action += np.array([-0.04,-0.04, -0.01, -0.01])
		
	action += np.array([-0.2,-0.2, 0.0, 0.0])
		
	observation, reward, done, info = env.step(action)	
	print(action)

plt.plot(w, label='west zone temp')
plt.plot(e, label='east zone temp')
plt.plot(p, label='HVAC power consumption')


plt.plot(itp, label='IT equip power consumption', linestyle=':')
plt.plot(out, label='outdoor temp')

plt.legend()
name = 'env_test.png'
plt.savefig(name)

print('save simulation result as : '+name)
print('HVAC power consumption per day is : ', sum(p)/float(day))
print('west zone temp well controlled : ',cw/(96.0*day))
print('east zone temp well controlled : ',ce/(96.0*day))

##/home/hchonglondon//working_space/rl-testbed-for-energyplus/env_test.png