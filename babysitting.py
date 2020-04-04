import pickle
import matplotlib.pyplot as plt
import sys
import os




histories = pickle.load(open(sys.argv[1],"rb"), encoding='latin1')

histories_mod = dict(histories)

metrics = ['episode_reward','episode_len','max_Q']
buffer_ = []
k = 400

for index,(f_name,values) in enumerate(histories.items()):
	if f_name in metrics:
		for i in range(len(values)-k+1):
			buffer_.append(sum(values[i:k+i])/k)
		histories_mod[f_name+'_r'] = list(buffer_)
	buffer_ = []


for index,(f_name,values) in enumerate(histories_mod.items()):
	plt.subplot(len(histories_mod),1,index+1)	
	plt.plot(values)
	plt.ylabel(f_name)
	plt.xlim(0,50000)
	#plt.xlabel('Training steps')

plt.show()