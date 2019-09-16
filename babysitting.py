import pickle
import matplotlib.pyplot as plt
import sys
import os


histories = pickle.load(open("histories.pickle","rb"), encoding='latin1')

for index,(f_name,values) in enumerate(histories.items()):
	plt.subplot(1,3,index+1)
	if f_name == 'cost':
		values = values[50:]
	if f_name == 'max_Q':
		values = values[400:]
		
	plt.plot(values)
	plt.ylabel(f_name)
	#plt.xlabel('Training steps')

plt.show()