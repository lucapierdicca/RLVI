import pickle
import matplotlib.pyplot as plt
import sys



 
histories = pickle.load(open("histories.pickle","rb"))

for index,(f_name,values) in enumerate(histories.items()):
	plt.subplot(1,3,index+1)
	plt.plot(values)
	plt.ylabel(f_name)
	#plt.xlabel('Training steps')

plt.show()