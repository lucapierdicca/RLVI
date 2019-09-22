import pickle
import matplotlib.pyplot as plt
import sys
import os




histories = pickle.load(open(sys.argv[1],"rb"), encoding='latin1')

print(histories)

for index,(f_name,values) in enumerate(histories.items()):
	plt.subplot(4,1,index+1)
		
	plt.plot(values)
	plt.ylabel(f_name)
	#plt.xlabel('Training steps')

plt.show()