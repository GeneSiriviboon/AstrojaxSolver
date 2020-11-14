# Python program to illustrate the concept 
# of threading 
# importing the threading module 
import threading 
import numpy as np

def print_cube(num): 
	""" 
	function to print cube of given num 
	"""
	print("Cube: {}".format(num * num * num)) 

def print_square(num): 
	""" 
	function to print square of given num 
	"""
	print("Square: {}".format(num * num)) 

def add_num(array, num, val):
    array[num] = val



if __name__ == "__main__": 
	# creating thread 

    vals = np.arange(100)
    z = np.zeros(100)

    t = []
    for i in range(100):
        t.append(threading.Thread(target=add_num, args=(z, i, i))) 
        
    for i in range(100):
        t[i].start()
    
    for i in range(100):
        t[i].join()

	# both threads completely executed 
    print("Done!", z) 
