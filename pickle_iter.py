#Only have to do this the first time Loadit is set to 1
#Do this in python command line (line by line)
import os 
import pickle

#check current directory 
print(os.getcwd())

#list files in directory 
print(os.listdir())
#index to pkl file 
print(os.listdir()[2])
#store name to variable (have to do this because an error comes up if manually typed)
file = os.listdir()[2]

#first time entering iter 
dict = {'iter': 33451}
f = open("/data/lung_seg/model/{}".format(file),"wb")
pickle.dump(dict, f)
f.close()
