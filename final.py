import numpy as np
from numpy import *;
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
import pandas as pd
 


# read the original data file
df=pd.read_csv("mask.csv",delimiter=",")
df['Date'] = pd.to_datetime(df['Date'])
ca_pre = df[df['state'] == 'Connecticut']
state = df['state'].unique()
i1=0


# calculate the distance between two vectors
def distance(e1, e2):
  return np.sqrt((e1[0]-e2[0])**2+(e1[1]-e2[1])**2)
 
    
# calculate the centre of each cluster
def means(arr):
  return np.array([np.mean([e[0] for e in arr]), np.mean([e[1] for e in arr])])
 
# looking for the element farthest to A in the array, initializing the cluster centre
def farthest(k_arr, arr):
  f = [0, 0]
  max_d = 0
  for e in arr:
    d = 0
    for i in range(k_arr.__len__()):
      d = d + np.sqrt(distance(k_arr[i], e))
    if d > max_d:
      max_d = d
      f = e
  return f
 
# looking for the element closest to A in the array
def closest(a, arr):
  c = arr[1]
  min_d = distance(a, arr[1])
  arr = arr[1:]
  for e in arr:
    d = distance(a, e)
    if d < min_d:
      min_d = d
      c = e
  return c


# using LES algorithm from HW2 to fit the curve
def Q1_1(X,Y):
    t=np.ones(len(X))
    h = np.vstack((X,t)).T
    Y = np.mat(Y)
    try:        # to avoid the singularity matrix problem caused by too few single group data after classification
        A=np.matmul(np.matmul(inv(np.matmul(h.T,h)),h.T),Y.T)   
        v=Y-np.matmul(h,A)
    except:
        return [0,0]
    else:
        return A


# kmeans clustering and showing the result
def kmeans(ca_pre,state_name):
  x=[i for i in range(0,len(ca_pre))]
  A  =  np.column_stack((x,ca_pre['cases']))
  arr = A   # import original dataset
 
  ## initialize the cluster center and cluster container
  m = 2
  r = np.random.randint(arr.__len__() - 1)
  k_arr = np.array([arr[r]])
  cla_arr = [[]]
  for i in range(m-1):
    k = farthest(k_arr, arr)
    k_arr = np.concatenate([k_arr, np.array([k])])
    cla_arr.append([])
 
  ## iterative clustering
  n = 50
  cla_temp = cla_arr
  for i in range(n):  # do n iterations
    for e in arr:  # clusters each element in the set to the nearest array
      ki = 0    # assume that the first one is closest
      min_d = distance(e, k_arr[ki])
      for j in range(1, k_arr.__len__()):
        if distance(e, k_arr[j]) < min_d:  # find a closer cluster center
          min_d = distance(e, k_arr[j])
          ki = j
      cla_temp[ki].append(e)
    # update the cluster centre during iteration
    for k in range(k_arr.__len__()):
      if n - 1 == i:
        break
      k_arr[k] = means(cla_temp[k])
      cla_temp[k] = []
 
  ## visualization
  col = ['HotPink', 'Aqua']
  Xi  =  []
  Yi  =  []
  Xi1 =  []
  Yi1 =  []
  cla0 = np.mat(cla_arr[0])
  for i in range(len(cla0)):
    if   i !=  0 :
        Xi.append(float(cla0[i,0]))
        Yi.append(float(cla0[i,1]))
  
  A1 = Q1_1(Xi,Yi)
  print("Group 1 result:",A1) # the slope and the intercept of group1
  
  cla1 = np.mat(cla_arr[1])
  for i in range(len(cla1)):
    if   i !=  0 :
        Xi1.append(float(cla1[i,0]))
        Yi1.append(float(cla1[i,1]))
  
  A2 = Q1_1(Xi1,Yi1)
  print("Group 2 result:",A2) # the slope and the intercept of group2
  # write the results to a text document for further analyzing
  with open('infection rate kmeans.txt','a',encoding='utf-8') as f:
    f.write("state:")
    f.write(state_name)
    f.write("K2-K1= ")
    f.write(str(A2[0]-A1[0]))
    f.write('\r\n')
  print("state:",state_name,"K2-K1= ",A2[0]-A1[0])
  
  for i in range(m):
    plt.scatter(k_arr[i][0], k_arr[i][1], linewidth=10, color=col[i])
    plt.scatter([e[0] for e in cla_temp[i]], [e[1] for e in cla_temp[i]], color=col[i])
    plt.show()
    

    
def cal(state_name,df):
    states_pre = df[df['state'] == state_name]  #filter for state (before)
    kmeans(states_pre,state_name)
    
    
    


for s in state:
    cal(s,df)
    i1=i1+1
    
