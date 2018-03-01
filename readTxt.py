#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/20/18 11:37 AM 

@author: Hantian Liu
"""

import os, csv, transforms3d, pickle
from UKF import UKF, init
from utils import quart2mat
import numpy as np
from sklearn.cluster import KMeans

foldername="./train_data"
choice="w"

def filter(ts, omgx, omgy, omgz, accx, accy, accz):
	"""
	process data with UKF, to get filtered data in the form of Euler angles
	:param ts: [float] n
	:param omgx: [float] n angular velocity
	:param omgy: [float] n
	:param omgz: [float] n
	:param accx: [float] n
	:param accy: [float] n
	:param accz: [float] n
	:return: data [float] n*3
	"""
	mu, sigma = init()
	data = np.zeros([len(ts), 3])
	for i in range(len(ts)):
		mu, sigma, q_ukf = UKF(omgx, omgy, omgz, ts, i, mu, sigma, True, accx, accy, accz)
		# q.append(q_ukf)
		my_mat = quart2mat(q_ukf)
		y, p, r = transforms3d.euler.mat2euler(my_mat, axes = 'szyx')
		data[i, 0] = y
		data[i, 1] = p
		data[i, 2] = r
	return data


def txttoData(foldername, choice, is_beat3, is_beat4):
	"""
	read txt files, i.e. training data, under certain motion of interest,
	to get data representing orientations in Euler angles (if filtered), or in original form (if not filtered)
	:param foldername: [str] folder name
	:param choice: [str] initial of the motion name
	:param is_beat3: [boolean] to distinguish beat3 and beat4
	:param is_beat4: [boolean]
	:return: data_choice [float] n*6
	         length [float] n : record corresponding length of the data
	"""
	#add first row for future stack
	data_choice = np.zeros([1,6]) #TODO
	length= []
	for filename in os.listdir(foldername):
		#find the motion of interest via initial
		if filename[0]==choice:
			# distinguish beat3 and beat4 in "b"
			if is_beat3==False and is_beat4==False:
				a = []
				with open(os.path.join(foldername, filename)) as f:
					#read lines in file
					for line in f:
						line = line.strip("\n")
						line = line.split("\t")
						a.append(np.asarray(line))
				a = np.asarray(a)
				a = a.astype(np.float)
				ts = a[:, 0]
				accx = a[:, 1]
				accy = a[:, 2]
				accz = a[:, 3]
				omgx = a[:, 4]
				omgy = a[:, 5]
				omgz = a[:, 6]

				data = a[:,1:]
				#filter data using UKF
				#data=filter(ts, omgx, omgy, omgz, accx, accy, accz) #TODO
				#append it to list of all data
				data_choice=np.vstack((data_choice, data))
				length.append(len(data))

			#distinguish beat3 and beat4 in "b"
			elif is_beat3==True:
				if filename[4]=="3":
					a = []
					with open(os.path.join(foldername, filename)) as f:
						# read lines in file
						for line in f:
							line = line.strip("\n")
							line = line.split("\t")
							a.append(np.asarray(line))
					a = np.asarray(a)
					a = a.astype(np.float)
					ts = a[:, 0]
					accx = a[:, 1]
					accy = a[:, 2]
					accz = a[:, 3]
					omgx = a[:, 4]
					omgy = a[:, 5]
					omgz = a[:, 6]

					data = a[:, 1:]
					# filter data using UKF
					#data = filter(ts, omgx, omgy, omgz, accx, accy, accz)
					# append it to list of all data
					data_choice = np.vstack((data_choice, data))
					length.append(len(data))

			# distinguish beat3 and beat4 in "b"
			elif is_beat4==True:
				if filename[4] == "4":
					a = []
					with open(os.path.join(foldername, filename)) as f:
						# read lines in file
						for line in f:
							line = line.strip("\n")
							line = line.split("\t")
							a.append(np.asarray(line))
					a = np.asarray(a)
					a = a.astype(np.float)
					ts = a[:, 0]
					accx = a[:, 1]
					accy = a[:, 2]
					accz = a[:, 3]
					omgx = a[:, 4]
					omgy = a[:, 5]
					omgz = a[:, 6]

					data = a[:, 1:]
					# filter data using UKF
					#data = filter(ts, omgx, omgy, omgz, accx, accy, accz)
					# append it to list of all data
					data_choice = np.vstack((data_choice, data))
					length.append(len(data))
	#delete first row
	data_choice = data_choice[1:,:]
	#turn the list to array
	length=np.asarray(length)

	return data_choice, length


def cluster(foldername, M):
	"""
	read all data under folder, cluster and assign label as the observation sequence for all the data sets
	:param foldername: [str]
	:param M: # of observation classes i.e. # of clusters
	:return: each motion is saved as a list, consists of all corresponding observance sequences as array
	"""
	#read all data from txt
	data_w, len_w = txttoData(foldername, "w", False, False)
	data_i, len_i = txttoData(foldername, "i", False, False)
	data_e, len_e = txttoData(foldername, "e", False, False)
	data_b3, len_b3 = txttoData(foldername, "b", True, False)
	data_b4, len_b4 = txttoData(foldername, "b", False, True)
	data_c, len_c = txttoData(foldername, "c", False, False)
	#stack sequence = label sequence
	all_data=np.vstack((data_w,data_i,data_e, data_b3, data_b4, data_c))

	#cluster
	kmeans=KMeans(n_clusters=M).fit(all_data)
	labels=kmeans.labels_

	# save the model to disk
	filename = 'kmeans_model.sav'
	pickle.dump(kmeans, open(filename, 'wb'))

	#assign labels to each type of motion
	#obtain discreted observation sequence
	obs_w=[]
	counter=0
	for i in range(len(len_w)):
		obs_w.append(labels[counter:counter+len_w[i]])
		counter=counter+len_w[i]
	obs_i = []
	for i in range(len(len_i)):
		obs_i.append(labels[counter:counter + len_i[i]])
		counter = counter + len_i[i]
	obs_e = []
	for i in range(len(len_e)):
		obs_e.append(labels[counter:counter + len_e[i]])
		counter = counter + len_e[i]
	obs_b3 = []
	for i in range(len(len_b3)):
		obs_b3.append(labels[counter:counter + len_b3[i]])
		counter = counter + len_b3[i]
	obs_b4 = []
	for i in range(len(len_b4)):
		obs_b4.append(labels[counter:counter + len_b4[i]])
		counter = counter + len_b4[i]
	obs_c = []
	for i in range(len(len_c)):
		obs_c.append(labels[counter:counter + len_c[i]])
		counter = counter + len_c[i]

	return obs_w, obs_i, obs_e, obs_b3, obs_b4, obs_c


if __name__ == '__main__':
	M = 90
	obs_w, obs_i, obs_e, obs_b3, obs_b4, obs_c=cluster(foldername, M)
	np.save('obs_w.npy', np.asarray(obs_w))
	np.save('obs_i.npy', np.asarray(obs_i))
	np.save('obs_e.npy', np.asarray(obs_e))
	np.save('obs_b3.npy', np.asarray(obs_b3))
	np.save('obs_b4.npy', np.asarray(obs_b4))
	np.save('obs_c.npy', np.asarray(obs_c))




