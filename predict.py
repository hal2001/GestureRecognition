#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/23/18 11:03 PM 

@author: Hantian Liu
"""

import pickle, os, math, pdb
from sklearn.cluster import KMeans
import numpy as np
from readTxt import filter
from HMM import HMM

############################
## MODIFY THESE VARIABLES ##
############################
foldername="./test_data"
############################

hmmfolder="./models"

N=10
M=60

def motiontest(obs, wavemodel, infmodel, eightmodel, circlemodel, beat3model, beat4model):
	"""MLE on given observance sequence, assign label and give confidence

	:param obs: size n one observation sequence
	:param wavemodel: hmm model
	:param infmodel:
	:param eightmodel:
	:param circlemodel:
	:param beat3model:
	:param beat4model:
	:return: motion [str] the estimation
	         confidence [float] confidence for the estimation
	"""
	# get probabilities under all models
	p = []
	p.append(wavemodel.get_prob(obs))
	p.append(infmodel.get_prob(obs))
	p.append(eightmodel.get_prob(obs))
	p.append(circlemodel.get_prob(obs))
	p.append(beat3model.get_prob(obs))
	p.append(beat4model.get_prob(obs))

	p=np.asarray(p)
	#pdb.set_trace()
	p[np.isnan(p)]=-np.float('inf')

	# find the model with largest probabilities
	motionind = np.argmax(p)
	if motionind == 0:
		motion = "wave"
	elif motionind == 1:
		motion = "inf"
	elif motionind == 2:
		motion = "eight"
	elif motionind == 3:
		motion = "circle"
	elif motionind == 4:
		motion = "beat3"
	elif motionind == 5:
		motion = "beat4"

	# calculate confidence for the estimation
	maxp = np.max(p)
	p[motionind] = -math.inf
	maxp2 = np.max(p)
	#pdb.set_trace()
	confidence = (1/maxp - 1/maxp2) / (1/maxp)

	return motion, confidence, maxp



def predict(foldername):
	"""

	:param foldername: folder of test data
	:return:
	"""

	# load the model from disk
	# cluster model
	modelname = os.path.join(hmmfolder, 'kmeans_model.sav')
	loaded_kmeans = pickle.load(open(modelname, 'rb'))

	# hmm model
	waveA = np.load(os.path.join(hmmfolder, "wave/waveA.npy"))
	waveB = np.load(os.path.join(hmmfolder, "wave/waveB.npy"))
	wavepi = np.load(os.path.join(hmmfolder, "wave/wavepi.npy"))
	wavemodel = HMM(waveA, waveB, wavepi, N, M)
	infA = np.load(os.path.join(hmmfolder, "inf/infA.npy"))
	infB = np.load(os.path.join(hmmfolder, "inf/infB.npy"))
	infpi = np.load(os.path.join(hmmfolder, "inf/infpi.npy"))
	infmodel = HMM(infA, infB, infpi, N, M)
	eightA = np.load(os.path.join(hmmfolder, "eight/eightA.npy"))
	eightB = np.load(os.path.join(hmmfolder, "eight/eightB.npy"))
	eightpi = np.load(os.path.join(hmmfolder, "eight/eightpi.npy"))
	eightmodel = HMM(eightA, eightB, eightpi, N, M)
	circleA = np.load(os.path.join(hmmfolder, "circle/circleA.npy"))
	circleB = np.load(os.path.join(hmmfolder, "circle/circleB.npy"))
	circlepi = np.load(os.path.join(hmmfolder, "circle/circlepi.npy"))
	circlemodel = HMM(circleA, circleB, circlepi, N, M)
	beat3A = np.load(os.path.join(hmmfolder, "beat3/beat3A.npy"))
	beat3B = np.load(os.path.join(hmmfolder, "beat3/beat3B.npy"))
	beat3pi = np.load(os.path.join(hmmfolder, "beat3/beat3pi.npy"))
	beat3model = HMM(beat3A, beat3B, beat3pi, N, M)
	beat4A = np.load(os.path.join(hmmfolder, "beat4/beat4A.npy"))
	beat4B = np.load(os.path.join(hmmfolder, "beat4/beat4B.npy"))
	beat4pi = np.load(os.path.join(hmmfolder, "beat4/beat4pi.npy"))
	beat4model = HMM(beat4A, beat4B, beat4pi, N, M)

	# load test data from folder and predict
	# one by one
	for filename in os.listdir(foldername):
		if filename == ".DS_Store":
			continue
		a = []
		with open(os.path.join(foldername, filename)) as f:
			# read lines in file
			for line in f:
				line = line.strip("\n")
				line = line.split("\t")
				a.append(np.asarray(line))
		a = np.asarray(a)
		a = a.astype(np.float)
		'''
		# filter data using UKF
		ts = a[:, 0]
		accx = a[:, 1]
		accy = a[:, 2]
		accz = a[:, 3]
		omgx = a[:, 4]
		omgy = a[:, 5]
		omgz = a[:, 6]
		data=filter(ts, omgx, omgy, omgz, accx, accy, accz) #TODO
		'''
		data = a[:, 1:]
		obs=loaded_kmeans.predict(data)
		motiontype, confidence, maxp=motiontest(obs, wavemodel, infmodel, eightmodel, circlemodel, beat3model, beat4model)
		print("\n"+filename+" belongs to the motion of "+motiontype+\
			 "    with maximum log likelihood of "+str(maxp)+\
			  "    with confidence of "+str(confidence))


if __name__ == '__main__':
	predict(foldername)


