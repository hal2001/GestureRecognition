#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/23/18 6:46 PM 

@author: Hantian Liu
"""

import numpy as np
from HMM import HMM
from readTxt import cluster
import io, math


N = 10
M = 90


def init(T):
	"""initialize all parameters for HMM

	:return: A N*N prev state*current state,
	         B M*N obs*state,
	         pi N*1 state*1
	"""
	#A=np.random.randint(N*N, size=(N,N))
	#A = A / np.tile(np.sum(A, axis = 1)[:,np.newaxis], (1,N))
	A=np.zeros([N,N])
	# entries on the diagonal equal to a, and next entry equal to 1-a
	#T = 2000  # avg length of observation sequence
	a = 1 - N / T
	#a = 0.7
	#b= 0.2
	b = (1-a)/3*2

	for row in range(N):
		A[row, row]=a
		if row+1>N-1:
			A[row, 0]=b
		else:
			A[row, row+1]=b
		if row+2>N-1:
			A[row, row]=A[row, row]+(1-a-b)
		else:
			A[row, row+2]=1-a-b


	#B = np.random.randint(N, size=(M,N))
	B = np.ones([M, N])
	B=B/np.tile(np.sum(B, axis = 0)[np.newaxis,:], (M,1))

	#pi=np.random.randint(N, size=(N,1))
	#pi=pi/np.linalg.norm(pi)
	pi=np.ones([N, 1])*1/N

	return A, B, pi


def train(motionname, motion_obs_seq):
	"""

	:param motionname: [str]
	:param motion_obs_seq: list of all observation sequences under the choice of motion
	:return:
	"""
	totlen=0
	for obs in motion_obs_seq:
		totlen=totlen+len(obs)
	avglen=totlen/len(motion_obs_seq)

	# initialize
	A,B,pi=init(avglen)

	obs_train = motion_obs_seq
	hmmmodel = HMM(A, B, pi, N, M)
	print("\n\nStart to train "+str(motionname)+" model!")

	# EM
	max_epoch = 200 #TODO
	tolerance=0.000005
	epoch = 0
	tot=0
	tot_prev=-math.inf
	while epoch <= max_epoch: #and tot-tot_prev>=tolerance:
		if epoch>0:
			tot_prev=tot
		hmmmodel.update(obs_train)
		counter = 0
		tot = 0
		for obs in obs:
			counter = counter + 1
			ll = hmmmodel.get_prob(obs)
			#print('obs NO. ' + str(counter)+'loglikelihood: ' + str(ll))
			tot = tot + ll

		#total likelihood
		print('epoch' + str(epoch)+'   total loglikelihood ' + str(tot))
		epoch = epoch + 1

	# save model
	hmmmodel.save(motionname)

	'''
	tswave=HMM(A,B,pi,N,M)
	tswave.load("wave")
	counter=0
	for obs in obs_w:
		counter=counter+1
		print('obs NO. '+str(counter))
		ll=tswave.get_prob(obs)
		print('loglikelihood: '+str(ll))
	'''


if __name__ == '__main__':
	# get hmm models
	# obs_w, obs_i, obs_e, obs_b3, obs_b4, obs_c = cluster("./train_data", M)
	obs_w = np.load('./models/obs_w.npy')
	obs_i = np.load('./models/obs_i.npy')
	obs_e = np.load('./models/obs_e.npy')
	obs_c = np.load('./models/obs_c.npy')
	obs_b3 = np.load('./models/obs_b3.npy')
	obs_b4 = np.load('./models/obs_b4.npy')

	train("wave", obs_w)
	train("inf", obs_i)
	train("eight", obs_e)
	train("circle", obs_c)
	train("beat3", obs_b3)
	train("beat4", obs_b4)
