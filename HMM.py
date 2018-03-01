#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/20/18 6:36 PM 

@author: Hantian Liu
"""

import numpy as np
from readTxt import cluster
import pdb
import _pickle as cPickle
import pickle


class HMM(object):
	"""left-to-right Hidden Markov model

	Attributes:
		N: # of hidden states
		M: # of observation classes
		A: N*N state transition probability matrix
		B: M*N emission probability matrix
		pi: N*1 initial state probability matrix
	"""

	def __init__(self, A, B, pi, N, M):
		"""initialization"""
		self.A=A
		self.B=B
		self.pi=pi
		self.N = N
		self.M = M

	def forward(self, obs):
		"""get forward variable for the observation sequence

		:param obs: n
		:return: alpha T*N, ct T*1 scaling coefficient
		"""
		T=len(obs)
		alpha=np.zeros([T, self.N])
		ct=np.zeros([T,1])
		alpha[0,:]=self.pi.transpose()*self.B[obs[0], :]
		ct[0,:]=1/np.sum(alpha[0,:])
		alpha[0, :] = alpha[0, :] * ct[0,:]
		for t in range(T-1):
			#a=sum(np.tile(np.transpose(alpha[t, :]), (self.N, 1))*self.A)
			#alpha0[t+1,:]= a[np.newaxis,:]*self.B[obs[t+1], :]

			asum=np.dot(alpha[t:t+1,:], self.A)
			asum=asum[np.newaxis, :]
			alpha[t+1,:]= asum*self.B[obs[t+1],:]
			'''
			asum=0
			for j in range(self.N):
				for i in range(self.N):
					asum=asum + alpha[t,i]*self.A[i,j]
				alpha[t+1,j]=asum*self.B[obs[t+1], j]
			'''
			#pdb.set_trace()
			ct[t + 1, :] = 1 /np.sum(alpha[t + 1, :])
			alpha[t+1,:]=alpha[t+1,:]*ct[t+1,:]
		return alpha, ct

	def get_prob(self, obs):
		"""get probability of the observance sequence given current HMM model

		:param obs: n
		:return: log likelihood
		"""
		T = len(obs)
		alpha, ct=self.forward(obs)
		logct=np.log(ct)
		loglk=-np.sum(logct)
		return loglk #sum(alpha[T-1, :])

	def backward(self, obs, ct):
		"""get backward variable for the observation sequence

		:param obs: n
		:param ct: T*1
		:return: beta T*N
		"""
		T = len(obs)
		beta = np.zeros([T, self.N])
		beta[T-1, :] = 1
		beta[T-1, :]=ct[T-1,:]*beta[T-1,:]
		for t_inv in range(1, T):
			t=T-1-t_inv
			b= np.dot(self.A, (self.B[obs[t+1],:]*beta[t+1, :]).transpose())
			#pdb.set_trace()
			beta[t,:]=b.transpose()
			#beta[t:t+1,:]=beta[t:t+1,:]/np.sum(beta[t:t+1,:])
			beta[t,:]=ct[t,:]*beta[t,:]
		return beta

	def get_zeta(self, obs):
		"""get probability of joint event, given observance sequence,
			being in state Si at time t, and state Sj at time t+1

		:param obs: n
		:return: N*N*T
		"""

		T=len(obs)
		alpha, ct=self.forward(obs)
		beta=self.backward(obs, ct)
		zeta=np.zeros([self.N, self.N, T])
		for t in range(T-1):
			for i in range(self.N):
				for j in range(self.N):
					zeta[i, j, t]=alpha[t,i]*self.A[i,j]*self.B[obs[t+1],j]*beta[t+1,j]
			#pdb.set_trace()
			zeta[:, :, t] = zeta[:, :, t] / np.sum(np.sum(zeta[:, :, t]))
		return zeta, alpha, beta

	def get_gamma(self, obs, zeta, alpha, beta):
		"""get probability of being in state Si at time t, given observance sequence

		:param obs: n
		:param zeta: N*N*T
		:return: 1*N*T
		"""

		T = len(obs)
		gamma = np.zeros([1, self.N, T])
		#gamma0 = np.zeros([1, self.N, T])
		#alpha, ct=self.forward(obs)
		#beta=self.backward(obs, ct)

		#zeta=self.get_zeta(obs)
		for t in range(T-1):
			g=np.sum(zeta[:,:,t], axis=1)
			gamma[:,:,t]=g[np.newaxis, :]
			#gamma[:, :, t] = alpha[t, :] * beta[t, :]
			#gamma[:, :, t] = gamma[:, :, t] / np.sum(gamma[:, :, t])
		return gamma


	def update(self, obs_seq):
		"""update HMM parameters

		:param obs_seq: list of all observance sequences
		:return:
		"""

		pimat=np.zeros([self.N,1])
		K=len(obs_seq)

		nume_A = np.zeros([self.N, self.N])
		deno_A = np.zeros([self.N, self.N])

		nume_B = np.zeros([self.M, self.N])
		deno_B = np.zeros([self.M, self.N])

		for obs in obs_seq:

			#prob_k = self.get_prob(obs)
			zeta, alpha, beta=self.get_zeta(obs)
			zeta_k = np.sum(zeta[:, :, :-1], axis = 2)  # t=1,...,T-1
			gamma_k=self.get_gamma(obs, zeta, alpha, beta)

			# update pi
			pimat=pimat+gamma_k[:,:,0].transpose()

			#update A
			gamma_k_asum = np.sum(gamma_k[:, :, :-1], axis = 2)
			gamma_k_asum = np.tile(gamma_k_asum.transpose(), (1, self.N))
			nume_A = nume_A + zeta_k
			deno_A = deno_A + gamma_k_asum

			#update B
			for i in range(self.M):
				b=np.sum(gamma_k[:,:,obs==i], axis=2)
				nume_B[i,:]=nume_B[i,:]+b[np.newaxis, :]
			gamma_k_bsum = np.sum(gamma_k, axis = 2)
			gamma_k_bsum = np.tile(gamma_k_bsum, (self.M, 1))
			deno_B=deno_B+gamma_k_bsum

		self.pi = pimat/K #/ np.linalg.norm(pimat)

		#deno_A[deno_A <= 1e-12] = 1e-12
		A = nume_A / deno_A

		#adiv = np.tile(np.sum(A, axis = 1)[:,np.newaxis], (1,self.N))
		#adiv[adiv <= 1e-12] = 1e-12
		#A = A / adiv
		self.A = A
		#print(np.sum(A, axis=1))

		#deno_B[deno_B <= 1e-12] = 1e-12
		B = nume_B / deno_B
		#pdb.set_trace()
		#print(np.sum(B, axis=0))

		#bdiv = np.tile(np.sum(B, axis = 0)[np.newaxis,:], (self.M,1))
		#bdiv[bdiv <= 1e-12] = 1e-12
		#B = B / bdiv

		B[B <= 1e-12] = 1e-12
		self.B = B
		#pdb.set_trace()



	def save(self, name):
		"""save the HMM parameters of current model (A, B, pi)

		:param name: [str] motion name to save as
		:return:
		"""
		'''
		"""save class as name.pickle"""
		path=name + '.pickle'

		save_dict = {'A': self.A, 'B': self.B, 'pi': self.pi, 'N':self.N, \
					 'M': self.M}
		with open(str(path), 'wb') as handle:
			pickle.dump(save_dict, handle)
		'''
		np.save(name+'A.npy',self.A)
		np.save(name + 'B.npy', self.B)
		np.save(name + 'pi.npy', self.pi)
		#np.save(name + 'N.npy', self.N)
		#np.save(name + 'M.npy', self.M)
		print("model successfully saved!")


	def load(self, name):
		"""load name.pickle"""
		path=name + '.pickle'
		with open(str(path), 'rb') as handle:
			save_dict = pickle.load(handle)

		self.A=save_dict['A']
		self.B=save_dict['B']
		self.pi=save_dict['pi']
		self.N=save_dict['N']
		self.M=save_dict['M']

		#loaded_objects = []
		#for i in range(3):
		#	loaded_objects.append(cPickle.load(f))
		#file.close()
		print("model successfully loaded!")



if __name__ == '__main__':
	N = 10
	M = 30
	'''
	obs_w, obs_i, obs_e, obs_b3, obs_b4, obs_c = cluster("./train_data", M)

	obs_i = np.load('obs_i.npy')
	obs_e = np.load('obs_e.npy')
	obs_c = np.load('obs_c.npy')
	obs_b3 = np.load('obs_b3.npy')
	obs_b4 = np.load('obs_b4.npy')
	'''
	obs_w=np.load('obs_w.npy')


	#pdb.set_trace()
	#A=np.random.randint(N*N, size=(N,N))
	#A = A / np.tile(np.sum(A, axis = 1)[:,np.newaxis], (1,N))
	A=np.zeros([N,N])
	#for row in range(N):
	#	A[row, row:min(N,row+2)]=1/(min(N,row+2)-row)
	r=np.arange(N)
	c=np.arange(N)
	A[r,c]=0.9
	c=c+1
	c[-1]=0
	A[r,c]=0.1
	#B = np.random.randint(M*N, size=(M,N))
	B = np.ones([M, N])
	B=B/np.tile(np.sum(B, axis = 0)[np.newaxis,:], (M,1))
	pi=np.ones([N, 1])*1/N

	obs_train=obs_w
	obs_test=obs_w[6]

	wave=HMM(A,B,pi,N,M)

	for epoch in range(50):
		print('\nepoch'+str(epoch))
		wave.update(obs_train)
		
		counter=0
		for obs in obs_w:
			counter=counter+1
			print('obs NO. '+str(counter))
			ll=wave.get_prob(obs)
			print('loglikelihood: '+str(ll))

	wave.save("wavetest")
	'''

	wave=HMM(A,B,pi,N,M)
	wave.load("wave")
	print("\n\n\nnew wave loaded")

	counter = 0
	for obs in obs_w:
		counter = counter + 1
		print('obs NO. ' + str(counter))
		ll = wave.get_prob(obs)
		print('loglikelihood: ' + str(ll))
	'''




