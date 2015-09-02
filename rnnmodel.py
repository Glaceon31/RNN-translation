import theano
import numpy
import os

from theano import tensor as T

class rnnmodel(object):

	def __init__(self, rng, sv, tv, hd, sl):
		'''
		rng: numpy.random.RandomState
		sv: source vocabulary size
		tv: target vocabulary size
		hd: dimension of hidden layer
		sl: sentence length
		'''

		self.inw = theano.shared(0.2*numpy.random.uniform(-1,1,(vo,hd)).astype(theano.config.floatX))
		self.recurrent = theano.shared(0.2*numpy.random.uniform(-1,1,(hd,hd)).astype(theano.config.floatX))
		self.outw = theano.shared(0.2*numpy.random.uniform(-1,1,(hd,vo)).astype(theano.config.floatX))

		self.params = [self.inw, self.recurrent, self.outw]

		def recurrence(x_t, h_tm1):
			h_t = T.nnet.sigmoid(x_t+T.dot(h_tm1, self.recurrent))
			s_t = T.nnet.softmax(T.dot(h_t, self.outw))
			return [h_t, s_t]

		inp = T.imatrix()
		x = self.inw[inp].reshape((inp.shape[0], hd*sl))
		y = T.iscalar('y')

		lr = T.scalar('lr')
		

		self.prediction = theano.function(input = [inp], output = y_pred)
		self.train = theano.function(input = [inp, y, lr], output = nll, update = updates)






