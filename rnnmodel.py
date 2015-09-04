import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class rnnmodel(object):

	def __init__(self, rng, inp, sv, tv, hd, maxl):
		'''
		rng: numpy.random.RandomState
		sv: source vocabulary size
		tv: target vocabulary size
		hd: dimension of hidden layer
		'''

		self.inw = theano.shared(0.2*numpy.random.uniform(-1,1,(sv,hd))\
			.astype(theano.config.floatX))
		self.recurrent = theano.shared(0.2*numpy.random.uniform(-1,1,(hd,hd))\
			.astype(theano.config.floatX))
		self.outw = theano.shared(0.2*numpy.random.uniform(-1,1,(hd,tv))\
			.astype(theano.config.floatX))
		self.h0 = theano.shared(numpy.zeros(hd, dtype=theano.config.floatX))

		def recurrence(x_t, h_tm1):
			h_t = T.nnet.sigmoid(x_t+T.dot(h_tm1, self.recurrent))
			s_t = T.nnet.softmax(T.dot(h_t, self.outw))
			return [h_t, s_t]
		
		self.input = inp
		x = [self.inw[inp[0]]]
		h = [self.inw[inp[0]]]
		self.p_y_given_x = [T.nnet.softmax(T.dot(h[0], self.outw))]
		self.pred = [T.argmax(self.p_y_given_x[0])]
		for i in xrange(1,maxl):
			x.append(self.inw[inp[i]])
			h.append(x[i]+T.dot(h[i-1], self.recurrent))
			self.p_y_given_x.append(T.nnet.softmax(T.dot(h[i], self.outw)))
			self.pred.append(T.argmax(self.p_y_given_x[i]))
		#self.pred = theano.

		#x = theano.shared(numpy.asarray(x))
		#x = self.inw[inp].reshape((inp.shape[0], hd*sl))

		#[h, s], _ = theano.scan(fn=recurrence, \
		#	sequences=x, outputs_info=[self.h0, None], \
		#	n_steps=x.shape[0])

		#self.p_y_given_x = s[-1,0,:]

		#p_y_given_x_lastword = s[-1,0,:]
		#p_y_given_x_sentence = s[:,0,:]
		#y_pred = T.argmax(p_y_given_x_sentence, axis=1)

		#nll = -T.mean(T.log(p_y_given_x_lastword)[y])
		#gradients = T.grad( nll, self.params )
		#updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))

		#self.prediction = theano.function(inputs = [inp, sl], outputs= y_pred)
		#self.train = theano.function(inputs = [inp, y, sl], outputs = nll, \
		#	updates = updates)
		#self.normalize = theano.function( inputs = [],
        #                 updates = {self.emb:\
        #                 self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
		#return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])



