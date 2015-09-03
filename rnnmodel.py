import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class rnnmodel(object):

	def __init__(self, rng, sv, tv, hd):
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

		self.params = [self.inw, self.recurrent, self.outw]

		def recurrence(x_t, h_tm1):
			h_t = T.nnet.sigmoid(x_t+T.dot(h_tm1, self.recurrent))
			s_t = T.nnet.softmax(T.dot(h_t, self.outw))
			return [h_t, s_t]

		inp = T.imatrix()

		y = T.iscalar('y')
		sl = T.iscalar('sl')
		lr = T.scalar('lr')

		x = self.inw[inp].reshape((inp.shape[0], hd*sl))

		[h, s], _ = theano.scan(fn=recurrence, \
			sequences=x, outputs_info=[self.outw, None], \
			n_steps=x.shape[0])

		p_y_given_x_lastword = s[-1,0,:]
		p_y_given_x_sentence = s[:,0,:]
		y_pred = T.argmax(p_y_given_x_sentence, axis=1)

		nll = -T.mean(T.log(p_y_given_x_lastword)[y])
		gradients = T.grad( nll, self.params )
		updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))

		y_pred = T.argmax(p_y_given_x_sentence, axis=1)

		self.prediction = theano.function(inputs = [inp, sl], outputs= y_pred)
		self.train = theano.function(inputs = [inp, y, sl], outputs = nll, updates = updates)






