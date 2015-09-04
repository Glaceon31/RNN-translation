#-*- coding: utf-8 -*-
from rnnmodel import rnnmodel

import numpy
import time
import sys
import subprocess
import os
import random

import theano
from theano import tensor as T

if __name__ == '__main__':

	learning_rate = 0.1
	train_set = [[['I', 'like', 'music', '</eos>'],['I', 'don\'t', 'like', 'painting', '</eos>']],[['我','喜欢','音乐', '</eos>'],['我','不','喜欢','画画', '</eos>']]]
	valid_set = [[['I', 'don\'t', 'like', 'music', '</eos>']],[['我','不','喜欢','音乐', '</eos>']]]
	test_set = [[['I', 'like', 'painting', '</eos>']],[['我','喜欢','画画', '</eos>']]]

	sourcedict = ['</eos>']
	targetdict = ['</eos>']

	train_input, train_output = train_set
	valid_input, valid_output = valid_set
	test_input, test_output = test_set

	maxslength = 0
	maxtlength = 0

	#construct dictionary
	for input_set in [train_input, valid_input, test_input]:
		for sentence in input_set:
			if len(sentence) > maxslength:
				maxslength = len(sentence)
			for i in range(0,len(sentence)):
				word = sentence[i]
				if not word in sourcedict:
					sourcedict.append(word)
				sentence[i] = sourcedict.index(word)

	for output_set in [train_output, valid_output, test_output]:
		for sentence in output_set:
			if len(sentence) > maxtlength:
				maxtlength = len(sentence)
			for i in range(0,len(sentence)):
				word = sentence[i]
				if not word in targetdict:
					targetdict.append(word)
				sentence[i] = targetdict.index(word)

	print maxslength, maxtlength
	for input_set in [train_input, valid_input, test_input]:
		for i in xrange(len(input_set)):
			sentence = input_set[i]
			if len(sentence) < maxslength:
				for i in xrange(maxslength-len(sentence)):
					sentence.append(0)

	for output_set in [train_output, valid_output, test_output]:
		for i in xrange(len(output_set)):
			sentence = output_set[i]
			if len(sentence) < maxslength:
				for i in xrange(maxslength-len(sentence)):
					sentence.append(0)

	nsentence = len(train_input)
	svocsize = len(sourcedict)
	tvocsize = len(targetdict)
	print sourcedict
	print targetdict
	print train_input
	print train_output

	train_input = theano.shared(numpy.asarray(train_input))
	train_output = theano.shared(numpy.asarray(train_output))

	print train_input
	print train_output
	#construct model
	
	rng = numpy.random.RandomState(23455)
	index = T.lscalar()

	x = T.lvector('x')
	y = T.lvector('y')
	rnn = rnnmodel(rng, x, svocsize, tvocsize, 100, maxslength)

	cost = rnn.negative_log_likelihood(y)
	params = [rnn.inw, rnn.recurrent, rnn.outw]
	grads = T.grad(cost, params)
	updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

	train_model = theano.function(
		[index],
		cost,
		updates=updates,
		givens={x: train_input[index],
				y: train_output[index]}
	)

	test_model = theano.function(
		[train_input[0]],
		rnn.pred
		)

	for epoch in range(0, 2):
		print epoch
		for i in range(0, nsentence):
			cost = train_model(i)

	print test_model()