#-*- coding: utf-8 -*-
from rnnmodel import rnnmodel

import numpy
import time
import sys
import subprocess
import os
import random

if __name__ == '__main__':

	train_set = [[['I', 'like', 'music', '</eos>'],['I', 'don\'t', 'like', 'painting', '</eos>']],[['我','喜欢','音乐', '</eos>'],['我','不','喜欢','画画', '</eos>']]]
	valid_set = [[['I', 'don\'t', 'like', 'music', '</eos>']],[['我','不','喜欢','音乐', '</eos>']]]
	test_set = [[['I', 'like', 'painting', '</eos>']],[['我','喜欢','画画', '</eos>']]]

	sourcedict = ['</eos>']
	targetdict = ['</eos>']

	train_input, train_output = train_set
	valid_input, valid_output = valid_set
	test_input, test_output = test_set

	#construct dictionary
	for input_set in [train_input, valid_input, test_input]:
		for sentence in input_set:
			for i in range(0,len(sentence)):
				word = sentence[i]
				if not word in sourcedict:
					sourcedict.append(word)
				sentence[i] = sourcedict.index(word)

	for output_set in [train_output, valid_output, test_output]:
		for sentence in output_set:
			for i in range(0,len(sentence)):
				word = sentence[i]
				if not word in targetdict:
					targetdict.append(word)
				sentence[i] = targetdict.index(word)

	print sourcedict
	print targetdict
	print train_input
	print train_output

	nsentence = len(train_input)
	svocsize = len(sourcedict)
	tvocsize = len(targetdict)

	rng = numpy.random.RandomState(23455)
	rnn = rnnmodel(rng, svocsize, tvocsize, 100)

	for epoch in range(0, 200):
		print epoch
		for i in range(0, nsentence):
			rnn.train(train_input[i], train_output[i], len(train_input))
		print rnn.predict(train_input[0])
