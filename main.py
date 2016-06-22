# -*- coding: utf-8 -*-
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from aRNNie import ARNNie

hidden_size = 100 # Size of hidden layer of neurons (H)
seq_length = 25 # Number of steps to unroll the RNN
learning_rate = 0.1

with open('data/input.txt') as f:    
    data = f.read().replace('\n', ' ').encode('ascii', 'ignore')

args = {
    'hidden_size': hidden_size,
    'seq_length': seq_length,
    'learning_rate': learning_rate,
    'data': data
}

# Initialized the RNN and run the first epoch
rnn = ARNNie(args)
inputs, hidden, loss = rnn.step()

i = 0

while True:
    inputs, hidden, loss = rnn.step(hidden)  

    if i % 100 == 0:
        print "Iteration {}:".format(i)
        print "Loss: {}".format(loss)
        print rnn.generate(hidden, inputs[0], 140)
        print ""

    i += 1