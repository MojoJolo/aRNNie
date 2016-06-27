# -*- coding: utf-8 -*-
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from rnn import RNN
import nltk

hidden_size = 100 # Size of hidden layer of neurons (H)
seq_length = 5 # Number of steps to unroll the RNN
learning_rate = 2e-3

with open('data/input.txt') as f:    
    data = f.read().replace('\n', ' ').encode('ascii', 'ignore')

data = data.lower()
data = nltk.word_tokenize(data)

args = {
    'hidden_size': hidden_size,
    'seq_length': seq_length,
    'learning_rate': learning_rate,
    'data': data
}

# Initialized the RNN and run the first epoch
rnn = RNN(args)
inputs, hidden, loss = rnn.step()

i = 0

while True:
    inputs, hidden, loss = rnn.step(hidden)  

    if i % 100 == 0:
        print "Iteration {}:".format(i)
        print "Loss: {}".format(loss)
        print ' '.join(rnn.generate(hidden, inputs[0], 15))
        print ""

    # if i % 10000 == 0:
    #     rnn.save_model()
    #     print "Checkpoint saved!"

    i += 1