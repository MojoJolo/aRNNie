# -*- coding: utf-8 -*-
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from rnn import RNN

# To load the model
rnn = RNN.load_model("model.json")
inputs, hidden, loss = rnn.step()

print rnn.generate(hidden, inputs[0], 140)