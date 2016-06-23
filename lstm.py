# -*- coding: utf-8 -*-
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import numpy as np
import json

class LSTM:
    def __init__(self, args):
        self.hidden_size = args['hidden_size']
        self.seq_length = args['seq_length']
        self.data = args['data']

        self.chars = list(set(self.data))
        self.batch_size = len(self.chars)