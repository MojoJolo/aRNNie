# -*- coding: utf-8 -*-
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import numpy as np
import json

class RNN:
    def __init__(self, args):
        self.hidden_size = args['hidden_size']
        self.seq_length = args['seq_length']
        self.learning_rate = args['learning_rate']
        self.data = args['data']

        self.chars = list(set(self.data))
        self.data_size = len(self.data)
        self.vocab_size = len(self.chars)

        print 'Data has {} characters, {} unique.'.format(self.data_size, self.vocab_size)

        # NN Parameters
        self.param_w_xh = args.get('param_w_xh', np.random.randn(self.hidden_size, self.vocab_size) * 0.01) # Input to hidden (U)
        self.param_w_hh = args.get('param_w_hh', np.random.randn(self.hidden_size, self.hidden_size) * 0.01) # Hidden to hidden (W)
        self.param_w_hy = args.get('param_w_hy', np.random.randn(self.vocab_size, self.hidden_size) * 0.01) # Hidden to output (V)

        self.bias_hidden = args.get('bias_hidden', np.zeros((self.hidden_size, 1))) # hidden bias
        self.bias_output_y = args.get('bias_output_y', np.zeros((self.vocab_size, 1))) # output bias

        self.mem_w_xh = args.get('mem_w_xh', np.zeros_like(self.param_w_xh))
        self.mem_w_hh = args.get('mem_w_hh', np.zeros_like(self.param_w_hh))
        self.mem_w_hy = args.get('mem_w_hy', np.zeros_like(self.param_w_hy))

        self.mem_bias_hidden = args.get('mem_bias_hidden', np.zeros_like(self.bias_hidden))
        self.mem_bias_output_y = args.get('mem_bias_output_y', np.zeros_like(self.bias_output_y))

        self.smooth_loss = args.get('smooth_loss', -np.log(1.0 / self.vocab_size) * self.seq_length) # cross-entropy loss

        self.char_to_ix = args.get('char_to_ix', {char: i for i, char in enumerate(self.chars)})
        self.ix_to_char = args.get('ix_to_char', {i: char for i, char in enumerate(self.chars)})

        self.pointer = args.get('pointer', 0) # Data pointer

    def step(self, hidden=None):
        if (self.pointer + self.seq_length + 1) >= len(self.data) or hidden == None:
            hidden = np.zeros((self.hidden_size, 1)) # reset RNN memory
            self.pointer = 0 # go from start of data

        inputs = [self.char_to_ix[char] for char in self.data[self.pointer:self.pointer + self.seq_length]]
        targets = [self.char_to_ix[char] for char in self.data[self.pointer + 1:self.pointer + self.seq_length + 1]]

        # input vector, output vector, hidden vector, softmax probablity, loss
        input_xs, output_ys, hidden_s, probs, loss = self.forward(inputs, targets, hidden)
        grad_w_xh, grad_w_hh, grad_w_hy, grad_bias_hidden, grad_bias_output_y = self.backward(inputs, targets, input_xs, hidden_s, probs)

        # CLIP THEM UP!
        # clip to mitigate exploding gradients
        for grad_param in [grad_w_xh, grad_w_hh, grad_w_hy, grad_bias_hidden, grad_bias_output_y]:
            np.clip(grad_param, -5, 5, out=grad_param)

        # Parameter update
        for param, grad_param, memory in zip(
                                            [self.param_w_xh, self.param_w_hh, self.param_w_hy, self.bias_hidden, self.bias_output_y],
                                            [grad_w_xh, grad_w_hh, grad_w_hy, grad_bias_hidden, grad_bias_output_y],
                                            [self.mem_w_xh, self.mem_w_hh, self.mem_w_hy, self.mem_bias_hidden, self.mem_bias_output_y]
                                        ):
            memory += grad_param * grad_param
            param += -self.learning_rate * grad_param / np.sqrt(memory + 1e-8)

        self.pointer += self.seq_length

        hidden = hidden_s[len(inputs) - 1]
        self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

        return inputs, hidden, self.smooth_loss

    def forward(self, inputs, targets, hidden_prev):
        # s = vector
        input_xs = {}
        hidden_s = {}
        output_ys = {}
        probs = {} # probablity

        hidden_s[-1] = np.copy(hidden_prev)
        loss = 0

        for i in xrange(len(inputs)):
            # Creating an equivalent one hot vector for each inputs
            input_xs[i] = np.zeros((self.vocab_size, 1))
            input_xs[i][inputs[i]] = 1

            # Calculating the current hidden state using the previous hiden state through tanh
            hidden_s[i] = self.tanh(self.param_w_xh, input_xs[i], self.param_w_hh, hidden_s[i - 1], self.bias_hidden)
            output_ys[i] = np.dot(self.param_w_hy, hidden_s[i]) + self.bias_output_y

            probs[i] = self.softmax(output_ys[i])
            loss += -np.log(probs[i][targets[i], 0])

        return input_xs, output_ys, hidden_s, probs, loss

    # backprop
    def backward(self, inputs, targets, input_xs, hidden_s, probs):
        grad_w_xh = np.zeros_like(self.param_w_xh)
        grad_w_hh = np.zeros_like(self.param_w_hh)
        grad_w_hy = np.zeros_like(self.param_w_hy)

        grad_bias_hidden = np.zeros_like(self.bias_hidden)
        grad_bias_output_y = np.zeros_like(self.bias_output_y)

        grad_hidden_next = np.zeros_like(hidden_s[0])

        for i in reversed(xrange(len(inputs))):
            grad_output_y = np.copy(probs[i])
            grad_output_y[targets[i]] -= 1

            grad_w_hy += np.dot(grad_output_y, hidden_s[i].T)
            grad_bias_output_y += grad_output_y

            grad_hidden = np.dot(self.param_w_hy.T, grad_output_y) + grad_hidden_next
            grad_hidden_raw = (1 - hidden_s[i] * hidden_s[i]) * grad_hidden # backprop through tanh nonlinearity
            grad_bias_hidden += grad_hidden_raw

            grad_w_xh += np.dot(grad_hidden_raw, input_xs[i].T)
            grad_w_hh += np.dot(grad_hidden_raw, hidden_s[i - 1].T)

            grad_hidden_next = np.dot(self.param_w_hh.T, grad_hidden_raw)

        return grad_w_xh, grad_w_hh, grad_w_hy, grad_bias_hidden, grad_bias_output_y

    def generate(self, hidden, seed_ix, chars_counter):
        input_x = np.zeros((self.vocab_size, 1))
        input_x[seed_ix] = 1
        ixes = []

        for i in xrange(chars_counter):
            hidden = np.tanh(np.dot(self.param_w_xh, input_x) + np.dot(self.param_w_hh, hidden) + self.bias_hidden) # tanh
            output_y = np.dot(self.param_w_hy, hidden) + self.bias_output_y
            prob = self.softmax(output_y)
            ix = np.random.choice(range(self.vocab_size), p=prob.ravel())

            input_x = np.zeros((self.vocab_size, 1))
            input_x[ix] = 1

            ixes.append(ix)

        return [self.ix_to_char[ix] for ix in ixes]

    def softmax(self, output_y):
        return np.exp(output_y) / np.sum(np.exp(output_y)) # softmax function

    def tanh(self, param_w_xh, input_x, param_w_hh, hidden, bias_hidden):
        return np.tanh(np.dot(param_w_xh, input_x) + np.dot(param_w_hh, hidden) + bias_hidden) # tanh

    def save_model(self):
        model = {
            'hidden_size': self.hidden_size,
            'seq_length': self.seq_length,
            'learning_rate': self.learning_rate,
            'data': self.data,

            'chars': self.chars,
            'data_size': self.data_size,
            'vocab_size': self.vocab_size,

            'param_w_xh': self.param_w_xh.tolist(),
            'param_w_hh': self.param_w_hh.tolist(),
            'param_w_hy': self.param_w_hy.tolist(),

            'bias_hidden': self.bias_hidden.tolist(),
            'bias_output_y': self.bias_output_y.tolist(),

            'mem_w_xh': self.mem_w_xh.tolist(),
            'mem_w_hh': self.mem_w_hh.tolist(),
            'mem_w_hy': self.mem_w_hy.tolist(),

            'mem_bias_hidden': self.mem_bias_hidden.tolist(),
            'mem_bias_output_y': self.mem_bias_output_y.tolist(),

            'smooth_loss': self.smooth_loss,

            'char_to_ix': self.char_to_ix,
            'ix_to_char': self.ix_to_char,

            'pointer': self.pointer,
        }

        with open('model.json', 'w') as jsonfile:
            json.dump(model, jsonfile, indent=4, separators=(',', ': '))

    @classmethod
    def load_model(cls, filename):
        with open(filename) as jsonfile:    
            model = json.load(jsonfile)

        model['param_w_xh'] = np.array(model['param_w_xh'])
        model['param_w_hh'] = np.array(model['param_w_hh'])
        model['param_w_hy'] = np.array(model['param_w_hy'])

        model['bias_hidden'] = np.array(model['bias_hidden'])
        model['bias_output_y'] = np.array(model['bias_output_y'])

        model['mem_w_xh'] = np.array(model['mem_w_xh'])
        model['mem_w_hh'] = np.array(model['mem_w_hh'])
        model['mem_w_hy'] = np.array(model['mem_w_hy'])

        model['mem_bias_hidden'] = np.array(model['mem_bias_hidden'])
        model['mem_bias_output_y'] = np.array(model['mem_bias_output_y'])

        model['ix_to_char'] = {int(ix): char for ix, char in model['ix_to_char'].iteritems()}

        return cls(model)
