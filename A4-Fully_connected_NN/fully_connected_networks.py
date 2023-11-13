"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from eecs598 import Solver


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for a linear (fully connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """

        out = x.flatten(start_dim=1).mm(w) + b

        cache = (x, w, b)

        return out, cache

    @staticmethod
    def backward(d_out, cache):
        """
        Computes the backward pass for a linear layer.
        Inputs:
        - d_out: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache

        num_samples = x.shape[0]

        dx = torch.mm(d_out, w.t()).reshape(x.shape)

        dw = torch.mm(x.reshape(num_samples, -1).t(), d_out)

        db = torch.sum(d_out, dim=0)

        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = x.clone()
        out[out < 0] = 0

        cache = x
        return out, cache

    @staticmethod
    def backward(d_out, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - d_out: Upstream derivatives, of any shape
        - cache: Input x, of the same shape as d_out
        Returns:
        - dx: Gradient with respect to x
        """
        x = cache

        dx = d_out.clone()
        dx[x <= 0] = 0

        return dx


class LinearRelu(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs a linear transform followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Dict to give to the backward pass
        """

        z, fully_connected_cache = Linear.forward(x, w, b)
        activation, relu_cache = ReLU.forward(z)
        cache = {"fully_connected_cache": fully_connected_cache, 'relu_cache': relu_cache}
        return activation, cache

    @staticmethod
    def backward(d_out, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fully_connected_cache, relu_cache = cache['fully_connected_cache'], cache['relu_cache']
        da = ReLU.backward(d_out, relu_cache)
        dx, dw, db = Linear.backward(da, fully_connected_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully connected neural network with ReLU nonlinear and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecture should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cuda'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype.
          Float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - Device: device to use for computation: 'Cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        self.params[f'W1'] = torch.randn(input_dim, hidden_dim, dtype=dtype, device=device) * weight_scale
        self.params[f'b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params[f'W2'] = torch.randn(hidden_dim, num_classes, dtype=dtype, device=device) * weight_scale
        self.params[f'b2'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, x, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N).
        Y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.

        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """

        w1 = self.params['W1']
        b1 = self.params['b1']
        w2 = self.params['W2']
        b2 = self.params['b2']

        hidden, cache1 = Linear.forward(x=x, w=w1, b=b1)
        scores, cache2 = Linear.forward(x=hidden, w=w2, b=b2)

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, d_loss = softmax_loss(x=scores, y=y)

        # L2 regularization
        loss += self.reg * (torch.sum(w1 * w1) + torch.sum(w2 * w2))

        dx, grads['W2'], grads['b2'] = Linear.backward(d_out=d_loss, cache=cache2)
        dx, grads['W1'], grads['b1'] = Linear.backward(d_out=dx, cache=cache1)

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully connected neural network with an arbitrary number of hidden layers,
    ReLU non-linearity, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {Linear - relu - [dropout]} x (L - 1) - linear - softmax

    Where dropout is optional, and the {...} Block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability
          for networks with dropout. If dropout=0 then the network
          should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deterministic, so we
          can gradient check the model.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. Float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'Cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.num_classes = num_classes

        hidden_dims.insert(0, input_dim)
        hidden_dims.append(num_classes)

        self.params = {}
        for layer in range(len(hidden_dims) - 1):
            wight = (torch.randn(hidden_dims[layer], hidden_dims[layer + 1], dtype=dtype, device=device) *
                     weight_scale)
            basis = torch.zeros(hidden_dims[layer + 1], dtype=dtype, device=device)

            self.params[f'W{layer + 1}'] = wight
            self.params[f'b{layer + 1}'] = basis

        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'use_dropout': self.use_dropout,
            'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, x, y=None):
        """
        Compute loss and gradient for the fully connected net.
        Input / output: Same as TwoLayerNet above.
        """
        x = x.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batch-norm params and dropout param
        # since they behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode

        # forward pass
        out = x.clone()
        caches = {}

        for layer in range(1, self.num_layers + 1):
            out, cache = LinearRelu.forward(x=out, w=self.params[f'W{layer}'], b=self.params[f'b{layer}'])
            caches[f'cache{layer}'] = cache

        if mode == 'test':
            return out

        # compute loss
        loss, d_loss = softmax_loss(x=out, y=y)

        # backward pass
        reg_wights_sum = 0
        dx = d_loss
        grads = {}

        for layer in range(self.num_layers, 0, -1):
            reg_wights_sum += torch.sum(self.params[f'W{layer}'] * self.params[f'W{layer}'])

            dx, grads[f'W{layer}'], grads[f'b{layer}'] = LinearRelu.backward(d_out=dx,
                                                                             cache=caches[f'cache{layer}'])
            # regularization
            grads[f'W{layer}'] += self.reg * self.params[f'W{layer}']

        loss += 0.5 * self.reg * reg_wights_sum

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    """
    Use a Solver instance to train a TwoLayerNet
    """
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)

    solver = Solver(model=model,
                    data=data_dict,
                    optim_config={'learning_rate': 0.01},
                    lr_decay=0.94,
                    num_epochs=15,
                    batch_size=32,
                    print_every=1000,
                    device=device)
    return solver


def get_three_layer_network_params():
    weight_scale = 0.04
    learning_rate = 0.91
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    Config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between zero and one giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v

    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    Config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between zero and one, giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw * dw
    next_w = w - config['learning_rate'] * dw / (torch.sqrt(config['cache']) + config['epsilon'])

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    # Increment the iteration number
    config['t'] += 1

    # Update biased first moment estimate
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw

    # Update biased second raw moment estimate
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dw ** 2)

    # Bias-corrected first moment estimate
    m_hat = config['m'] / (1 - config['beta1'] ** config['t'])

    # Bias-corrected second raw moment estimate
    v_hat = config['v'] / (1 - config['beta2'] ** config['t'])

    # Adam update formula
    next_w = w - config['learning_rate'] * m_hat / (torch.sqrt(v_hat) + config['epsilon'])

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            ##############################################################
            # TODO: Implement training phase forward pass for            #
            # inverted dropout.                                          #
            # Store the dropout mask in the mask variable.               #
            ##############################################################
            # Replace "pass" statement with your code
            pass
            ##############################################################
            #                   END OF YOUR CODE                         #
            ##############################################################
        elif mode == 'test':
            ##############################################################
            # TODO: Implement the test phase forward pass for            #
            # inverted dropout.                                          #
            ##############################################################
            # Replace "pass" statement with your code
            pass
            ##############################################################
            #                      END OF YOUR CODE                      #
            ##############################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            ###########################################################
            # TODO: Implement training phase backward pass for        #
            # inverted dropout                                        #
            ###########################################################
            # Replace "pass" statement with your code
            pass
            ###########################################################
            #                     END OF YOUR CODE                    #
            ###########################################################
        elif mode == 'test':
            dx = dout
        return dx
