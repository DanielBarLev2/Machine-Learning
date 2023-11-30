"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import math
import torch
from eecs598 import Solver
from a3_helper import softmax_loss
from fully_connected_networks import Linear, ReLU, LinearRelu, adam


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
          - 'Pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e., equally on both sides) along the height and
        width axes of the input.
        Be careful not to modify the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        number, channel, x_height, x_width = x.shape
        filter_size, _, w_height, w_width = w.shape
        stride = conv_param['stride']
        pad = conv_param['pad']

        # pad the input
        x_pad = torch.nn.functional.pad(input=x, pad=(pad, pad, pad, pad), value=0)

        # output tensor dimensions and initialization
        out_height = math.floor(1 + (x_height + 2 * pad - w_height) / stride)
        out_width = math.floor(1 + (x_width + 2 * pad - w_width) / stride)

        out = torch.zeros(size=[number, filter_size, out_height, out_width], dtype=x.dtype, device=x.device)

        for n in range(number):

            for i in range(out_height):

                for j in range(out_width):
                    x_slice = x_pad[n, :, i * stride:i * stride + w_height, j * stride:j * stride + w_width]
                    z = (x_slice * w).sum(dim=[1, 2, 3]) + b
                    out[n, :, i, j] = z

        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(d_out, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - d_out: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """

        x, w, b, conv_param = cache
        pad = conv_param['pad']

        numbers, channels, x_height, x_width = x.shape
        filters, _, w_height, w_width = w.shape
        _, _, d_out_height, d_out_width = d_out.shape

        x_padded = torch.nn.functional.pad(input=x, pad=(pad, pad, pad, pad))

        dx = torch.zeros_like(x_padded)
        dw = torch.zeros_like(w)
        db = d_out.sum(dim=[0, 2, 3])

        for n in range(numbers):

            for f in range(filters):
                h_stride = 0

                for i in range(d_out_height):
                    w_stride = 0

                    for j in range(d_out_width):
                        dx[n, :, h_stride:w_height + h_stride, w_stride:w_width + w_stride] += w[f] * d_out[n, f, i, j]
                        dw[f] += x_padded[n, :, h_stride:w_height + h_stride, w_stride:w_width + w_stride] * d_out[
                            n, f, i, j]
                        w_stride += 1
                    h_stride += 1

        dx = dx[:, :, pad:-pad, pad:-pad]

        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        number, chanel, x_height, x_width = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']

        out_height = int(1 + (x_height - pool_height) / stride)
        out_width = int(1 + (x_width - pool_width) / stride)

        out = torch.zeros(size=[number, chanel, out_height, out_width], device=x.device, dtype=x.dtype)

        for n in range(number):

            for c in range(chanel):

                for i in range(out_height):

                    for j in range(out_width):
                        # preform the max pooling
                        x_slice = x[n, c, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
                        out[n, c, i, j] = torch.max(x_slice)

        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(d_out, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - d_out: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        x, conv_param = cache
        number, chanel, x_height, x_width = x.shape
        _, _, out_height, out_width = d_out.shape

        stride = conv_param['stride']
        pool_height = conv_param['pool_height']
        pool_width = conv_param['pool_width']

        dx = torch.zeros_like(x)

        for n in range(number):

            for c in range(chanel):

                for i in range(out_height):

                    for j in range(out_width):
                        x_slice = x[n, c, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
                        row, col = divmod(x_slice.argmax().item(), x_slice.shape[1])
                        dx[n, c, i * stride + row, j * stride + col] += d_out[n, c, i, j]

        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_dims=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filters: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype.Float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation.'Cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        channels, x_height, _ = input_dims

        # H' = 1 + (H + 2 * pad - HH) / stride
        # W' = 1 + (W + 2 * pad - WW) / stride
        hh = input_dims[1] + 2 * ((filter_dims - 1) // 2) - filter_dims + 1
        ww = input_dims[2] + 2 * ((filter_dims - 1) // 2) - filter_dims + 1

        # H' = 1 + (H - pool_height) / stride
        # W' = 1 + (W - pool_width) / stride
        hhh = 1 + ((hh - 2) / 2)
        www = 1 + ((ww - 2) / 2)

        # convolution layer no.1
        self.params[f'W1'] = (torch.randn(num_filters, channels, filter_dims, filter_dims,
                                          dtype=dtype, device=device) * weight_scale)
        self.params[f'b1'] = torch.zeros(num_filters, dtype=dtype, device=device)
        # fully connected layer no.1
        self.params[f'W2'] = (torch.randn(int(hhh * www * num_filters), hidden_dim, dtype=dtype, device=device) *
                              weight_scale)
        self.params[f'b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        # fully connected layer no.2
        self.params[f'W3'] = torch.randn(hidden_dim, num_classes, dtype=dtype, device=device) * weight_scale
        self.params[f'b3'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, x, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        x = x.to(self.dtype)
        w1, b1 = self.params['W1'], self.params['b1']
        w2, b2 = self.params['W2'], self.params['b2']
        w3, b3 = self.params['W3'], self.params['b3']

        filter_size = w1.shape[2]

        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # forward pass
        conv, pool_1_cache = ConvReluPool.forward(x=x, w=w1, b=b1, conv_param=conv_param, pool_param=pool_param)
        hidden, linear_1_cache = LinearRelu.forward(x=conv, w=w2, b=b2)
        scores, linear_2_cache = Linear.forward(x=hidden, w=w3, b=b3)

        if y is None:
            return scores

        # computing loss
        loss, d_loss = softmax_loss(x=scores, y=y)

        # L2 regularization
        loss += self.reg * (torch.sum(w1 * w1) + torch.sum(w2 * w2) + torch.sum((w3 * w3)))

        # backpropagation
        grads = {}

        layer_3_dx, grads['W3'], grads['b3'] = Linear.backward(d_out=d_loss, cache=linear_2_cache)
        layer_2_dx, grads['W2'], grads['b2'] = LinearRelu.backward(d_out=layer_3_dx, cache=linear_1_cache)
        dx, grads['W1'], grads['b1'] = ConvReluPool.backward(d_out=layer_2_dx, cache=pool_1_cache)

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {Conv - [batch-norm?] - relu - [pool?]} x (L - 1) - linear

    Each "{...}" structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU non-linearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully connected layer
    is used to predict the class scores.

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=(8, 8, 8, 8, 8),
                 max_pools=(0, 1, 2, 3, 4),
                 batch_norm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batch_norm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
        - reg: Scalar giving L2 regularization strength, L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batch_norm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. Float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'Cpu' or 'cuda'
        """
        self.num_layers = len(num_filters) + 1
        self.max_pools = max_pools
        self.batch_norm = batch_norm
        self.reg = reg
        self.dtype = dtype
        self.params = {}

        if device == 'cuda':
            device = 'cuda:0'

        index = 0
        stride, pad, filter_size, pool_size, pool_stride = 1, 1, 3, 2, 2
        filter = num_filters[0]
        channel, input_height, input_width = input_dims

        # initializing weights
        for layer, filter in enumerate(num_filters):
            index = layer + 1

            # kaiming or std initialization
            if weight_initializer == 'kaiming':
                self.params[f'W{index}'] = kaiming_initializer(filter, channel, K=filter_size,
                                                               device=device, dtype=dtype)
                self.params[f'b{index}'] = torch.zeros(filter, dtype=dtype, device=device)

            else:
                self.params[f'W{index}'] = torch.randn(filter, channel, filter_size, filter_size,
                                                       dtype=dtype, device=device) * weight_scale
                self.params[f'b{index}'] = torch.zeros(filter, dtype=dtype, device=device)

            channel = filter

            # in the case of max-pooling, update the inputs' dimensions
            if layer in self.max_pools:
                input_height = int(1 + ((1 + (input_height + 2 * pad - filter_size) / stride - pool_size) / 2))
                input_width = int(1 + ((1 + (input_width + 2 * pad - filter_size) / stride - pool_size) / 2))

        index += 1

        # initialize the last fully connected weight with kaiming or std initialization
        if weight_initializer == 'kaiming':
            self.params[f'W{index}'] = kaiming_initializer(int(filter * input_height * input_width), num_classes,
                                                           dtype=dtype, device=device)
            self.params[f'b{index}'] = torch.zeros(num_classes, dtype=dtype, device=device)
        else:
            self.params[f'W{index}'] = torch.randn(int(filter * input_height * input_width), num_classes,
                                                   dtype=dtype, device=device) * weight_scale
            self.params[f'b{index}'] = torch.zeros(num_classes, dtype=dtype, device=device)

        # bach normalization
        self.bn_params = []
        if self.batch_norm:
            self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]

        # check that we got the right number of parameters
        if not self.batch_norm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'max_pools': self.max_pools,
            'batchnorm': self.batch_norm,
            'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batch_norm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, x, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        z = x.to(self.dtype)
        caches = {}
        filter_size = 3

        # padding and stride chosen to preserve the input spatial size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # set mode
        if y is not None:
            mode = 'test'
        else:
            mode = 'train'

        # set train/test mode for batch norm params since they behave differently during training and testing.
        if self.batch_norm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        # forward pass
        for layer in range(self.num_layers - 1):

            wight = self.params[f'W{layer + 1}']
            bias = self.params[f'b{layer + 1}']

            if layer in self.max_pools:
                z, cache = ConvReluPool.forward(x=z, w=wight, b=bias, conv_param=conv_param, pool_param=pool_param)
            else:
                z, cache = ConvRelu.forward(x=z, w=wight, b=bias, conv_param=conv_param)

            caches[f'cache{layer + 1}'] = cache

        # forward pass for last fully connected layer
        wight, bias = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        z, cache = Linear.forward(x=z, w=wight, b=bias)
        caches[f'cache{self.num_layers}'] = cache

        scores = z

        if y is None:
            return scores

        grads = {}

        # computing loss
        loss, dx = softmax_loss(x=scores, y=y)

        # backward pass for last fully connected layer
        dx, dw, db = Linear.backward(d_out=dx, cache=caches[f'cache{self.num_layers}'])
        grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = dw, db

        # backward pass
        for layer in range(self.num_layers - 1, 0, -1):

            if layer - 1 in self.max_pools:
                dx, dw, db = ConvReluPool.backward(d_out=dx, cache=caches[f'cache{layer}'])
                grads[f'W{layer}'], grads[f'b{layer}'] = dw + 2 * dw * self.reg, db
            else:
                dx, dw, db = ConvRelu.backward(d_out=dx, cache=caches[f'cache{layer}'])
                grads[f'W{layer}'], grads[f'b{layer}'] = dw + 2 * dw * self.reg, db

        # l2 regularization
        weight_sum = 0

        for layer in range(self.num_layers):
            weight_sum = torch.sum(self.params[f'W{layer + 1}'] ** 2)

        loss += weight_sum * self.reg

        return loss, grads


def find_over_fit_parameters():
    weight_scale = 2e-1
    learning_rate = 1e-3
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    weight_scale = 2e-1
    learning_rate = 3e-3

    num_train = 500
    small_data = {'X_train': data_dict['X_train'][:num_train],
                  'y_train': data_dict['y_train'][:num_train],
                  'X_val': data_dict['X_val'],
                  'y_val': data_dict['y_val'],
                  }
    
    input_dims = small_data['X_train'].shape[1:]

    model = DeepConvNet(input_dims=input_dims,
                        num_classes=10,
                        num_filters=[32, 64],
                        max_pools=[0, 1],
                        reg=1e-5,
                        weight_scale=weight_scale,
                        weight_initializer='kaiming',
                        device=device,
                        dtype=dtype)

    solver = Solver(model=model,
                    data=data_dict,
                    print_every=50,
                    num_epochs=2000,
                    batch_size=128,
                    update_rule=adam,
                    optim_config={'learning_rate': learning_rate},
                    device='cuda')

    return solver


def kaiming_initializer(din, d_out, K=None, relu=True, device='cpu', dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - din, d_out: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      din input dimensions and d_out output dimensions. Otherwise, if K is
      a non-negative integer, then initialize the weights for a convolution
      layer with Din input channels, d_out output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU non-linearity (Kaiming initialization); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - Device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, d_out); for a
      convolution layer it should have shape (d_out, Din, K, K).
    """
    gain = 2. if relu else 1.

    if K is None:
        weight = torch.randn(din, d_out, device=device, dtype=dtype) * (gain / din) ** (1 / 2)
    else:
        weight = torch.randn(din, d_out, K, K, device=device, dtype=dtype) * ((gain / (din * K * K)) ** (1 / 2))

    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training, the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training, we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        Running_mean = momentum * running_mean + (one - momentum) * sample_mean
        running_var = momentum * running_var + (one - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation, we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D),
        - beta: Shift parameter of shape (D)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D) giving running mean
            of features
          - running_var Array of shape (D) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            pass
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            pass
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - d_gamma: Gradient with respect to scale parameter gamma,
          of shape (D)
        - d_beta: Gradient with respect to shift parameter beta,
          of shape (D)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code
        pass
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation, you should work out the derivatives
        for the batch normalization backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batch norm_backward, but might not use all
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        # Replace "pass" statement with your code
        pass
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C)
        - beta: Shift parameter, of shape (C)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. Momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C) giving running mean of features
          - running_var Array of shape (C) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code
        pass
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(d_out, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - d_out: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - d_gamma: Gradient with respect to scale parameter, of shape (C)
        - d_beta: Gradient with respect to shift parameter, of shape (C)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code
        pass
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta


##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                torch.zeros_like(layer.weight), \
                torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class ConvRelu(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(d_out, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(d_out, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class ConvReluPool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(d_out, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(d_out, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs a linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2) and (D2) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
