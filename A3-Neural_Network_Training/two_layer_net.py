from typing import Dict, Callable, Optional
from linear_classifier import sample_batch
import random
import torch


# Template class modules that we will use later: Do not edit/modify this class
class TwoLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        std: float = 1e-4,
    ):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - dtype: Optional, data type of each initial weight params
        - device: Optional, whether the weight params is on GPU or CPU
        - std: Optional, initial weight scaler.
        """
        # reset seed before start
        random.seed(0)
        torch.manual_seed(0)

        self.params = {"W1": std * torch.randn(input_size, hidden_size, dtype=dtype, device=device),
                       "b1": torch.zeros(hidden_size, dtype=dtype, device=device),
                       "W2": std * torch.randn(hidden_size, output_size, dtype=dtype, device=device),
                       "b2": torch.zeros(output_size, dtype=dtype, device=device)}

    def loss(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, x, y, reg)

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        # fmt: off
        return nn_train(
            self.params, nn_forward_backward, nn_predict, x, y,
            x_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )
        # fmt: on

    def predict(self, x: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, x)

    def save(self, path: str):
        torch.save(self.params, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "W2", "b2"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


def nn_forward_pass(params: Dict[str, torch.Tensor], x: torch.Tensor):
    """
    The first stage of our neural network implementation: Run the forward pass
    of the network to compute the hidden layer features and classification
    scores. The network architecture should be:

    FC layer -> ReLU (hidden) -> FC layer (scores)

    As a practice, we will NOT allow to use torch.relu and torch.nn ops
    just for this time (you can use it from A3).

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have the following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H, )
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C, )
    - x: Input data of shape (N, D). Each X[i] is a training sample.

    Returns a tuple of:
    - scores: Tensor of shape (N, C) giving the classification scores for X
    - hidden: Tensor of shape (N, H) giving the hidden layer representation
      for each input value (after the ReLU).
    """

    # unpack variables from the params dictionary
    w1, b1 = params["W1"], params["b1"]
    w2, b2 = params["W2"], params["b2"]

    # compute the forward pass
    hidden = x.mm(w1) + b1

    # relu
    hidden[hidden < 0] = 0

    scores = hidden.mm(w2) + b2

    return scores, hidden


def nn_forward_backward(params: Dict[str, torch.Tensor],
                        x: torch.Tensor,
                        y: Optional[torch.Tensor] = None,
                        reg: float = 0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. When you implement loss and gradient, please don't forget to
    scale the losses/gradients by the batch size.

    Inputs: First two parameters (params, X) are same as nn_forward_pass
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have the following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H, )
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C, )
    - x: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    w1, b1 = params["W1"], params["b1"]
    w2, b2 = params["W2"], params["b2"]

    num_samples = x.shape[0]

    scores, h1 = nn_forward_pass(params, x)

    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss

    # shift the scores to prevent numerical instability.
    scores += scores.max(dim=1, keepdim=True)[0]

    # compute loss
    exp_scores = torch.exp(scores)
    probabilities = exp_scores / exp_scores.sum(dim=1, keepdim=True)
    loss = torch.sum(-torch.log(probabilities[torch.arange(num_samples), y]))

    # normalize and regularization
    loss = loss / num_samples + reg * (torch.sum(w1 * w1) + torch.sum(w2 * w2))

    # Backward pass: compute gradients
    grads = {}

    # create a one-hot tensor
    y_hot = torch.zeros_like(h1.mm(w2))
    y_hot[torch.arange(y.shape[0]), y] = 1

    # compute gradients
    dz2 = probabilities - y_hot
    dw2 = (1 / num_samples) * h1.t().mm(dz2)
    db2 = (1 / num_samples) * dz2.sum(dim=0, keepdims=True)
    da1 = dz2.mm(w2.t())
    dz1 = da1 * (h1 > 0)
    dw1 = (1 / num_samples) * x.t().mm(dz1)
    db1 = (1 / num_samples) * dz1.sum(dim=0, keepdims=True)

    grads["W1"], grads["b1"] = dw1, db1
    grads["W2"], grads["b2"] = dw2, db2

    return loss, grads


def nn_train(
    params: Dict[str, torch.Tensor],
    loss_func: Callable,
    pred_func: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - params: a dictionary of PyTorch Tensor that stores the weights of a model.
      It should have the following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H, )
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C, )
    - loss_func: a loss function that computes the loss and the gradients.
      It takes as input:
      - params: Same as input to nn_train
      - x_batch: A minibatch of inputs of shape (B, D)
      - y_batch: Ground-truth labels for X_batch
      - reg: Same as input to nn_train
      And it returns a tuple of:
        - loss: Scalar giving the loss on the minibatch
        - grads: Dictionary mapping parameter names to gradients of the loss with
          respect to the corresponding parameter.
    - pred_func: prediction function that im
        - X: A PyTorch tensor of shape (N, D) giving training data.
        - y: A PyTorch tensor of shape (N, ) giving training labels; y[i] = c means
            that X[i] has label c, where 0 <= c < C.
    - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
    - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Returns: A dictionary giving statistics about the training process
    """
    num_train = x.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        x_batch, y_batch = sample_batch(x, y, batch_size)

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_func(params, x_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        # update params
        params['W1'] -= learning_rate * grads['W1']
        params['b1'] -= learning_rate * grads['b1'].reshape(-1)
        params['W2'] -= learning_rate * grads['W2']
        params['b2'] -= learning_rate * grads['b2'].reshape(-1)

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss.item()))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, x_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, x_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
    }


def nn_predict(params: Dict[str, torch.Tensor], loss_func: Callable, x: torch.Tensor):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points For each data point, we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - params: a dictionary of PyTorch Tensor that stores the weights of a model.
      It should have the following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H, )
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C, )
    - loss_func: a loss function that computes the loss and the gradients
    - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    scores = loss_func(params=params, x=x)
    y_pred = scores.argmax(dim=1)

    return y_pred
