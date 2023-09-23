from typing import Dict, List, Callable, Optional
from abc import abstractmethod
import statistics
import random
import torch


class LinearClassifier:
    """An abstract class for the linear classifiers"""

    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.w = None

    def train(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            learning_rate: float = 1e-3,
            reg: float = 1e-5,
            num_iters: int = 100,
            batch_size: int = 200,
            verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.w,
            x_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.w, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, x: torch.Tensor):
        return predict_linear_classifier(self.w, x)

    @abstractmethod
    def loss(
            self,
            w: torch.Tensor,
            x_batch: torch.Tensor,
            y_batch: torch.Tensor,
            reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, x_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.w, x_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.w}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        w_dict = torch.load(path, map_location="cpu")
        self.w = w_dict["W"]
        if self.w is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
            self,
            w: torch.Tensor,
            x_batch: torch.Tensor,
            y_batch: torch.Tensor,
            reg: float,
    ):
        return svm_loss_vectorized(w, x_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
            self,
            w: torch.Tensor,
            x_batch: torch.Tensor,
            y_batch: torch.Tensor,
            reg: float,
    ):
        return softmax_loss_vectorized(w, x_batch, y_batch, reg)


def svm_loss_naive(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, reg: float):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on mini-batches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """

    loss = 0.0
    d_w = torch.zeros_like(w)

    num_classes = w.shape[1]
    num_train = x.shape[0]

    for i in range(num_train):
        scores = w.t().mv(x[i])
        correct_class_score = scores[y[i]]

        for j in range(num_classes):

            if j == y[i]:
                continue

            # compares all incorrect labels' scores to the correct label score
            margin = scores[j] - correct_class_score + 1

            if margin > 0:
                loss += margin
                # gradient update for incorrect class.
                d_w[:, j] += x[i]
                # gradient update for correct class.
                d_w[:, y[i]] -= x[i]

    loss /= num_train
    d_w /= num_train

    # l2 regularization
    loss += reg * torch.sum(w * w)
    d_w += reg * w

    return loss, d_w


def svm_loss_vectorized(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, reg: float):
    """
    Structured SVM loss function, vectorized implementation.
    The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - w: A PyTorch tensor of shape (D, C) containing weights.
    - x: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N, ) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """

    num_train = x.shape[0]

    predicted_labels = x.mm(w)

    true_labels = predicted_labels[torch.arange(num_train), y]

    # subtract from each sample prediction value, the correct sample prediction value - 1
    margins = torch.max(torch.zeros_like(predicted_labels), predicted_labels - true_labels.view(-1, 1) + 1)

    # init each samples' correct prediction value
    margins[torch.arange(num_train), y] = 0

    # calculate the SVM loss as the mean of the positive margins plus L2 regularization.
    loss = torch.sum(margins) / num_train + reg * torch.sum(w * w)
    
    adj_margins = margins
    adj_margins[adj_margins > 0] = 1
    adj_margins[torch.arange(num_train), y] = - adj_margins.sum(dim=1)

    d_w = x.t().mm(adj_margins)
    d_w = d_w / num_train + reg * w

    return loss, d_w


def sample_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.

    return x_batch, y_batch
    """

    num_samples = x.shape[0]

    # Randomly select indices for the mini-batch.
    batch_indices = torch.randperm(num_samples)[:batch_size]

    # Extract the mini-batch data and labels based on the selected indices.
    x_batch = x[batch_indices]
    y_batch = y[batch_indices]

    return x_batch, y_batch


def train_linear_classifier(loss_func: Callable,
                            w: torch.Tensor,
                            x: torch.Tensor,
                            y: torch.Tensor,
                            learning_rate: float = 1e-3,
                            reg: float = 1e-5,
                            num_iters: int = 100,
                            batch_size: int = 200,
                            verbose: bool = False):

    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N, ) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """

    # initialize w
    if w is None:
        data_size = x.shape[1]
        class_num = y.max() + 1
        normalizer = 0.000001

        w = normalizer * torch.randn(data_size, class_num, device=x.device, dtype=x.dtype)

    loss_history = []

    # stochastic gradient descent
    for epoch in range(num_iters):

        x_batch, y_batch = sample_batch(x=x, y=y, batch_size=batch_size)

        loss, d_w = loss_func(w, x_batch, y_batch, reg)
        w -= learning_rate * d_w

        if epoch % 100 == 0:
            loss_history.append(loss)

            if verbose:
                print("iteration %d / %d: loss %f" % (epoch, num_iters, loss))

    return w, loss_history


def predict_linear_classifier(w: torch.Tensor, x: torch.Tensor):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - w: A PyTorch tensor of shape (D, C), containing weights of a model
    - x: A PyTorch tensor of shape (N, D) containing training data; there are N training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N, ) giving predicted labels for each
      element of x. Each element of y_pred should be between 0 and C - 1.
    """

    y_pred = torch.argmax(x.mm(w), dim=1)

    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates.
    - regularization_strengths: regularization strengths candidates, e.g.
    """

    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    regularization_strengths = [1, 10, 100, 1000, 10000]

    return learning_rates, regularization_strengths


def test_one_param_set(cls: LinearClassifier,
                       data_dict: Dict[str, torch.Tensor],
                       lr: float,
                       reg: float,
                       num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
    """
    train_acc = 0.0  # The accuracy is simply the fraction of data points
    val_acc = 0.0  # that are correctly classified.
    ###########################################################################
    # TODO:                                                                   #
    # Write code that, train a linear SVM on the training set, compute its    #
    # accuracy on the training and validation sets                            #
    #                                                                         #
    # Hint: Once you are confident that your validation code works, you       #
    # should rerun the validation code with the final value for num_iters.    #
    # Before that, please test with small num_iters first                     #
    ###########################################################################
    # Feel free to uncomment this, at the very beginning,
    # and don't forget to remove this line before submitting your final version
    # num_iters = 100

    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################

    return cls, train_acc, val_acc


# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(
        W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax-loss function, naive implementation (with loops).  When you implement
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient).

    Inputs have dimension D, there are C classes, and we operate on mini-batches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(
        W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax-loss function, vectorized version.  When you implement the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Don't forget the            #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO: Add your own hyper parameter lists. This should be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select   #
    # different hyperparameters to achieve good performance with the softmax  #
    # classifier.                                                             #
    ###########################################################################
    # Replace "pass" statement with your code
    pass
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths
