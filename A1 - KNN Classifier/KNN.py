"""
Implements a K-Nearest Neighbor classifier in PyTorch with cifar-10-batch.
"""
from typing import List

import torch


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses a naive set of nested loops over the training and
    test data.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    dists = x_train.new_zeros(num_train, num_test, dtype=x_test.dtype)

    for train_index in range(num_train):
        train_vector = x_train[train_index].view(-1)

        for test_index in range(num_test):
            test_vector = x_test[test_index].view(-1)
            dists[train_index][test_index] = torch.sqrt(((train_vector - test_vector) ** 2).sum())

    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses only a single loop over the training data.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    dists = x_train.new_zeros(num_train, num_test)

    # iterate through the train data
    for index in range(num_train):
        # flatten to a vector size of 3x16x16
        train_vector = x_train[index].flatten()
        # convert the test tensor to dim 2 such: (num_test, 736)
        test_array_vectors = x_test.flatten(start_dim=1)
        # calculate the difference between train vector and the "num_test" test vectors
        # automatically broadcasts train_vector to size 100x768
        difference = (train_vector - test_array_vectors) ** 2
        # calculate the sum, in the second dimension and return a tensor size num_test
        dists[index] = torch.sqrt(torch.sum(difference, dim=1))

    # after 100 iterations, the desired size is num_train * num_test
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation should not use any Python loops. For memory-efficiency,
    it also should not create any large intermediate tensors; in particular you
    should not create any intermediate tensors with O(num_train * num_test)
    elements.

    Args:
        x_train: Tensor of shape (num_train, C, H, W)
        x_test: Tensor of shape (num_test, C, H, W)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is
            the squared Euclidean distance between the i-th training point and
            the j-th test point.
    """

    # convert the test tensor to dim 2 such: (num_test, 736)
    train_norms = torch.sum(x_train.view(x_train.shape[0], -1) ** 2, dim=1, keepdim=True)
    test_norms = torch.sum(x_test.view(x_test.shape[0], -1) ** 2, dim=1, keepdim=True)

    inner_product = torch.matmul(x_train.view(x_train.shape[0], -1), x_test.view(x_test.shape[0], -1).t())

    dists = torch.sqrt(train_norms + test_norms.t() - 2 * inner_product)

    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k):
    """
    Given distances between all pairs of training and test samples, predict a
    label for each test sample by taking a MAJORITY VOTE among its `k` nearest
    neighbors in the training set.

    In the event of a tie, this function SHOULD return the smallest label.

    Args:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is the
            squared Euclidean distance between the i-th training point and the
            j-th test point.
        y_train: Tensor of shape (num_train,) giving labels for all training
            samples. Each label is an integer in the range [0, num_classes - 1]
        k: The number of nearest neighbors to use for classification.

    Returns:
        y_pred: int64 Tensor of shape (num_test,) giving predicted labels for
            the test data, where y_pred[j] is the predicted label for the j-th
            test example. Each label should be an integer in the range
            [0, num_classes - 1].
    """
    num_train, num_test = dists.shape

    y_predict = torch.zeros(num_test, dtype=torch.int64)

    for index in range(num_test):
        label, position = dists[:, index].topk(k=k, largest=False)

        labels, votes = y_train[position].unique(return_counts=True)

        prediction = labels[votes.argmax()]

        y_predict[index] = prediction

    return y_predict


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        Create a new K-Nearest Neighbor classifier with the specified training
        data. In the initializer we simply memorize the provided training data.

        Args:
            x_train: Tensor of shape (num_train, C, H, W) giving training data
            y_train: int64 Tensor of shape (num_train, ) giving training labels
        """

        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test: torch.Tensor, k):
        """
        Make predictions using the classifier.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            k: The number of neighbors to use for predictions.

        Returns:
            y_test_pred: Tensor of shape (num_test,) giving predicted labels
                for the test samples.
        """

        distance = compute_distances_no_loops(x_train=self.x_train, x_test=x_test)

        y_test_predict = predict_labels(dists=distance, y_train=self.y_train, k=k)

        return y_test_predict

    def check_accuracy(
            self,
            x_test: torch.Tensor,
            y_test: torch.Tensor,
            k: int = 1,
            quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test
        data. Returns the accuracy of the classifier on the test data, and
        also prints a message giving the accuracy.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            y_test: int64 Tensor of shape (num_test,) giving test labels.
            k: The number of neighbors to use for prediction.
            quiet: If True, don't print a message.

        Returns:
            accuracy: Accuracy of this classifier on the test data, as a
                percent. Python float in the range [0, 100]
        """
        y_test_predict = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_predict).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_cross_validate(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        num_folds: int = 5,
        k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    """
    Perform cross-validation for `KnnClassifier`.

    Args:
        x_train: Tensor of shape (num_train, C, H, W) giving all training data.
        y_train: int64 Tensor of shape (num_train,) giving labels for training
            data.
        num_folds: Integer giving the number of folds to use.
        k_choices: List of integers giving the values of k to try.

    Returns:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.
    """

    x_train_folds = x_train.chunk(num_folds)
    y_train_folds = y_train.chunk(num_folds)

    k_to_accuracies = {}

    for choice in k_choices:

        accuracy_list = []

        for validation_number in range(num_folds):
            index_list = [x for x in range(5)]
            index_list.remove(validation_number)

            x_train_chunk = torch.cat((x_train_folds[index_list[0]], x_train_folds[index_list[1]], x_train_folds[
                index_list[2]], x_train_folds[index_list[3]]))

            y_train_chunk = torch.cat((y_train_folds[index_list[0]], y_train_folds[index_list[1]], y_train_folds[
                index_list[2]], y_train_folds[index_list[3]]))

            x_train_valid = (x_train_folds[validation_number])
            y_train_valid = (y_train_folds[validation_number])

            knn = KnnClassifier(x_train=x_train_chunk, y_train=y_train_chunk)

            accuracy = knn.check_accuracy(x_test=x_train_valid, y_test=y_train_valid, k=choice, quiet=True)

            accuracy_list.append(accuracy)

        k_to_accuracies[choice] = accuracy_list

    return k_to_accuracies
