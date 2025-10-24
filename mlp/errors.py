# -*- coding: utf-8 -*-
"""Error functions.

This module defines error functions, with the aim of model training being to
minimise the error function given a set of inputs and target outputs.

The error functions will typically measure some concept of distance between the
model outputs and target outputs, averaged over all data points in the data set
or batch.
"""

import numpy as np
from mlp.penalties import L1Penalty


class SumOfSquaredDiffsError(object):
    """Sum of squared differences (squared Euclidean distance) error."""

    def __call__(self, outputs, targets):
        """Calculates error function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar cost function value.
        """
        return 0.5 * np.mean(np.sum((outputs - targets)**2, axis=1))

    def grad(self, outputs, targets):
        """Calculates gradient of error function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of error function with respect to outputs.
        """
        return (outputs - targets) / outputs.shape[0]

    def __repr__(self):
        return 'MeanSquaredErrorCost'


class BinaryCrossEntropyError(object):
    """Binary cross entropy error."""

    def __call__(self, outputs, targets):
        """Calculates error function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar error function value.
        """
        return -np.mean(
            targets * np.log(outputs) + (1. - targets) * np.log(1. - outputs))

    def grad(self, outputs, targets):
        """Calculates gradient of error function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of error function with respect to outputs.
        """
        return ((1. - targets) / (1. - outputs) -
                (targets / outputs)) / outputs.shape[0]

    def __repr__(self):
        return 'BinaryCrossEntropyError'


class BinaryCrossEntropySigmoidError(object):
    """Binary cross entropy error with logistic sigmoid applied to outputs."""

    def __call__(self, outputs, targets):
        """Calculates error function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar error function value.
        """
        probs = 1. / (1. + np.exp(-outputs))
        return -np.mean(
            targets * np.log(probs) + (1. - targets) * np.log(1. - probs))

    def grad(self, outputs, targets):
        """Calculates gradient of error function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of error function with respect to outputs.
        """
        probs = 1. / (1. + np.exp(-outputs))
        return (probs - targets) / outputs.shape[0]

    def __repr__(self):
        return 'BinaryCrossEntropySigmoidError'


class CrossEntropyError(object):
    """Multi-class cross entropy error."""

    def __call__(self, outputs, targets):
        """Calculates error function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar error function value.
        """
        return -np.mean(np.sum(targets * np.log(outputs), axis=1))

    def grad(self, outputs, targets):
        """Calculates gradient of error function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of error function with respect to outputs.
        """
        return -(targets / outputs) / outputs.shape[0]

    def __repr__(self):
        return 'CrossEntropyError'


class CrossEntropySoftmaxError(object):
    """Multi-class cross entropy error with Softmax applied to outputs."""

    def __call__(self, outputs, targets):
        """Calculates error function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar error function value.
        """
        normOutputs = outputs - outputs.max(-1)[:, None]
        logProb = normOutputs - np.log(np.sum(np.exp(normOutputs), axis=-1)[:, None])
        return -np.mean(np.sum(targets * logProb, axis=1))

    def grad(self, outputs, targets):
        """Calculates gradient of error function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of error function with respect to outputs.
        """
        probs = np.exp(outputs - outputs.max(-1)[:, None])
        probs /= probs.sum(-1)[:, None]
        return (probs - targets) / outputs.shape[0]

    def __repr__(self):
        return 'CrossEntropySoftmaxError'

# class CrossEntropySoftmaxL1Error(CrossEntropySoftmaxError):
#     """Multi-class cross entropy error with Softmax applied to outputs and L1 penalty."""
    
#     def __init__(self, l1_coefficient=0.001):
#         """Create a new CrossEntropySoftmaxL1Error object.

#         Args:
#             l1_coefficient: Positive constant to scale L1 penalty term by.
#         """
#         super().__init__()
#         assert l1_coefficient > 0., 'L1 penalty coefficient must be positive.'
#         self.l1_coefficient = l1_coefficient

#     def __call__(self, outputs, targets, model_params):
#         """Calculates error function given a batch of outputs and targets.

#         Args:
#             outputs: Array of model outputs of shape (batch_size, output_dim).
#             targets: Array of target outputs of shape (batch_size, output_dim).
#             model_params: List of model parameters to apply L1 penalty to.

#         Returns:
#             Scalar error function value.
#         """
#         base_error = super().__call__(outputs, targets)
#         l1_penalty = L1Penalty(self.l1_coefficient)
#         return base_error + l1_penalty(model_params)

#     def grad(self, outputs, targets, model_params):
#         """Calculates gradient of error function with respect to outputs.

#         Args:
#             outputs: Array of model outputs of shape (batch_size, output_dim).
#             targets: Array of target outputs of shape (batch_size, output_dim).

#         Returns:
#             Gradient of error function with respect to outputs.
#         """
#         base_grad = super().grad(outputs, targets)
#         l1_penalty = L1Penalty(self.l1_coefficient)
#         return base_grad + l1_penalty.grad(model_params)

#     def __repr__(self):
#         return 'CrossEntropySoftmaxL1Error'