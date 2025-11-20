# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import pickle
import gzip
import numpy as np
import os
from mlp import DEFAULT_SEED


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None, smooth_labels=False):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
            smooth_labels (bool): turn on label smoothing
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.smooth_labels = smooth_labels
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch

class MNISTDataProvider(DataProvider):
    """Data provider for MNIST handwritten digit images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, smooth_labels=False):
        """Create a new MNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
            smooth_labels (bool): enable/disable label smoothing
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 10
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'mnist-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs, targets = loaded['inputs'], loaded['targets']
        inputs = inputs.astype(np.float32)
        # pass the loaded data to the parent class __init__
        super(MNISTDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng, smooth_labels)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(MNISTDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1 of K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets

class EMNISTDataProvider(DataProvider):
    """Data provider for EMNIST handwritten digit images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, smooth_labels=False):
        """Create a new EMNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the EMNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
            smooth_labels (bool): enable/disable label smoothing
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 47
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'emnist-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)
        print(loaded.keys())
        inputs, targets = loaded['inputs'], loaded['targets']
        inputs = inputs.astype(np.float32)
        inputs = np.reshape(inputs, newshape=(-1, 28*28))
        inputs = inputs / 255.0
        # pass the loaded data to the parent class __init__
        super(EMNISTDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng, smooth_labels)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(EMNISTDataProvider, self).next()

        if self.smooth_labels:
            targets_batch_mat = self.label_smoothing(targets_batch)
        else:
            targets_batch_mat = self.to_one_of_k(targets_batch)
        return inputs_batch, targets_batch_mat
        


    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1 of K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets
    
    def label_smoothing(self, int_targets, alpha=0.1):
        """Converts integer coded class target to 1 of K coded targets with label smoothing.

        Args:
        int_targets (ndarray): Array of integer coded class targets (i.e.
        where an integer from 0 to `num_classes` - 1 is used to
        indicate which is the correct class). This should be of shape
        (num_data,).
        alpha (float): Smoothing factor.

        Returns:
        Array of 1 of K coded targets with label smoothing i.e. an array of shape
        (num_data, num_classes)

        """
        
        smoothed_targets = np.full((int_targets.shape[0], self.num_classes), alpha / (self.num_classes - 1))
        smoothed_targets[range(int_targets.shape[0]), int_targets] = 1 - alpha
        return smoothed_targets
  
    

class MetOfficeDataProvider(DataProvider):
    """South Scotland Met Office weather data provider."""

    def __init__(self, window_size, batch_size=10, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new Met Office data provider object.

        Args:
            window_size (int): Size of windows to split weather time series
               data into. The constructed input features will be the first
               `window_size - 1` entries in each window and the target outputs
               the last entry in each window.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'HadSSP_daily_qc.txt')
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        raw = np.loadtxt(data_path, skiprows=3, usecols=range(2, 32))
        assert window_size > 1, 'window_size must be at least 2.'
        self.window_size = window_size
        # filter out all missing datapoints and flatten to a vector
        filtered = raw[raw >= 0].flatten()
        # normalise data to zero mean, unit standard deviation
        mean = np.mean(filtered)
        std = np.std(filtered)
        normalised = (filtered - mean) / std
        # create a view on to array corresponding to a rolling window
        shape = (normalised.shape[-1] - self.window_size + 1, self.window_size)
        strides = normalised.strides + (normalised.strides[-1],)
        windowed = np.lib.stride_tricks.as_strided(
            normalised, shape=shape, strides=strides)
        # inputs are first (window_size - 1) entries in windows
        inputs = windowed[:, :-1]
        # targets are last entry in windows
        targets = windowed[:, -1]
        super(MetOfficeDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)


class MixupCutmixDataProvider(object):
    """Wraps a base provider and applies Mixup and/or CutMix per batch."""

    def __init__(self, base_provider, mixup_alpha=0.2, cutmix_alpha=1.0,
                 mixup_prob=0.5, cutmix_prob=0.5, rng=None):
        """
        Args:
            base_provider: DataProvider instance returning inputs and 1-of-K targets.
            mixup_alpha: Beta distribution alpha used for Mixup.
            cutmix_alpha: Beta distribution alpha used for CutMix.
            mixup_prob: Probability of applying Mixup on a batch.
            cutmix_prob: Probability of applying CutMix on a batch.
            rng: Optional numpy RandomState.
        """
        self.base_provider = base_provider
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.rng = rng if rng is not None else np.random.RandomState(DEFAULT_SEED)
        self.num_batches = base_provider.num_batches
        self.batch_size = base_provider.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def new_epoch(self):
        self.base_provider.new_epoch()

    def reset(self):
        self.base_provider.reset()
        self.num_batches = self.base_provider.num_batches

    def _sample_lambda(self, alpha):
        if alpha <= 0:
            return 1.0
        return self.rng.beta(alpha, alpha)

    def _rand_bbox(self, width, height, lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(width * cut_rat)
        cut_h = int(height * cut_rat)
        cx = self.rng.randint(width)
        cy = self.rng.randint(height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        return x1, y1, x2, y2

    def _apply_mixup(self, inputs, targets):
        lam = self._sample_lambda(self.mixup_alpha)
        perm = self.rng.permutation(inputs.shape[0])
        mixed_inputs = lam * inputs + (1. - lam) * inputs[perm]
        mixed_targets = lam * targets + (1. - lam) * targets[perm]
        return mixed_inputs, mixed_targets

    def _apply_cutmix(self, inputs, targets):
        flat_dim = inputs.shape[1]
        side = int(np.sqrt(flat_dim))
        if side * side != flat_dim:
            # Fallback to Mixup for non-square features
            return self._apply_mixup(inputs, targets)

        lam = self._sample_lambda(self.cutmix_alpha)
        perm = self.rng.permutation(inputs.shape[0])
        x1, y1, x2, y2 = self._rand_bbox(side, side, lam)
        inputs_view = inputs.reshape(-1, side, side)
        mixed = inputs_view.copy()
        mixed[:, y1:y2, x1:x2] = inputs_view[perm, y1:y2, x1:x2]
        lam_adjusted = 1. - ((x2 - x1) * (y2 - y1) / (side * side))
        mixed_targets = lam_adjusted * targets + (1. - lam_adjusted) * targets[perm]
        return mixed.reshape(inputs.shape), mixed_targets

    def next(self):
        inputs_batch, targets_batch = self.base_provider.next()
        prob = self.rng.rand()
        if prob < self.mixup_prob:
            return self._apply_mixup(inputs_batch, targets_batch)
        if prob < self.mixup_prob + self.cutmix_prob:
            return self._apply_cutmix(inputs_batch, targets_batch)
        return inputs_batch, targets_batch


class CCPPDataProvider(DataProvider):

    def __init__(self, which_set='train', input_dims=None, batch_size=10,
                 max_num_batches=-1, shuffle_order=True, rng=None):
        """Create a new Combined Cycle Power Plant data provider object.

        Args:
            which_set: One of 'train' or 'valid'. Determines which portion of
                data this object should provide.
            input_dims: Which of the four input dimension to use. If `None` all
                are used. If an iterable of integers are provided (consisting
                of a subset of {0, 1, 2, 3}) then only the corresponding
                input dimensions are included.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'ccpp_data.npz')
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # check a valid which_set was provided
        assert which_set in ['train', 'valid'], (
            'Expected which_set to be either train or valid '
            'Got {0}'.format(which_set)
        )
        # check input_dims are valid
        if not input_dims is not None:
            input_dims = set(input_dims)
            assert input_dims.issubset({0, 1, 2, 3}), (
                'input_dims should be a subset of {0, 1, 2, 3}'
            )
        loaded = np.load(data_path)
        inputs = loaded[which_set + '_inputs']
        if input_dims is not None:
            inputs = inputs[:, input_dims]
        targets = loaded[which_set + '_targets']
        super(CCPPDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)


class AugmentedMNISTDataProvider(MNISTDataProvider):
    """Data provider for MNIST dataset which randomly transforms images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, transformer=None):
        """Create a new augmented MNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'test'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
            transformer: Function which takes an `inputs` array of shape
                (batch_size, input_dim) corresponding to a batch of input
                images and a `rng` random number generator object (i.e. a
                call signature `transformer(inputs, rng)`) and applies a
                potentiall random set of transformations to some / all of the
                input images as each new batch is returned when iterating over
                the data provider.
        """
        super(AugmentedMNISTDataProvider, self).__init__(
            which_set, batch_size, max_num_batches, shuffle_order, rng)
        self.transformer = transformer

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(
            AugmentedMNISTDataProvider, self).next()
        transformed_inputs_batch = self.transformer(inputs_batch, self.rng)
        return transformed_inputs_batch, targets_batch
