"""Functions to load data from folders and augment it"""

import itertools
import logging
import random
import numpy as np

import torch
from torch.utils.data.sampler import Sampler

from torchtext import data
from torchtext import datasets

LOG = logging.getLogger('main')
NO_LABEL = -1


def make_imdb_dataset(number_of_labeled_to_keep, vectors, random_seed=1978, use_gpu=True):
    """
    Uses pytorch.datasets to build IMDB dataset
    :param number_of_labeled_to_keep: number of labeled datapoints to KEEP
                if -1, keep all original labels
    :param vectors: <Vector> clas to be used
    :param seed: random seed to be used for removing labels
    :return: train <Dataset>, test <Dataset>,
    """

    TEXT = data.Field(
        lower=True,
        include_lengths=True,
        batch_first=True,
        tensor_type=torch.cuda.LongTensor if use_gpu else torch.LongTensor
    )
    LABEL = data.Field(
        sequential=False,
        batch_first=True,
        use_vocab=False,
        tensor_type=torch.cuda.LongTensor if use_gpu else torch.LongTensor
    )#, postprocssing=str_to_label)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # print information about the data
    # print('train.fields', train.fields)
    # print('len(train)', len(train))
    # print('vars(train[0])', vars(train[0]))

    # build the vocabulary
    if vectors:
        TEXT.build_vocab(train, vectors=vectors)
    else:
        TEXT.build_vocab(train)

    # print vocab information
    # print('len(TEXT.vocab)', len(TEXT.vocab))
    # print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    def str_to_label(str_):
        if str_ == "pos":
            return 1
        elif str_ == "neg":
            return 0

    if number_of_labeled_to_keep != -1:

        labeled = []
        unlabeled = []

        # randomly remove some labels
        random.seed(random_seed)
        random_labels_to_remove = \
            set(
                random.sample(
                    range(len(train.examples)),
                    len(train.examples) - number_of_labeled_to_keep
                )
            )

        for idx in range(len(train.examples)):
            if idx in random_labels_to_remove:
                train.examples[idx].label = NO_LABEL
                unlabeled.append(idx)
            else:
                train.examples[idx].label = str_to_label(train.examples[idx].label)
                labeled.append(idx)

        print("removed {} labels from the {} total examples".format(
            len(unlabeled),
            len(unlabeled) + len(labeled)
            )
        )
    else:
        for idx in range(len(train.examples)):
            train.examples[idx].label = str_to_label(train.examples[idx].label)
            print("kept all original labels")

    # converting test labels to <int>
    for idx in range(len(test.examples)):
        test.examples[idx].label = str_to_label(test.examples[idx].label)

    return train, test


class CustomIterator(data.BucketIterator):

    def __init__(self, dataset, batch_size, num_labeled_in_batch, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=None, shuffle=None, sort=None,
                 sort_within_batch=None):

        self.num_labeled_in_batch = num_labeled_in_batch
        data.BucketIterator.__init__(self,
                                     dataset, batch_size, sort_key, device, batch_size_fn,
                                     train, repeat, shuffle, sort, sort_within_batch
                                     )

    def create_batches(self):
        if self.sort:
            self.batches = self._batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = self._pool(self.data(), self.batch_size,
                                self.sort_key, self.num_labeled_in_batch, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch)

    @staticmethod
    def _batch(data, batch_size, batch_size_fn=None):
        """Yield elements from data in chunks of batch_size."""
        if batch_size_fn is None:
            def batch_size_fn(new, count, sofar):
                return count
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
        if minibatch:
            yield minibatch

    @staticmethod
    def _batch_custom(data, batch_size, sort_key, num_labeled_in_batch, batch_size_fn=None):
        """Yield elements from data in chunks of batch_size."""
        print("num_labeled_in_batch", num_labeled_in_batch)
        if num_labeled_in_batch > batch_size:
            raise Exception("num_labeled_in_batch must be < batch_size, {} !< {}".format(
                num_labeled_in_batch, batch_size
            )
        )
        if batch_size_fn is None:
            def batch_size_fn(new, count, sofar):
                return count
        minibatch, size_so_far = [], 0
        data = list(data)
        labeled_data = itertools.cycle(sorted(list(filter(lambda x: x.label in [0,1], data)), key=sort_key))
        unlabeled_data = sorted(list(filter(lambda x: x.label not in [0,1], data)), key=sort_key)
        num_unlabeled = batch_size - num_labeled_in_batch
        i = 0
        unlabeled_i = 0
        while i <= len(data) - 1:
            # add unlabeled first
            if unlabeled_i <= len(unlabeled_data) - 1:
                for j in range(num_unlabeled):
                    minibatch.append(unlabeled_data[unlabeled_i])
                    unlabeled_i += 1
                    i += 1
            # add labeled
            for k in range(num_labeled_in_batch):
                next_ = next(labeled_data)
                minibatch.append(next_)
                #             labeled_i += 1
                i += 1
            yield minibatch
            minibatch = []
        if minibatch:
            yield minibatch

    def _pool(self, data, batch_size, key, num_labeled_in_batch,
              batch_size_fn=lambda new, count, sofar: count,
              random_shuffler=None, shuffle=False, sort_within_batch=False):
        """Sort within buckets, then batch, then shuffle batches.
        Partitions data into chunks of size 100*batch_size, sorts examples within
        each chunk using sort_key, then batch these examples and shuffle the
        batches.
        """
        if random_shuffler is None:
            random_shuffler = random.shuffle
        for p in self._batch(data, batch_size * 100, batch_size_fn):
            p_batch = \
                self._batch_custom(
                    sorted(p, key=key),
                    batch_size,
                    key,
                    num_labeled_in_batch,
                    batch_size_fn
                ) if sort_within_batch \
                else self._batch(p, batch_size, batch_size_fn)
            if shuffle:
                for b in random_shuffler(list(p_batch)):
                    yield b
            else:
                for b in list(p_batch):
                    yield b


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
