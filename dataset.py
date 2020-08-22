import numpy


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel().astype(int)] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, features, labels, date):
        self._num_examples = 0
        self._index_in_epoch = self._num_examples
        if len(features) != 0:
            assert features.shape[0] == labels.shape[0], (
                    "images.shape: %s labels.shape: %s" % (features.shape,
                                                           labels.shape))
            self._num_examples = features.shape[0]
            features = features.astype(numpy.float32)
        self._features = features
        self._labels = labels
        self._date = date
        self._last_label = []
        self._closing_price = []
        self._epochs_completed = 0
        self.label_max = 0
        self.label_min = 0

    def label_normalization(self):
        self.label_max = self.labels.max()
        self.label_min = self.labels.min()
        self._labels = (self.labels - self.label_min) / (self.label_max - self.label_min)

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def date(self):
        return self._date

    @property
    def last_label(self):
        return self._last_label

    @property
    def closing_price(self):
        return self._closing_price

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        self._num_examples = self.features.shape[0]
        self._index_in_epoch = self._num_examples
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            numpy.random.shuffle(perm)
            numpy.random.shuffle(perm)
            numpy.random.shuffle(perm)
            numpy.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]
