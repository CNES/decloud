# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2022 INRAE

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
"""Dataset classes"""
import abc
import collections.abc
import json
import logging
import multiprocessing
import threading
import time
import tensorflow as tf
import numpy as np
import random
from decloud.core import system


# -------------------------------------------------- Buffer class ------------------------------------------------------


class Buffer:
    """
    Buffer class
    Used to store and access list of objects
    """

    def __init__(self, max_length):
        self.max_length = max_length
        self.container = []

    def size(self):
        """ Return buffer size """
        return len(self.container)

    def add(self, new_elem):
        """ Add a new element to the buffer """
        self.container.append(new_elem)
        assert self.size() <= self.max_length

    def is_complete(self):
        """ Return True if the buffer is complete"""
        return self.size() == self.max_length


# ---------------------------------------------- RandomIterator class --------------------------------------------------


class BaseIterator(abc.ABC):
    """
    Base class for iterators
    """
    @abc.abstractmethod
    def __init__(self, acquisitions_layout, tile_handlers, tile_rois):

        self.tuples_grids = {tile_name: tile_handler.tuple_search(acquisitions_layout=acquisitions_layout,
                                                                  roi=tile_rois[tile_name] if tile_name in tile_rois
                                                                  else None)
                             for tile_name, tile_handler in tile_handlers.items()}

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        """ Provides a sequence of (tile_name, tuple_pos, tuple_indices) """

    @abc.abstractmethod
    def shuffle(self):
        """ Shuffle the sequence """


class RandomIterator(BaseIterator):
    """
    The most basic iterator. Pick a random tuple within all the available ones, regardless its position in the grid,
    or anything.
    A mapping tuple_id --> (tuple_pos, tuple_indices) is created, then a random tuple is picked among the tuple_ids,
    then the corresponding tuple is accessed.
    """

    def __init__(self, acquisitions_layout, tile_handlers, tile_rois):

        # Count the number of available tuples, and map the tuples: id --> (tile_name, tuple_pos, tuple_indices)
        super().__init__(acquisitions_layout, tile_handlers, tile_rois)
        # Tuple ID list
        self.nb_of_tuples = 0
        self.tuples_map = dict()

        for tile_name, tuples_grid in self.tuples_grids.items():
            for tuple_pos, tuple_indices in tuples_grid.items():
                for tuple_idx in tuple_indices:
                    self.tuples_map[self.nb_of_tuples] = (tile_name, tuple_pos, tuple_idx)
                    self.nb_of_tuples += 1

        self.indices = np.arange(0, self.nb_of_tuples)
        self.shuffle()
        self.count = 0

    def __next__(self):
        current_index = self.indices[self.count]
        ret = self.tuples_map[current_index]
        if self.count < self.nb_of_tuples - 1:
            self.count += 1
        else:
            self.shuffle()
            self.count = 0
        return ret

    def shuffle(self):
        np.random.shuffle(self.indices)


class ConstantIterator(BaseIterator):
    """
    An iterator that aims to deliver the same number of samples at each patch location.
    A mapping tuple_id --> (tuple_pos, tuple_indices) is created, then a random tuple is picked among the tuple_ids,
    then the corresponding tuple is accessed.
    Note that after one epoch, samples are still the same: we don't pick random samples at each new iteration, which
    is not really a problem if we convert them in tfrecords directly.
    """

    def __init__(self, acquisitions_layout, tile_handlers, tile_rois, nbsample_max=10):

        # Count the number of available tuples, and map the tuples: id --> (tile_name, tuple_pos, tuple_indices)
        super().__init__(acquisitions_layout, tile_handlers, tile_rois)
        # Tuple ID list
        self.nb_of_tuples = 0
        self.tuples_map = dict()

        for tile_name, tuples_grid in self.tuples_grids.items():
            for tuple_pos, tuple_indices in tuples_grid.items():
                rand_tuple_indices = random.sample(tuple_indices, nbsample_max) if len(tuple_indices) > nbsample_max \
                    else tuple_indices
                for tuple_idx in rand_tuple_indices:
                    self.tuples_map[self.nb_of_tuples] = (tile_name, tuple_pos, tuple_idx)
                    self.nb_of_tuples += 1

        self.indices = np.arange(0, self.nb_of_tuples)
        self.shuffle()
        self.count = 0

    def __next__(self):
        current_index = self.indices[self.count]
        ret = self.tuples_map[current_index]
        if self.count < self.nb_of_tuples - 1:
            self.count += 1
        else:
            self.shuffle()
            self.count = 0
        return ret

    def shuffle(self):
        np.random.shuffle(self.indices)


def update(tuple_map, tmp):
    """ Update the tuple map """
    for key, value in tmp.items():
        if isinstance(value, collections.abc.Mapping):
            tuple_map[key] = update(tuple_map.get(key, {}), value)
        else:
            tuple_map[key] = value
    return tuple_map


class OversamplingIterator(BaseIterator):
    """
    Iterator that provides the same amount of samples for each season
    Seasons are defined in self.months_list
    """
    def __init__(self, acquisitions_layout, tile_handlers, tile_rois):
        super().__init__(acquisitions_layout, tile_handlers, tile_rois)

        self.months_list = [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]]

        self.distribution = dict()

        for i in range(len(self.months_list)):
            self.distribution[i] = dict({"count": 0, "number": 0})

        self.tuples_map = dict()
        self.nb_of_tuples = 0
        for tile_name, tuples_grid in self.tuples_grids.items():
            for tuple_pos, tuple_indices in tuples_grid.items():
                for tuple_idx in tuple_indices:
                    idx = None
                    for idx_month, month in enumerate(self.months_list):
                        if tile_handlers[tile_name].s2_images[tuple_idx["t"]["s2"]].acq_date.month in month:
                            idx = idx_month

                    assert idx is not None,\
                        "Date from image {} not in date range".format(
                            tile_handlers[tile_name].s2_images[tuple_idx["t"]["s2"]])

                    tmp = {idx: {self.distribution[idx]["number"]: (tile_name, tuple_pos, tuple_idx)}}
                    self.distribution[idx]["number"] += 1
                    self.tuples_map = update(self.tuples_map, tmp)
                    self.nb_of_tuples += 1

        self.keys = list(self.tuples_map.keys())
        self.indices = dict()

        for idx in self.distribution:
            indices = {idx: np.arange(0, self.distribution[idx]["number"])}
            self.indices.update(indices)
            self.shuffle_indices(idx)

        logging.info("Distribution: %s", self.indices)

    def shuffle_indices(self, idx):
        """ Shuffle the indices of the idx-th season """
        np.random.shuffle(self.indices[idx])

    def shuffle(self):
        for idx in self.distribution:
            self.shuffle_indices(idx)

    def __next__(self):
        """
        Provides a sequence of (tile_name, tuple_pos, tuple_indices)
        """

        idx = int(np.random.randint(len(self.keys)))

        pos = self.keys[idx]

        current_index = self.indices[pos][self.distribution[pos]["count"]]

        ret = self.tuples_map[pos][current_index]
        if self.distribution[pos]["count"] < self.distribution[pos]["number"] - 1:
            self.distribution[pos]["count"] += 1
        else:
            self.shuffle_indices(pos)
            self.distribution[pos]["count"] = 0
        return ret


class LimitedIterator(OversamplingIterator):
    """
    Iterator that ends after a fixed number of samples
    """
    def __init__(self, acquisitions_layout, tile_handlers, tile_rois, nb_samples=1000):
        super().__init__(acquisitions_layout, tile_handlers, tile_rois)
        self.nb_of_tuples = nb_samples


# ------------------------------------------------- Dataset class ------------------------------------------------------


class Dataset:
    """
    Handles the "mining" of the tile handlers.
    This class has a thread that extract tuples from the tile handlers, while ensuring the access of already gathered
    tuples.
    """
    def __init__(self, acquisitions_layout, tile_handlers, tile_rois, buffer_length=128, iterator_class=RandomIterator,
                 max_nb_of_samples=None):
        """

        :param acquisitions_layout: The acquisitions layout (instance of sensing_layout.AcquisitionsLayout)
        :param tile_handlers: A dict() of tile_io.TileHandler instances. The keys of the dict() are the tile name.
        :param tile_rois: A dict() of ROIs. The keys of the dict() are the tile name.
        :param buffer_length: The number of samples that are stored in the buffer.
        :param iterator_class: An iterator that provides a sequence of (tile_name, tuple_pos, tuple_indices)
        :param max_nb_of_samples: Optional, max number of samples to consider
         for the samples to be read.
        """

        # tile handlers
        self.tile_handlers = tile_handlers

        # iterator
        self.iterator = iterator_class(acquisitions_layout, tile_handlers, tile_rois)
        self.size = min(self.iterator.nb_of_tuples,
                        max_nb_of_samples) if max_nb_of_samples else self.iterator.nb_of_tuples

        # Get patches sizes and type, of the first sample of the first tile
        self.output_types = dict()
        self.output_shapes = dict()

        # Here we read the first available sample, and we exit breaking all loops
        for tile_name, tile_tuples_grid in self.iterator.tuples_grids.items():
            for tuple_pos, tuple_indices in tile_tuples_grid.items():
                for indices in tuple_indices:
                    new_sample = self.tile_handlers[tile_name].read_tuple(tuple_pos=tuple_pos, tuple_indices=indices)
                    for key, np_arr in new_sample.items():
                        if isinstance(np_arr, (np.ndarray, np.generic)):
                            self.output_shapes[key] = np_arr.shape
                            self.output_types[key] = tf.dtypes.as_dtype(np_arr.dtype)
                    break
                break
            if self.output_shapes:  # we break only if there was a sample, otherwise we continue to next tile
                break
        logging.info("output_types: %s", self.output_types)
        logging.info("output_shapes: %s", self.output_shapes)

        # buffers
        self.miner_buffer = Buffer(buffer_length)
        self.consumer_buffer = Buffer(buffer_length)
        self.consumer_buffer_pos = 0
        self.tot_wait = 0
        self.miner_thread = self._summon_miner_thread()
        self.read_lock = multiprocessing.Lock()
        self._dump()

        # Prepare tf dataset
        self.tf_dataset = tf.data.Dataset.from_generator(self._generator,
                                                         output_signature={
                                                             name: tf.TensorSpec(shape=self.output_shapes[name],
                                                                                 dtype=self.output_types[name],
                                                                                 name=name) for name in
                                                             self.output_types}).repeat(1)

    def read_one_sample(self):
        """
        Read one element of the consumer_buffer
        The lock is used to prevent different threads to read and update the internal counter concurrently
        """
        with self.read_lock:
            output = None
            if self.consumer_buffer_pos < self.consumer_buffer.max_length:
                output = self.consumer_buffer.container[self.consumer_buffer_pos]
                self.consumer_buffer_pos += 1
            if self.consumer_buffer_pos == self.consumer_buffer.max_length:
                self._dump()
                self.consumer_buffer_pos = 0
            return output

    def _dump(self):
        """
        This function dumps the miner_buffer into the consumer_buffer, and restart the miner_thread
        """
        # Wait for miner to finish his job
        start_time = time.time()
        self.miner_thread.join()
        self.tot_wait += time.time() - start_time

        # Copy miner_buffer.container --> consumer_buffer.container
        self.consumer_buffer.container = self.miner_buffer.container.copy()

        # Clear miner_buffer.container
        self.miner_buffer.container.clear()

        # Restart miner_thread
        self.miner_thread = self._summon_miner_thread()

    def _collect(self):
        """
        This function collects samples.
        It is threaded by the miner_thread.
        """
        # Fill the miner_container until it's full
        while not self.miner_buffer.is_complete():
            try:
                tile_name, tuple_pos, tuple_indices = next(self.iterator)
                new_sample = self.tile_handlers[tile_name].read_tuple(tuple_pos=tuple_pos, tuple_indices=tuple_indices)
                self.miner_buffer.add(new_sample)
            except KeyboardInterrupt:
                logging.info("Interrupted by user. Exiting.")
                system.terminate()

    def _summon_miner_thread(self):
        """
        Create and starts the thread for the data collect
        """
        miner_thread = threading.Thread(target=self._collect)
        miner_thread.start()
        return miner_thread

    def _generator(self):
        """
        Generator function, used for the tf dataset
        """
        for _ in range(self.size):
            yield self.read_one_sample()

    def get_tf_dataset(self, batch_size, drop_remainder=True):
        """
        Returns a TF dataset, ready to be used with the provided batch size
        :param batch_size: the batch size
        :param drop_remainder: drop incomplete batches when True
        :return: The TF dataset
        """
        return self.tf_dataset.batch(batch_size, drop_remainder=drop_remainder)

    def get_total_wait_in_seconds(self):
        """
        Returns the number of seconds during which the data gathering was delayed due to I/O bottleneck
        :return: duration in seconds
        """
        return self.tot_wait


class RoisLoader(dict):
    """
    A class that instantiate some ROIs from a json file
    Keys:
     - "ROIS_ROOT_DIR": str
     - "TRAIN_TILES": str
     - "VALID_TILES": str

    Example of a .json file:
    {
      "ROIS_ROOT_DIR": "/data/decloud/ROI",
      "TRAIN_TILES":["T31TCK", "T31TDJ"],
      "VALID_TILES":["T31TEJ", "T31TCJ", "T31TDH"]
    }
    """

    def __init__(self, the_json):
        super().__init__()
        logging.info("Loading rois from %s", the_json)

        with open(the_json) as json_file:
            data = json.load(json_file)

        root_dir_key = "ROIS_ROOT_DIR"
        assert root_dir_key in data
        self.rois_root_dir = data[root_dir_key]
        assert isinstance(self.rois_root_dir, str)
        self.rois_root_dir = system.pathify(self.rois_root_dir)

        def get_list(key):
            """
            Retrieve a list of str
            :param key: key
            :return: list of str
            """
            assert key in data
            item = data[key]
            assert isinstance(item, list)
            return item

        # Tiles list
        self.train_tiles_list = get_list("TRAIN_TILES")
        self.valid_tiles_list = get_list("VALID_TILES")

        self.fill_dict(self.train_tiles_list, "train")
        self.fill_dict(self.valid_tiles_list, "valid")

    def fill_dict(self, tiles_list, suffix):
        """
        Check if files are there and fill dict
        :param tiles_list: tile list
        :param suffix: file suffix (e.g. "train")
        """
        tiles = {}
        for tile in tiles_list:
            roi_file = "{}{}_{}.tif".format(self.rois_root_dir, tile, suffix)
            assert system.file_exists(roi_file)
            tiles.update({tile: roi_file})
        self.update({"roi_{}".format(suffix): tiles})
