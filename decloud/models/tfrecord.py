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
"""Classes and helpers for TFRecords"""
import logging
import os
import json
from functools import partial
import glob
import tensorflow as tf
from tqdm import tqdm
from decloud.core import system
from decloud.preprocessing.normalization import normalize


class TFRecords:
    """
    This class allows to convert Dataset objects to TFRecords and to load them in dataset tensorflows format.
    """

    def __init__(self, path):
        """
        :param path: Can be a directory where TFRecords must be save/loaded or a single TFRecord path
        """
        if system.is_dir(path) or not os.path.exists(path):
            self.dirpath = path
            system.mkdir(self.dirpath)
            self.tfrecords_pattern_path = f"{self.dirpath}/*.records"
        else:
            self.dirpath = system.dirname(path)
            self.tfrecords_pattern_path = path
        self.output_types_file = f"{self.dirpath}/output_types.json"
        self.output_shape_file = f"{self.dirpath}/output_shape.json"
        self.output_shape = self.load(self.output_shape_file) if os.path.exists(self.output_shape_file) else None
        self.output_types = self.load(self.output_types_file) if os.path.exists(self.output_types_file) else None

    def _bytes_feature(self, value):
        """
        Used to convert a value to a type compatible with tf.train.Example.
        :param value: value
        :return a bytes_list from a string / byte.
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def ds2tfrecord(self, dataset, n_samples_per_shard=100, drop_remainder=True):
        """
        Convert and save samples from dataset object to tfrecord files.
        :param dataset: Dataset object to convert into a set of tfrecords
        :param n_samples_per_shard: Number of samples per shard
        :param drop_remainder: Whether additional samples should be dropped. Advisable if using multiworkers training.
                               If True, all TFRecords will have `n_samples_per_shard` samples
        """
        logging.info("%s samples", dataset.size)

        nb_shards = (dataset.size // n_samples_per_shard)
        if not drop_remainder and dataset.size % n_samples_per_shard > 0:
            nb_shards += 1

        self.convert_dataset_output_shapes(dataset)

        def _convert_data(data):
            """
            Convert data
            """
            data_converted = {}

            for k, d in data.items():
                data_converted[k] = d.name

            return data_converted

        self.save(_convert_data(dataset.output_types), self.output_types_file)

        for i in tqdm(range(nb_shards)):

            if (i + 1) * n_samples_per_shard <= dataset.size:
                nb_sample = n_samples_per_shard
            else:
                nb_sample = dataset.size - i * n_samples_per_shard

            filepath = os.path.join(self.dirpath, f"{i}.records")

            # Geographic info of all samples of the record
            geojson_path = os.path.join(self.dirpath, f"{i}.geojson")
            geojson_dic = {"type": "FeatureCollection",
                           "name": "{}_geoinfo".format(i),
                           "features": []}

            with tf.io.TFRecordWriter(filepath) as writer:
                for s in range(nb_sample):
                    sample = dataset.read_one_sample()
                    serialized_sample = {name: tf.io.serialize_tensor(fea) for name, fea in sample.items()}
                    features = {name: self._bytes_feature(serialized_tensor) for name, serialized_tensor in
                                serialized_sample.items()}
                    tf_features = tf.train.Features(feature=features)
                    example = tf.train.Example(features=tf_features)
                    writer.write(example.SerializeToString())

                    # write the geographic info of the sample inside the geojson dic
                    UL_lon, UL_lat, LR_lon, LR_lat = sample['geoinfo']
                    geojson_dic['features'].append({"type": "Feature", "properties": {"sample_id": s},
                                                    "geometry": {"type": "Polygon", "coordinates": [[[UL_lon, UL_lat],
                                                                                                     [LR_lon, UL_lat],
                                                                                                     [LR_lon, LR_lat],
                                                                                                     [UL_lon, LR_lat],
                                                                                                     [UL_lon,
                                                                                                      UL_lat]]]}})
            with open(geojson_path, 'w') as f:
                json.dump(geojson_dic, f, indent=4)

    @staticmethod
    def save(data, filepath):
        """
        Save data to pickle format.
        :param data: Data to save json format
        :param filepath: Output file name
        """

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load(filepath):
        """
        Return data from pickle format.
        :param filepath: Input file name
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    def convert_dataset_output_shapes(self, dataset):
        """
        Convert and save numpy shape to tensorflow shape.
        :param dataset: Dataset object containing output shapes
        """
        output_shapes = {}

        for key in dataset.output_shapes.keys():
            output_shapes[key] = (None,) + dataset.output_shapes[key]

        self.save(output_shapes, self.output_shape_file)

    @staticmethod
    def parse_tfrecord(example, features_types, target_keys):
        """
        Parse example object to sample dict.
        :param example: Example object to parse
        :param features_types: List of types for each feature
        :param target_keys: list of keys of the targets
        """
        read_features = {key: tf.io.FixedLenFeature([], dtype=tf.string) for key in features_types}
        example_parsed = tf.io.parse_single_example(example, read_features)

        for key in read_features.keys():
            example_parsed[key] = tf.io.parse_tensor(example_parsed[key], out_type=features_types[key])

        # Differentiating inputs and outputs
        input_parsed = {key: value for (key, value) in example_parsed.items() if key not in target_keys}
        target_parsed = {key: value for (key, value) in example_parsed.items() if key in target_keys}

        return input_parsed, target_parsed

    def normalize(self, inputs, outputs):
        """
        Normalize inputs
        :param inputs: inputs
        :params outputs: outputs (modified in the function)
        """
        normalized_outputs = {key: normalize(key, tensor) for key, tensor in outputs.items()}
        return inputs, normalized_outputs

    def read(self, batch_size, target_keys, n_workers=1, drop_remainder=True, shuffle_buffer_size=None):
        """
        Read all tfrecord files matching with pattern and convert data to tensorflow dataset.
        :param batch_size: Size of tensorflow batch
        :param target_key: Key of the target, e.g. 's2_out'
        :param n_workers: number of workers, e.g. 4 if using 4 GPUs
                                             e.g. 12 if using 3 nodes of 4 GPUs
        :param drop_remainder: whether the last batch should be dropped in the case it has fewer than
                               `batch_size` elements. True is advisable when training on multiworkers.
                               False is advisable when evaluating metrics so that all samples are used
        :param shuffle_buffer_size: is None, shuffle is not used. Else, blocks of shuffle_buffer_size
                                    elements are shuffled using uniform random.
        """
        options = tf.data.Options()
        if shuffle_buffer_size:
            options.experimental_deterministic = False  # disable order, increase speed
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO  # for multiworker
        parse = partial(self.parse_tfrecord, features_types=self.output_types, target_keys=target_keys)

        # TODO: to be investigated :
        # 1/ num_parallel_reads useful ? I/O bottleneck of not ?
        # 2/ num_parallel_calls=tf.data.experimental.AUTOTUNE useful ?
        # 3/ shuffle or not shuffle ?
        matching_files = glob.glob(self.tfrecords_pattern_path)
        logging.info('Searching TFRecords in %s...', self.tfrecords_pattern_path)
        logging.info('Number of matching TFRecords: %s', len(matching_files))
        matching_files = matching_files[:n_workers * (len(matching_files) // n_workers)]  # files multiple of workers
        nb_matching_files = len(matching_files)
        if nb_matching_files == 0:
            raise Exception("At least one worker has no TFRecord file in {}. Please ensure that the number of TFRecord "
                            "files is greater or equal than the number of workers!".format(self.tfrecords_pattern_path))
        logging.info('Reducing number of records to : %s', nb_matching_files)
        dataset = tf.data.TFRecordDataset(matching_files)  # , num_parallel_reads=2)  # interleaves reads from xxx files
        dataset = dataset.with_options(options)  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.normalize)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # TODO voir si on met le prefetch avant le batch cf https://keras.io/examples/keras_recipes/tfrecord/

        return dataset

    def read_one_sample(self, target_keys):
        """
        Read one tfrecord file matching with pattern and convert data to tensorflow dataset.
        :param target_key: Key of the target, e.g. 's2_out'
        """
        matching_files = glob.glob(self.tfrecords_pattern_path)
        one_file = matching_files[0]
        parse = partial(self.parse_tfrecord, features_types=self.output_types, target_keys=target_keys)
        dataset = tf.data.TFRecordDataset(one_file)
        dataset = dataset.map(parse)
        dataset = dataset.map(self.normalize)
        dataset = dataset.batch(1)

        sample = iter(dataset).get_next()
        return sample
