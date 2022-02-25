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
"""
Helpers for model training
"""
from tensorflow.python.client import device_lib


# ------------------------------------------------ GPU Helper --------------------------------------------------------
def get_available_gpus():
    """
    Returns a list of the identifiers of all visible GPUs.
    Source: https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# ----------------------------------------------- Saving Helper -------------------------------------------------------
# cf https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#model_saving_and_loading

def _is_chief(strategy):
    # Note: there are two possible `TF_CONFIG` configuration.
    #   1) In addition to `worker` tasks, a `chief` task type is use;
    #      in this case, this function should be modified to
    #      `return task_type == 'chief'`.
    #   2) Only `worker` task type is used; in this case, worker 0 is
    #      regarded as the chief. The implementation demonstrated here
    #      is for this case.
    # For the purpose of this Colab section, the `task_type is None` case
    # is added because it is effectively run with only a single worker.

    if strategy.cluster_resolver:  # this means MultiWorkerMirroredStrategy
        task_type, task_id = strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id
        return (task_type == 'chief') or (task_type == 'worker' and task_id == 0) or task_type is None
    else:  # strategy with only one worker
        return True
