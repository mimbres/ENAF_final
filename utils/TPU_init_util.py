# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Setup TPU.

Created on Wed Jul  8 10:46:31 2020
@author: skchang@cochlear.ai
"""
import tensorflow as tf

def init_tpu(tpu_name, n_replicas):
    
    # Initialization
    print(tf.__version__)
    tf.config.set_soft_device_placement(True) # Since TF 2.1, allow CPU-ops within TPU workers (eg. tf.print and loggers)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    
    # Device assignment
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology=topology, 
        num_replicas=n_replicas)
    
    strategy = tf.distribute.experimental.TPUStrategy(resolver,
        device_assignment=device_assignment)
    
    print('Number of replicas in sync: {}'.format(
        strategy.num_replicas_in_sync))
    
    return strategy