import tensorflow as tf
import numpy as np


def initial_params(dims, dtype=np.float32, names=None):
    dims = list(dims)
    if names is None:
        names = list(names) * len(dims)

    if len(dims) != len(names):
        raise("dims argument gets length %i, and names gets  length %i."
              "Length needs to be match".format(len(dims), len(names)))
    params = {}
    for dim, name, in zip(dims, names):
        params[name] = tf.Variable(tf.random_normal(dim, stddev=0.35),
                                   name=name, dtype=dtype)
    return params


def initial_params_dict(config, dtype=np.float32):
    names = config.keys()
    dims = config.values()
    return initial_params(dims, names, dtype=np.float32)
