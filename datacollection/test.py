import numpy as np
import tensorflow as tf
x = tf.Variable(tf.random.uniform([5, 30], -1, 1))
split0, split1, split2 = tf.split(x, [10, 10, 10], 1)
print(split0)