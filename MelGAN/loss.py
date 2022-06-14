
import tensorflow as tf

def mae(x, y):
  return tf.reduce_mean(tf.abs(x - y))

def mse(x, y):
  return tf.reduce_mean((x - y)**2)

def loss_travel(sa, sab, sa1, sab1):
  l1 = tf.reduce_mean(((sa - sa1) - (sab - sab1))**2)
  l2 = tf.reduce_mean(tf.reduce_sum(-(tf.nn.l2_normalize(sa - sa1, axis = [-1]) * tf.nn.l2_normalize(sab - sab1, axis = [-1])), axis = -1))
  return l1 + l2

def loss_siamese(sa, sa1, delta):
  logits = tf.sqrt(tf.reduce_sum((sa - sa1)**2, axis = -1, keepdims = True))
  return tf.reduce_mean(tf.square(tf.maximum((delta - logits), 0)))

def d_loss_f(fake):
  return tf.reduce_mean(tf.maximum(1 + fake, 0))

def d_loss_r(real):
  return tf.reduce_mean(tf.maximum(1 - real, 0))

def g_loss_f(fake):
  return tf.reduce_mean(-fake)

