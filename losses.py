import tensorflow as tf


def multi_scale_loss(gt, multiscale_predictions, loss_fn, weights=(1., 1., 1., 2., 4.)):
    loss = 0.0
    for pred, w in zip(multiscale_predictions, weights):
        loss += w * loss_fn(gt, pred)
    return loss


def loss_per_scale(gt, prediction):
    resized_gt = tf.image.resize(gt, size=tf.shape(prediction)[1:3])
    loss = tf.reduce_mean(tf.linalg.norm(resized_gt / 20.0 - prediction, axis=-1))
    return loss


def finetuning_loss(gt, prediction):
    gtmask = gt[:,:,:,2]
    masked_gt = tf.boolean_mask(gt, gtmask)
    resized_pred = tf.image.resize(prediction, size=tf.shape(gt)[1:3])
    masked_pred = tf.boolean_mask(resized_pred, gtmask)
    loss = tf.reduce_mean(tf.pow(tf.reduce_sum(tf.abs((masked_gt / 20.0) - masked_pred), axis=-1) + 0.01, 0.4))
    return loss
