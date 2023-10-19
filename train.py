#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend
from tqdm import tqdm

import network
import datasets
import losses
import metrics


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
      tf.config.experimental.set_memory_growth(device, True)
    except:
      pass # Invalid device or cannot modify virtual devices once initialized.


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--finetune', action='store_true', default=True)
parser.add_argument('--noocc', action='store_true', default=False)
parser.add_argument('--init_with', action='store', default=None)
parser.add_argument('--train_spring', action='store_true', default=False)
args = parser.parse_args()

if args.pretrain:
    modelname = "pwoc3d-ft3d" + ('-noocc' if args.noocc else '')
    batch_size = 2
    train_data = datasets.get_ft3d_dataset(subset='TRAIN', batch_size=batch_size, augment=True, shuffle=True, temporal_augmentation=True)
    valid_data = datasets.get_ft3d_dataset(subset='VALID', batch_size=1)
    n_train_samples = datasets.FT3D_TRAINING_SAMPLES
    n_val_samples = datasets.FT3D_VALIDATION_SAMPLES
    mean_pixel = datasets.FT3D_MEAN_PIXEL
    n_epochs = 100
    steps_per_epoch = int(np.ceil(n_train_samples/batch_size))
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([x*steps_per_epoch for x in [65, 85]], [0.0001, 0.00005, 0.00001], name='learning_rate')
    loss_fn = lambda flows, gt: losses.multi_scale_loss(gt, flows, losses.loss_per_scale, weights=[1., 1., 1., 2., 4.])

elif args.train_spring:

    train_cache = None
    val_cache = None

    modelname = "pwoc3d-spring" + ('-noocc' if args.noocc else '')
    batch_size = 2
    spring_scene_dict = datasets.SPRING_SCENE_DICT
    
    train_dataset = datasets.SpringDataset(datasets.BASEPATH_SPRING, datasets.SPRING_TRAINING_IDXS, spring_scene_dict, shuffle=True)
    val_dataset = datasets.SpringDataset(datasets.BASEPATH_SPRING, datasets.SPRING_VALIDATION_IDXS, spring_scene_dict)
    n_train_samples = len(train_dataset)
    n_val_samples = len(val_dataset)
    train_data = datasets.get_spring_dataset(train_dataset, batch_size, augment=True, cache_path=train_cache)
    valid_data = datasets.get_spring_dataset(val_dataset, 1, cache_path=val_cache)
    mean_pixel = datasets.SPRING_MEAN_PIXEL
    steps_per_epoch = int(np.ceil(n_train_samples/batch_size))
    if args.init_with:
        n_epochs = 125
        learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100 * steps_per_epoch, ], [0.00005, 0.00001], name='learning_rate')
        
    else:
        n_epochs = 100
        learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([x*steps_per_epoch for x in [65, 85]], [0.0001, 0.00005, 0.00001], name='learning_rate')
    
    loss_fn = lambda flows, gt: losses.finetuning_loss(gt, flows[-1])
    
    
else:
    modelname = "pwoc3d-kitti" + ('-noocc' if args.noocc else '')
    if not args.init_with:
        args.init_with = "data/pwoc3d-ft3d"
    batch_size = 1
    train_data = datasets.get_kitti_dataset(datasets.KITTI_TRAIN_IDXS, batch_size=batch_size, augment=True, shuffle=True)
    valid_data = datasets.get_kitti_dataset(datasets.KITTI_VALIDATION_IDXS, batch_size=1)
    n_train_samples = datasets.KITTI_TRAIN_SAMPLES
    n_val_samples = datasets.KITTI_VALIDATION_SAMPLES
    mean_pixel = datasets.KITTI_MEAN_PIXEL
    n_epochs = 125
    steps_per_epoch = int(np.ceil(n_train_samples/batch_size))
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100 * steps_per_epoch, ], [0.00005, 0.00001], name='learning_rate')
    loss_fn = lambda flows, gt: losses.finetuning_loss(gt, flows[-1])

with backend.get_graph().as_default():
    net = network.Network(occlusion=not args.noocc, mean_pixel=mean_pixel)

if args.init_with:
    print("Loading pre-trained weights from %s..." % args.init_with)
    net.load_weights(args.init_with)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0), best_sf_koe=tf.Variable(100.0), optimizer=optimizer, model=net)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory="./models/"+modelname, max_to_keep=5, checkpoint_name='ckpt')
if checkpoint_manager.latest_checkpoint:
    print("Resuming training from %s..." % checkpoint_manager.latest_checkpoint)
    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)

train_metric = metrics.SceneFlowMetrics()
validation_metric = metrics.SceneFlowMetrics()
avg_train_loss = tf.keras.metrics.Mean()
avg_validation_loss = tf.keras.metrics.Mean()

train_summary_writer = tf.summary.create_file_writer("./summaries/"+modelname+"/train")
validation_summary_writer = tf.summary.create_file_writer("./summaries/"+modelname+"/validation")


@tf.function
def training_step(image_batch, gt_batch):
    with tf.GradientTape() as tape:
        pred, flows = net(image_batch, training=True)
        loss_value = loss_fn(flows, gt_batch)

    grads = tape.gradient(loss_value, net.trainable_weights)
    optimizer.apply_gradients(zip(grads, net.trainable_weights))
    return pred, flows, loss_value


@tf.function
def validation_step(image_batch, gt_batch):
    pred, flows = net(image_batch, training=True)
    loss_value = loss_fn(flows, gt_batch)
    return pred, flows, loss_value


initial_epoch = checkpoint.epoch.numpy() + 1
for epoch in range(initial_epoch,n_epochs+1):

    # training
    loss_value = tf.constant(0.0)
    with tqdm(enumerate(train_data), desc=('Epoch %d' % epoch), total=steps_per_epoch) as t:
        t.set_description('Epoch %d:' % epoch)
        for step, (image_batch, gt_batch) in t:
            t.set_postfix(loss=loss_value.numpy())

            pred, flows, loss_value = training_step(image_batch, gt_batch)

            train_metric.update_state(gt_batch, pred)
            avg_train_loss.update_state(loss_value)

    print("Training of epoch %d complete." % epoch)
    avg_loss = avg_train_loss.result()
    sf_koe, of_koe, d1_koe, d0_koe, sum_epe, sf_epe, of_epe, d1_epe, d0_epe = train_metric.result()
    print("Average Loss: %.4f \t Average SF KOE: %.2f \t Average EPE %.2f" % (avg_loss, sf_koe, sf_epe))

    # validation
    loss_value = tf.constant(0.0)
    with tqdm(enumerate(valid_data), desc=('Validation, Epoch %d' % epoch), total=n_val_samples) as t:
        for step, (image_batch, gt_batch) in t:
            t.set_postfix(loss=loss_value.numpy())
            pred, flows, loss_value = validation_step(image_batch, gt_batch)
            validation_metric.update_state(gt_batch, pred)
            avg_validation_loss.update_state(loss_value)

    print("Validation after epoch %d complete." % epoch)
    validation_avg_loss = avg_validation_loss.result()
    validation_sf_koe, validation_of_koe, validation_d1_koe, validation_d0_koe, validation_sum_epe, validation_sf_epe, validation_of_epe, validation_d1_epe, validation_d0_epe = validation_metric.result()
    print("Average Validation Loss: %.4f \t Average Validation SF KOE: %.2f \t Average Validation EPE %.2f" % (validation_avg_loss, validation_sf_koe, validation_sf_epe))

    # logging
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', avg_train_loss.result(), step=epoch)
    avg_train_loss.reset_states()
    train_metric.log(writer=train_summary_writer, epoch=epoch)
    train_metric.reset_states()
    with validation_summary_writer.as_default():
        tf.summary.scalar('loss', avg_validation_loss.result(), step=epoch)
    avg_validation_loss.reset_states()
    validation_metric.log(writer=validation_summary_writer, epoch=epoch)
    validation_metric.reset_states()

    # saving
    checkpoint.epoch.assign_add(1)
    if validation_sf_koe < checkpoint.best_sf_koe:
        checkpoint.best_sf_koe.assign(validation_sf_koe)
        checkpoint_manager.save(checkpoint_number=epoch)

print("Training complete.")
print("Saving weights for best checkpoint...")
checkpoint.restore(checkpoint_manager.latest_checkpoint)
net.save_weights("./models/"+modelname+"/"+modelname)
print("...done.")