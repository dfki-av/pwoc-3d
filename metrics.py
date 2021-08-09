import sys
import tensorflow as tf


class SceneFlowMetrics(tf.keras.metrics.Metric):

    def __init__(self, name="SF", **kwargs):
        super(SceneFlowMetrics, self).__init__(name=name, **kwargs)

        self.sum_d0_epe = self.add_weight(name="sum_d0_epe", initializer="zeros")
        self.sum_d1_epe = self.add_weight(name="sum_d1_epe", initializer="zeros")
        self.sum_of_epe = self.add_weight(name="sum_of_epe", initializer="zeros")
        self.sum_sf_epe = self.add_weight(name="sum_sf_epe", initializer="zeros")
        self.sum_epe = self.add_weight(name="sum_epe", initializer="zeros")
        self.sum_d0_outliers = self.add_weight(name="sum_d0_outliers", initializer="zeros")
        self.sum_d1_outliers = self.add_weight(name="sum_d1_outliers", initializer="zeros")
        self.sum_of_outliers = self.add_weight(name="sum_of_outliers", initializer="zeros")
        self.sum_sf_outliers = self.add_weight(name="sum_sf_outliers", initializer="zeros")
        self.sum_valid_pixels = self.add_weight(name="sum_valid_pixels", initializer="zeros")


    def update_state(self, y_true, y_pred, sample_weight=None):
        gt_mask = y_true[:, :, :, 2] > 0.
        gt_masked = tf.boolean_mask(y_true, gt_mask)
        pred_masked = tf.boolean_mask(y_pred, gt_mask)
        gt_flow_mag = tf.norm(gt_masked[:, :2], axis=-1)
        abs_diff = tf.abs(pred_masked - gt_masked)

        flow_epe = tf.norm(abs_diff[:, :2], axis=-1, name='flow_epe')
        d0_epe = abs_diff[:, 2]
        d1_epe = abs_diff[:, 3]

        d0_outliers = tf.logical_and(d0_epe > 3., d0_epe > 0.05*gt_masked[:, 2])
        d1_outliers = tf.logical_and(d1_epe > 3., d1_epe > 0.05*gt_masked[:, 3])

        flow_outliers = tf.logical_and(flow_epe > 3., flow_epe > 0.05*gt_flow_mag)

        sf_outlier = tf.logical_or(flow_outliers, tf.logical_or(d0_outliers, d1_outliers))

        self.sum_d0_epe.assign_add(tf.reduce_sum(d0_epe))
        self.sum_d1_epe.assign_add(tf.reduce_sum(d1_epe))
        self.sum_of_epe.assign_add(tf.reduce_sum(flow_epe))
        self.sum_sf_epe.assign_add(tf.reduce_sum(tf.norm(abs_diff, axis=-1)))
        self.sum_epe.assign_add(tf.reduce_sum(flow_epe+d0_epe+d1_epe))
        self.sum_d0_outliers.assign_add(tf.reduce_sum(tf.cast(d0_outliers, tf.float32)))
        self.sum_d1_outliers.assign_add(tf.reduce_sum(tf.cast(d1_outliers, tf.float32)))
        self.sum_of_outliers.assign_add(tf.reduce_sum(tf.cast(flow_outliers, tf.float32)))
        self.sum_sf_outliers.assign_add(tf.reduce_sum(tf.cast(sf_outlier, tf.float32)))
        self.sum_valid_pixels.assign_add(tf.cast(tf.shape(gt_masked)[0], tf.float32))


    def result(self):
        return (
            (self.sum_sf_outliers / self.sum_valid_pixels) * 100,
            (self.sum_of_outliers / self.sum_valid_pixels) * 100,
            (self.sum_d1_outliers / self.sum_valid_pixels) * 100,
            (self.sum_d0_outliers / self.sum_valid_pixels) * 100,
            self.sum_epe / self.sum_valid_pixels,
            self.sum_sf_epe / self.sum_valid_pixels,
            self.sum_of_epe / self.sum_valid_pixels,
            self.sum_d1_epe / self.sum_valid_pixels,
            self.sum_d0_epe / self.sum_valid_pixels,
        )


    def log(self, writer, epoch):
        sf_koe, of_koe, d1_koe, d0_koe, sum_epe, sf_epe, of_epe, d1_epe, d0_epe = self.result()
        with writer.as_default():
            tf.summary.scalar('outlier/SF_KOE', sf_koe, step=epoch)
            tf.summary.scalar('outlier/OF_KOE', of_koe, step=epoch)
            tf.summary.scalar('outlier/D1_KOE', d1_koe, step=epoch)
            tf.summary.scalar('outlier/D0_KOE', d0_koe, step=epoch)
            tf.summary.scalar('end-point-error/SF_EPE', sf_epe, step=epoch)
            tf.summary.scalar('end-point-error/OF_EPE', of_epe, step=epoch)
            tf.summary.scalar('end-point-error/D1_EPE', d1_epe, step=epoch)
            tf.summary.scalar('end-point-error/D0_EPE', d0_epe, step=epoch)
        writer.flush()
        return


    def print(self, stream=sys.stdout):
        sf_koe, of_koe, d1_koe, d0_koe, sum_epe, sf_epe, of_epe, d1_epe, d0_epe = self.result()
        stream.write("EPE (px):\tD1\tD2\tOF\tSF(sum)\tSF(4d)\t\tKOE (%):\tD1\tD2\tOF\tSF\n")
        stream.write("\t\t%.02f\t%.02f\t%.02f\t%.02f\t%.02f\t\t\t\t%.02f\t%.02f\t%.02f\t%.02f\n" % (
            d0_epe, d1_epe, of_epe, sum_epe, sf_epe, d0_koe, d1_koe, of_koe, sf_koe))
        return
