import tensorflow as tf
import tensorflow_addons as tfa

from modules import FeaturePyramidNetwork, SceneFlowEstimator, ContextNetwork, OcclusionEstimator, CostVolumeLayer


class Network(tf.keras.Model):

    def __init__(self, occlusion=True, mean_pixel=None):
        super(Network, self).__init__()
        self.occlusion = occlusion
        self.mean_pixel = tf.Variable(
            (0.0, 0.0, 0.0), trainable=False, name='mean_pixel', dtype=tf.float32)
        if mean_pixel:
            self.mean_pixel.assign(mean_pixel)

        with tf.name_scope('model'):
            self.correlation_layer = CostVolumeLayer()
            with tf.name_scope('feature_pyramid_network'):
                self.feature_pyramid_network = FeaturePyramidNetwork(
                    tf.keras.Input(shape=(None, None, 3), dtype=tf.float32))
            self.sceneflow_estimators = []
            for (d, l) in zip([367, 307, 275, 243, 211], [6, 5, 4, 3, 2]):
                with tf.name_scope('scene_flow_estimator_' + str(l)):
                    self.sceneflow_estimators.append(SceneFlowEstimator(tf.keras.Input(
                        shape=(None, None, d), dtype=tf.float32), level=l, highest_resolution=(l == 2)))
            with tf.name_scope('context_network'):
                self.context_network = ContextNetwork(
                    tf.keras.Input(shape=(None, None, 36), dtype=tf.float32))
            if occlusion:
                self.occlusion_estimators = []
                for (d, l) in zip([392, 258, 194, 130, 66], [6, 5, 4, 3, 2]):
                    with tf.name_scope('occlusion_estimator_'+str(l)):
                        self.occlusion_estimators.append(OcclusionEstimator(tf.keras.Input(
                            shape=(None, None, d), dtype=tf.float32), level=l, highest_resolution=(l == 2)))

    def call(self, inputs, training=False, mask=None):
        
        # special case for spring dataset.
        if len(inputs) == 2:
            inputs, cam_signal = inputs
            cam_signal = tf.cast(cam_signal, dtype=inputs[0].dtype)
            cam_signal  = cam_signal[:, tf.newaxis, tf.newaxis]
        else:
            cam_signal = 1 # tf.constant([1]*len(inputs))
        

        input_shape = tf.shape(inputs[0])
        input_h, input_w = input_shape[1], input_shape[2]
        h_fix = tf.cast(
            tf.round(tf.cast(input_h, tf.float32) / 64.) * 64, tf.int32)
        w_fix = tf.cast(
            tf.round(tf.cast(input_w, tf.float32) / 64.) * 64, tf.int32)
        new_size = tf.convert_to_tensor([h_fix, w_fix])

        nl1 = tf.image.resize(inputs[0], new_size) - self.mean_pixel
        nr1 = tf.image.resize(inputs[1], new_size) - self.mean_pixel
        nl2 = tf.image.resize(inputs[2], new_size) - self.mean_pixel
        nr2 = tf.image.resize(inputs[3], new_size) - self.mean_pixel

        pyramid_l1 = self.feature_pyramid_network(nl1)
        pyramid_r1 = self.feature_pyramid_network(nr1)
        pyramid_l2 = self.feature_pyramid_network(nl2)
        pyramid_r2 = self.feature_pyramid_network(nr2)

        up_flow, up_feature = None, None
        features, flow = None, None
        occ_features_up, occ_mask_up = [], []
        flows = []  # multi-scale output

        # for each relevant pyramid level
        for i, (fl1, fr1, fl2, fr2) in enumerate(zip(pyramid_l1, pyramid_r1, pyramid_l2, pyramid_r2)):
            level = 6 - i
            first_iteration = (i == 0)
            last_iteration = (level == 2)

            if first_iteration:
                wr1 = fr1
                wl2 = fl2
                wr2 = fr2
            else:
                # Careful! dense_image_warp expects flow of shape (B x) H x W x [v,u] and  s u b t r a c t s  the displacement. --> adjust scene flow accordingly
                disparity_displacement = tf.stack([tf.zeros_like(
                    up_flow[:, :, :, 2]), up_flow[:, :, :, 2]*cam_signal], axis=-1) * 20.0 / (2.0 ** level)
                wr1 = tfa.image.dense_image_warp(fr1, disparity_displacement)
                flow_displacement = -up_flow[:, :,
                                             :, 1::-1] * 20.0 / (2.0 ** level)
                wl2 = tfa.image.dense_image_warp(fl2, flow_displacement)
                cross_displacement = tf.stack([-up_flow[:, :, :, 1], (up_flow[:, :, :, 3]*cam_signal - up_flow[:, :, :, 0])], axis=-1) * 20.0 / (2.0 ** level)
                wr2 = tfa.image.dense_image_warp(fr2, cross_displacement)

            if self.occlusion:
                occ_masks = []
                for warped in [wr1, wl2, wr2]:
                    occ_inputs = tf.concat([fl1, warped], axis=-1)
                    if not first_iteration:  # all but the first iteration
                        occ_inputs = tf.concat(
                            [occ_inputs, occ_features_up.pop(0), occ_mask_up.pop(0)], axis=-1)
                    if last_iteration:
                        occ_mask = self.occlusion_estimators[i](occ_inputs)
                    else:
                        occ_mask, feat_up, mask_up = self.occlusion_estimators[i](
                            occ_inputs)
                        occ_features_up.append(feat_up)
                        occ_mask_up.append(mask_up)
                    occ_masks.append(occ_mask)
                wr1 *= occ_masks[0]
                wl2 *= occ_masks[1]
                wr2 *= occ_masks[2]

            cvr1 = self.correlation_layer(
                fl1, wr1, dimension=1, cam_signals=cam_signal)
            cvl2 = self.correlation_layer(fl1, wl2, dimension=2)
            cvr2 = self.correlation_layer(fl1, wr2, dimension=2)

            input_list = [cvr1, cvl2, cvr2, fl1]

            if not first_iteration:  # all but the first iteration
                input_list.append(up_flow)
                input_list.append(up_feature)
            estimator_input = tf.concat(input_list, axis=-1)
            if last_iteration:
                features, flow = self.sceneflow_estimators[i](estimator_input)
            else:
                flow, up_flow, up_feature = self.sceneflow_estimators[i](
                    estimator_input)
                flows.append(tf.identity(flow, name='prediction'+str(level)))

        residual_flow = self.context_network(
            tf.concat([features, flow], axis=-1))
        refined_flow = flow + residual_flow
        flows.append(refined_flow)

        prediction = tf.multiply(tf.image.resize(refined_flow, size=(
            input_h, input_w)), 20.0, name='final_prediction')

        if training:
            return prediction, flows
        else:
            return prediction
