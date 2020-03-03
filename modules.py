import tensorflow as tf


class CostVolumeLayer(tf.Module):

    def __init__(self, search_range=4):
        super(CostVolumeLayer, self).__init__(name='cost_volume')
        self.window = search_range

    def __call__(self, x, warped, dimension=2):
        assert dimension in [1, 2]

        total = []
        keys = []

        if dimension == 1:
            row_shifted = warped
            for i in range(2 * self.window + 1):
                if i != 0:
                    row_shifted = tf.pad(row_shifted, [[0, 0], [0, 0], [1, 0], [0, 0]])
                    row_shifted = tf.keras.layers.Cropping2D([[0, 0], [0, 1]])(row_shifted)
                total.append(tf.reduce_mean(row_shifted * x, axis=-1))
            stacked = tf.stack(total, axis=3)
            return stacked / (2 * self.window + 1)

        else:  # dimension == 2
            row_shifted = [warped]
            for i in range(self.window+1):
                if i != 0:
                    row_shifted = [tf.pad(row_shifted[0], [[0, 0], [0, 1], [0, 0], [0, 0]]),
                                   tf.pad(row_shifted[1], [[0, 0], [1, 0], [0, 0], [0, 0]])]
                    row_shifted = [tf.keras.layers.Cropping2D([[1, 0], [0, 0]])(row_shifted[0]),
                                   tf.keras.layers.Cropping2D([[0, 1], [0, 0]])(row_shifted[1])]
                for side in range(len(row_shifted)):
                    total.append(tf.reduce_mean(row_shifted[side] * x, axis=-1))
                    keys.append([i * (-1) ** side, 0])
                    col_previous = [row_shifted[side], row_shifted[side]]
                    for j in range(1, self.window+1):
                        col_shifted = [tf.pad(col_previous[0], [[0, 0], [0, 0], [0, 1], [0, 0]]),
                           tf.pad(col_previous[1], [[0, 0], [0, 0], [1, 0], [0, 0]])]
                        col_shifted = [tf.keras.layers.Cropping2D([[0, 0], [1, 0]])(col_shifted[0]),
                                       tf.keras.layers.Cropping2D([[0, 0], [0, 1]])(col_shifted[1])]
                        for col_side in range(len(col_shifted)):
                            total.append(tf.reduce_mean(col_shifted[col_side] * x, axis=-1))
                            keys.append([i * (-1) ** side, j * (-1) ** col_side])
                        col_previous = col_shifted

                if i == 0:
                    row_shifted *= 2

            total = [t for t, _ in sorted(zip(total, keys), key=lambda pair: pair[1])]
            stacked = tf.stack(total, axis=3)

            return stacked / ((2.0*self.window+1)**2.0)


def FeaturePyramidNetwork(image, activation_fn=tf.keras.layers.LeakyReLU(0.1), reg_constant=0):

    def Conv(filters, stride, name_suffix):
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=(stride, stride), padding='same', activation=activation_fn, kernel_initializer=tf.keras.initializers.Orthogonal, kernel_regularizer=tf.keras.regularizers.l2(reg_constant), name='conv'+name_suffix)

    def UpConv(filters, name_suffix):
        return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=(2, 2), padding='same', activation=activation_fn, kernel_initializer=tf.keras.initializers.Orthogonal, kernel_regularizer=tf.keras.regularizers.l2(reg_constant), name='deconv'+name_suffix)

    conv1a = Conv(filters=16, stride=2, name_suffix='1a')(image)
    conv1b = Conv(filters=16, stride=1, name_suffix='1b')(conv1a)
    conv2a = Conv(filters=32, stride=2, name_suffix='2a')(conv1b)
    conv2b = Conv(filters=32, stride=1, name_suffix='2b')(conv2a)
    conv3a = Conv(filters=64, stride=2, name_suffix='3a')(conv2b)
    conv3b = Conv(filters=64, stride=1, name_suffix='3b')(conv3a)
    conv4a = Conv(filters=96, stride=2, name_suffix='4a')(conv3b)
    conv4b = Conv(filters=96, stride=1, name_suffix='4b')(conv4a)
    conv5a = Conv(filters=128, stride=2, name_suffix='5a')(conv4b)
    conv5b = Conv(filters=128, stride=1, name_suffix='5b')(conv5a)
    conv6a = Conv(filters=196, stride=2, name_suffix='6a')(conv5b)
    conv6b = Conv(filters=196, stride=1, name_suffix='6b')(conv6a)

    pyr_top = tf.keras.layers.Conv2D(filters=196, kernel_size=1, strides=(1, 1), padding='same', activation=activation_fn, kernel_initializer=tf.keras.initializers.Orthogonal, kernel_regularizer=tf.keras.regularizers.l2(reg_constant), name='conv_pyr_top')(conv6b)

    pyramid = [pyr_top]

    for i, skip_features in zip(range(4, 0, -1), [conv5b, conv4b, conv3b, conv2b]):
        channels = skip_features.shape.as_list()[-1]
        upsampled = UpConv(filters=channels, name_suffix='_upsample' + str(i + 1) + 'to' + str(i))(pyramid[-1])
        merged_features = upsampled + skip_features
        refine = Conv(filters=channels, stride=1, name_suffix='_refine' + str(i))(merged_features)
        pyramid.append(refine)

    return tf.keras.Model(inputs=image, outputs=pyramid, name='feature_pyramid_network')


def SceneFlowEstimator(feature_volume, level, activation_fn=tf.keras.layers.LeakyReLU(0.1), reg_constant=0, highest_resolution=False):

    def Conv(filters, name_suffix, activation=activation_fn):
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=(1, 1), padding='same', activation=activation, kernel_initializer=tf.keras.initializers.Orthogonal, kernel_regularizer=tf.keras.regularizers.l2(reg_constant), name='conv'+name_suffix)

    def UpConv(filters, name_suffix, activation=activation_fn):
        return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=(2, 2), padding='same', activation=activation, kernel_initializer=tf.keras.initializers.Orthogonal, kernel_regularizer=tf.keras.regularizers.l2(reg_constant), name='deconv'+name_suffix)

    conv1 = Conv(filters=128, name_suffix='1')(feature_volume)
    conv2 = Conv(filters=128, name_suffix='2')(conv1)
    conv3 = Conv(filters=96, name_suffix='3')(conv2)
    conv4 = Conv(filters=64, name_suffix='4')(conv3)

    f_lev = Conv(filters=32, name_suffix='_f')(conv4)
    w_lev = Conv(filters=4, name_suffix='_w', activation=tf.keras.activations.linear)(f_lev)

    if highest_resolution:
        return tf.keras.Model(inputs=feature_volume, outputs=(f_lev, w_lev), name='scene_flow_estimator_'+str(level))
    else:
        flow_up = UpConv(filters=4, name_suffix='_up_flow', activation=tf.keras.activations.linear)(w_lev)
        feature_up = UpConv(filters=4, name_suffix='_up_feature', activation=tf.keras.activations.linear)(f_lev)
        return tf.keras.Model(inputs=feature_volume, outputs=(w_lev, flow_up, feature_up), name='scene_flow_estimator_'+str(level))


def ContextNetwork(feature_volume, activation_fn=tf.keras.layers.LeakyReLU(0.1), reg_constant=0):

    def Conv(filters, name_suffix, dilation, activation=activation_fn):
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=(1, 1), padding='same', activation=activation, dilation_rate=(dilation, dilation), kernel_initializer=tf.keras.initializers.Orthogonal, kernel_regularizer=tf.keras.regularizers.l2(reg_constant), name='conv'+name_suffix)

    conv1 = Conv(filters=128, name_suffix='1', dilation=1)(feature_volume)
    conv2 = Conv(filters=128, name_suffix='2', dilation=2)(conv1)
    conv3 = Conv(filters=128, name_suffix='3', dilation=4)(conv2)
    conv4 = Conv(filters=96, name_suffix='4', dilation=8)(conv3)
    conv5 = Conv(filters=64, name_suffix='5', dilation=16)(conv4)
    conv6 = Conv(filters=32, name_suffix='6', dilation=1)(conv5)
    conv7 = Conv(filters=4, name_suffix='7', dilation=1, activation=tf.keras.activations.linear)(conv6)

    return tf.keras.Model(inputs=feature_volume, outputs=conv7, name='context_network')


def OcclusionEstimator(features, level, activation_fn=tf.keras.layers.LeakyReLU(0.1), reg_constant=0, highest_resolution=False):

    def Conv(filters, name_suffix, activation=activation_fn):
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation=activation, kernel_initializer=tf.keras.initializers.Orthogonal, kernel_regularizer=tf.keras.regularizers.l2(reg_constant), name='conv'+name_suffix)

    def UpConv(filters, name_suffix, activation=activation_fn):
        return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=(2, 2), padding='same', activation=activation, kernel_initializer=tf.keras.initializers.Orthogonal, kernel_regularizer=tf.keras.regularizers.l2(reg_constant), name='deconv'+name_suffix)

    conv1 = Conv(filters=128, name_suffix='1')(features)
    conv2 = Conv(filters=96, name_suffix='2')(conv1)
    conv3 = Conv(filters=64, name_suffix='3')(conv2)
    conv4 = Conv(filters=32, name_suffix='4')(conv3)
    feat = Conv(filters=16, name_suffix='_feat')(conv4)
    occ_mask = Conv(filters=1, name_suffix='_occ_mask', activation=tf.keras.activations.sigmoid)(feat)

    if highest_resolution:
        return tf.keras.Model(inputs=features, outputs=occ_mask, name='context_network'+str(level))
    else:
        features_up = UpConv(filters=1, name_suffix='_up_feat', activation=tf.keras.activations.sigmoid)(feat)
        occ_mask_up = UpConv(filters=1, name_suffix='_up_occ_mask', activation=tf.keras.activations.sigmoid)(occ_mask)
        return tf.keras.Model(inputs=features, outputs=(occ_mask, features_up, occ_mask_up), name='context_network'+str(level))
