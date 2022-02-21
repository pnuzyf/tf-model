# coding=utf-8
# @Time     : 2021/5/12 20:11
# @Auther   : lionpeng@tencent.com
import tensorflow as tf


class BPRLoss(tf.keras.losses.Loss):
    def __init__(self, name="BPR_loss"):
        super(BPRLoss, self).__init__(name=name)

    @tf.function
    def call(self, y_true, y_pred):
        # [batch, 1]
        loss = tf.math.log1p(tf.math.exp(-y_pred))
        return loss


class PairHingeLoss(tf.keras.losses.Loss):
    def __init__(self, margin, name="pair_hinge_loss"):
        super(PairHingeLoss, self).__init__(name=name)
        assert margin >= 0, "'margin' must be >= 0, got {}".format(margin)
        self.margin = float(margin)

    @tf.function
    def call(self, y_true, y_pred):
        # [batch, 1]
        gap = self.margin - y_pred
        # [batch, 1]
        loss = tf.math.maximum(0., gap)
        return loss


#梯度修正，只有正样本才计算梯度
class TruncatedMseLoss(tf.keras.losses.Loss):
    def __init__(self, name="tuncated_mse_loss"):
        super(TruncatedMseLoss, self).__init__(name=name)

    @tf.function
    def call(self, y_true, y_pred):
        # [batch, 1] loss = square(y_true - y_pred)`
        denominator = tf.math.count_nonzero(y_true)
        index = tf.not_equal(y_true, 0)
        y_pred = tf.where(index, y_pred, y_true)
        sqr = tf.square(y_pred - y_true)
        loss = tf.reduce_sum(sqr) / tf.cast(denominator, dtype=tf.float32)
        return loss

class InfoNceLoss(tf.keras.losses.Loss):
    def __init__(self, name="info_nce"):
        super(InfoNceLoss, self).__init__(name=name)
        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                      reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def call(self, y_true, y_pred):
        num_queries = tf.shape(y_pred)[0] #batch_size
        num_candidates = tf.shape(y_pred)[1] #batch_size
        labels = tf.eye(num_queries, num_candidates) #创建单位矩阵 -- 对角线才是label
        loss = self.cce(labels, y_pred)
        return loss