
"""Jaccard Coefficient and Jaccard Distance loss."""

import tensorflow as tf


class JaccardCoefficient(tf.keras.metrics.Metric):
    """Jaccard Coefficient calculator."""
    
    def __init__(self, threshold=0.5, name='jaccard_coefficient', **kwargs):
        
        super(JaccardCoefficient, self).__init__(name=name, **kwargs)
        self.threshold = tf.constant(threshold)
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        assert sample_weight is None

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.less_equal(self.threshold, y_pred)

        intersection = tf.logical_and(y_true, y_pred)
        intersection = tf.cast(intersection, self.dtype)
        self.intersection.assign_add(tf.reduce_sum(intersection))

        union = tf.logical_or(y_true, y_pred)
        union = tf.cast(union, self.dtype)
        self.union.assign_add(tf.reduce_sum(union))

    def result(self):
        return tf.divide(self.intersection, self.union)

    def reset_state(self):
        self.intersection.assign(0)
        self.union.assign(0)
        
        
def jaccard_distance_loss(y_true, y_pred):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true + y_pred)
    jac = intersection / (sum_ - intersection)
    return (1 - jac)