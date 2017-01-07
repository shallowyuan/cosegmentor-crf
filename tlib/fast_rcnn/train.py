# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, timdb, vimdb, troidb, vroidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.timdb = timdb
        self.timdb = vimdb
        self.troidb = troidb
        self.vroidb = vroidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(
                roidb)
        print 'done'

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100)


    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
            total_loss: Total loss from loss().
        Returns:   
            loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total
        # loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name + ' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    def eval_model(self, sess):

        data_layer = get_data_layer(self.vroidb, 'TEST')

        cls_score = self.net.get_output('cls_score')

        correct = tf.nn.in_top_k(cls_score, self.net.get_output('labels'), 1)

        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        seg_pre = tf.boolean_mask(self.net.get_output(
            'seg_score'), self.net.get_output('sweight'))
        seg_gt = tf.boolean_mask(self.net.get_output(
            'segmentation'), self.net.get_output('sweight'))

        seg_correct = tf.reduce_sum(tf.cast(tf.greater(
            tf.mul(seg_gt, tf.where(tf.greater(seg_pre, 0), 1, -1)), 0), tf.int32))

        true_count = 0

        num_examples = len(self.vroidb) * \
            cfg.TRAIN.BATCH_SIZE / cfg.TRAIN.IMS_PER_BATCH

        for i in xrange(len(self.vroidb) / cfg.TRAIN.IMS_PER_BATCH):
            blobs = data_layer.forward()

            # Make one SGD update
            feed_dict = {self.net.data: blobs['data'], self.net.rois: blob['rois'], self.net.labels: blobs['label'], self.net.keep_prob: 0.5,
                         self.net.sweights: blobs['mask_weights'] > 0, self.net.segmentation: blobs['mask_targets']}
            true_count, seg_count = sess.run(
                eval_correct, seg_correct,  feed_dict=feed_dict)
            seg_pre += float(seg_count) / (cfg.MHEIGHT * cfg.MWIDTH)
            cls_pre += true_count

        cls_precision = float(cls_pre) / num_examples
        seg_precision = seg_pre / num_examples
        print('  Num examples: %d Cls-Precision @ 1: %0.04f Seg-Precision: %0.04f' %
              (num_examples, cls_precision, seg_precision))

    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.troidb, 'TRAIN')

#        # RPN
#        # classification loss
#        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
#        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
#        # ignore_label(-1)
#        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
#        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
#        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score, rpn_label))#
#

#        # bounding box regression L1 loss
#        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
#        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
#        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
#        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])
#        smoothL1_sign = tf.cast(tf.less(tf.abs(tf.sub(rpn_bbox_pred, rpn_bbox_targets)),1),tf.float32)
#        rpn_loss_box = tf.mul(tf.reduce_mean(tf.reduce_sum(tf.mul(rpn_bbox_outside_weights,tf.add(
#                       tf.mul(tf.mul(tf.pow(tf.mul(rpn_bbox_inside_weights, tf.sub(rpn_bbox_pred, rpn_bbox_targets))*3,2),0.5),smoothL1_sign),
# tf.mul(tf.sub(tf.abs(tf.sub(rpn_bbox_pred,
# rpn_bbox_targets)),0.5/9.0),tf.abs(smoothL1_sign-1)))),
# reduction_indices=[1,2])),10)

        # R-CNN
        # classification loss
        cls_score = self.net.get_output('cls_score')
        #label = tf.placeholder(tf.int32, shape=[None])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            cls_score, self.net.get_output('labels')))

        tf.add_to_collection('losses', cross_entropy)


#       # bounding box regression L1 loss
#        bbox_pred = self.net.get_output('bbox_pred')
#        bbox_targets = self.net.get_output('roi-data')[2]
#        bbox_inside_weights = self.net.get_output('roi-data')[3]
#        bbox_outside_weights = self.net.get_output('roi-data')[4]
#        loss_box = tf.reduce_mean(tf.reduce_sum(tf.mul(bbox_outside_weights,tf.mul(bbox_inside_weights, tf.abs(tf.sub(bbox_pred, bbox_targets)))), reduction_indices=[1]))
        seg_gt = tf.boolean_mask(self.net.get_output(
            'segmentation'), self.net.get_output('sweight'))
        seg_pre = tf.boolean_mask(self.net.get_output(
            'seg_score'), self.net.get_output('sweight'))
        tf.image_summary('groundtruth', tf.reshape(seg_gt,[-1,cfg.MHEIGHT,cfg.MWIDTH,1]), max_images=100)
        tf.image_summary('prediction', tf.reshape(seg_pre,[-1,cfg.MHEIGHT,cfg.MWIDTH,1]), max_images=100)
        seg_loss = tf.reduce_mean(tf.nn.softplus(
            tf.mul(tf.mul(seg_pre, seg_gt), -1)))
        tf.add_to_collection('losses', seg_loss)

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(self.output_dir, sess.graph)
        print self.output_dir

        loss = tf.add_n([cross_entropy, seg_loss],'totalloss')
        loss_averages_op = self._add_loss_summaries(loss)

        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.MomentumOptimizer(lr, momentum)
            #train_op = opt.minimize(loss, global_step=global_step)
            grads = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # iintialize variables
        sess.run(tf.initialize_all_variables())
        if self.pretrained_model is not None:
            print('Loading pretrained model '
                  'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # get one batch
            blobs = data_layer.forward()
            print iter
            print blobs['data'].shape, blobs['rois'].shape

            # Make one SGD update
            feed_dict = {self.net.data: blobs['data'], self.net.rois: blobs['rois'], self.net.labels: blobs['labels'], self.net.keep_prob: 0.5,
                         self.net.sweights: blobs['mask_weights'] > 0, self.net.segmentation: blobs['mask_targets']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            seg_loss_value, cls_loss_value, _ = sess.run([seg_loss, cross_entropy, train_op],  feed_dict=feed_dict,
                                                         options=run_options,
                                                         run_metadata=run_metadata)

            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) +
                                  '-train-timeline.ctf.json', 'w')
                trace_file.write(
                    trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter + 1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, seg_loss: %.4f, cls_loss: %.4f,lr: %f' %\
                    (iter + 1, max_iters, cls_loss_value + seg_loss_value,
                     seg_loss_value, cls_loss_value, lr.eval())
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)
                self.eval_model(self, sess)
            if (iter + 1) % 100 == 0:
                summary_str = sess.run(summary_op,feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, iter+1)
                print 'summary %d done '%(iter + 1)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED and imdb._image_set != 'val':
        # if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    layer = RoIDataLayer(roidb, num_classes)

    return layer
def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    if num>30000:
        delinds=[23895,23895+49999]
    else:
        delinds=[]
    filtered_roidb = [entry for i, entry in enumerate(roidb) if is_valid(entry) and i not in delinds]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net(network, timdb, vimdb, troidb, vroidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    troidb = filter_roidb(troidb)
    vroidb = filter_roidb(vroidb)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, network, timdb, vimdb, troidb,
                           vroidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
