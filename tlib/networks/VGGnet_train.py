import tensorflow as tf
from networks.network import Network


#define

n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        #self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        #self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.segmentation=tf.placeholder(tf.float32,shape=[None,900])
        self.rois=tf.placeholder(tf.float32,shape=[None,5])
        #self.mweights=tf.placeholder(tf.float32,shape=[None,2])
        self.sweights=tf.placeholder(tf.bool,shape=[None])
        self.labels=tf.placeholder(tf.int32,shape=[None])
        self.layers = dict({'data':self.data, 'segmentation':self.segmentation, 'sweight':self.sweights, 'labels': self.labels, "rois": self.rois})
        self.trainable = trainable
        self.setup()


    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3'))
                     #=========ROIPOOLING=======
        (self.feed('conv4_3','rois')
             .roi_pool(7, 7, 1.0/16, name='pool_4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool5'))


        #========= RPN ============
#        (self.feed('conv5_3')
#             .conv(3,3,512,1,1,name='rpn_conv/3x3')
#             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))#

#        (self.feed('rpn_cls_score','gt_boxes','im_info','data')
#             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))#

#        # Loss of rpn_cls & rpn_boxes

#        (self.feed('rpn_conv/3x3')
#             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
#        (self.feed('rpn_cls_score')
#             .reshape_layer(2,name = 'rpn_cls_score_reshape')
#             .softmax(name='rpn_cls_prob'))
#
#        (self.feed('rpn_cls_prob')
#             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))
#
#        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
#             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))
#
#        (self.feed('rpn_rois','gt_boxes')
#             .proposal_target_layer(n_classes,name = 'roi-data'))


        #========= RCNN ============
        (self.feed('pool5')
             .fc(1024, name='fc6')
             .dropout(0.5, name='drop6')
             .fc(1024, name='fc7')
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

       # (self.feed('drop7')
       #     .fc(n_classes*4, relu=False, name='bbox_pred'))

        #==========segment network===
        (self.feed('conv5_3')
             .conv(1,1,512,1 , 1, padding='VALID', name='conv5_4')
             .fc(512, name='fc8')
             .fc(900, relu=False, name='seg_score'))

