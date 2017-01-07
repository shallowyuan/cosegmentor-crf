# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a cosegment R-CNN network.

MaskLossLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
import pickle


class MaskLossLayer(caffe.Layer):
    """SEG R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        #self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape((1))
        self._name_to_top_map['loss'] = idx
        idx += 1
        top[idx].reshape((1))
        self._name_to_top_map['accu'] = idx
        idx += 1
        self.count = 0
        self.losscount = 0
        self.smoothloss = 0.0
        self.smoothaccu = 0.0
        self.losscache = np.ones((100, 1), dtype='float32') * -1
        self.acache = np.ones((100, 1), dtype='float32') * -1

        print 'MaskRegLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""

        predict = bottom[0].data.squeeze()
        pweights = bottom[1].data.squeeze()
        labels = np.where(pweights > 0)[0]
        gt = bottom[2].data.squeeze()
    #    print np.max(gt),'--------------mask\n'
    #	print gt.shape,predict.shape
        assert gt.shape == predict.shape
        # if self.count==0:
        # print gt,'\n-------------------',pweights,'\n----------------------',predict,'\n---------------'
        #   self.count+=1
        # print np.mean(np.log(1+np.exp(-np.multiply(gt,predict))),axis=1),'\n',pweights
        # debug
#        fn ='shape_%d.pkl'%self.count
#        with open(fn,'w') as f:
#            pickle.dump(gt,f)
        predict = predict[labels, ...]
        pweights = pweights[labels]
        gt = gt[labels, ...]
        # print gt.shape,predict.shape

        loss = np.average(np.mean(
            np.log(1 + np.exp(-np.multiply(gt, predict))), axis=1), weights=pweights)
        top[0].data[0] = loss

        # for balanced accuracy
        ypredict=np.where(predict>=0.1,1,-1);
        eaccu = np.mean(ypredict == gt)
        top[1].data[0] = eaccu
#        npos = np.mean(gt == 1, 1)
#        nneg = 1 - npos
#        plabels = np.where(gt==1)[0]
#        predict_pos=
#        predict_pos[plabel]=np.inf
#        predict_neg[plabel]=np.inf
#        pweights_pos=pweights*npos
#        pweights_neg=pweights*nneg
        # output loss
        if self.losscache[self.count] < 0:
            self.losscache[self.count] = loss
            self.acache[self.count] = eaccu
            self.smoothloss = (
                self.count * self.smoothloss + loss) / (self.count + 1)
            self.smoothaccu = (
                self.count * self.smoothaccu + eaccu) / (self.count + 1)
        else:
            self.smoothloss += (loss - self.losscache[self.count]) / 100
            self.smoothaccu += (eaccu - self.acache[self.count]) / 100
            self.losscache[self.count] = loss
            self.acache[self.count] = eaccu
        self.count = (self.count + 1) % 100
        if self.count == 0 and self.phase == "TRAIN":
            output = open(cfg.TRAIN_SEGLOSS_OUTPUT, 'a')
            # print self.smoothloss,' loss \n'
            output.write('%f %f\n' % (self.smoothaccu, self.smoothloss))
            output.close()

    def backward(self, top, propagate_down, bottom):
        """This layer does  propagate gradients."""
        # print 'backward pass\n'
        if propagate_down[0]:
            gradients = np.zeros(bottom[0].shape, dtype=np.float32)
            weights = bottom[1].data.squeeze()
            labels = np.where(weights > 0)[0]
            gt = bottom[2].data[labels, ...]
            predict = bottom[0].data[labels, ...]
            weights = weights[labels]
            assert gt.shape == predict.shape
            pweights = np.tile(
                weights, [predict.shape[1], 1]).transpose([1, 0])
            pweights = pweights / np.sum(pweights)

            gradients[labels, ...] = -pweights * \
                np.exp(-gt * predict) * gt / (1 + np.exp(-gt * predict))
            # print gradients.squeeze(),'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            # print top[0].diff,'top_diff\n'
            bottom[0].diff[...] = gradients * top[0].diff.flat[0]

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
