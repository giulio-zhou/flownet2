#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import skvideo.io

parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('videopath', help='video on which to run flownet')
parser.add_argument('num_frames', help='number of video frames', default=100, type=int)
parser.add_argument('outputpath', help='path to numpy array output')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')

args = parser.parse_args()
vreader = skvideo.io.vreader(args.videopath)
metadata = skvideo.io.ffprobe(args.videopath)['video']
height, width = int(metadata['@height']), int(metadata['@width'])

if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)

# num_blobs = 2
# input_data = []

# Substitute dimensions into prototxt
vars = {}
vars['TARGET_WIDTH'] = width
vars['TARGET_HEIGHT'] = height

divisor = 64.
vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

proto = open(args.deployproto).readlines()
for line in proto:
    for key, value in vars.items():
        tag = "$%s$" % key
        line = line.replace(tag, str(value))

    tmp.write(line)

tmp.flush()

# Initialize Caffe model.
if not args.verbose:
    caffe.set_logging_disabled()
caffe.set_device(args.gpu)
caffe.set_mode_gpu()
net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)

img1, img2 = None, next(vreader) 
output_flow = []
for j in range(args.num_frames):
    img1, img2 = img2, next(vreader) 
    input_dict = {}
    input_dict[net.inputs[0]] = np.transpose(np.expand_dims(img1, 0), (0, 3, 1, 2))
    input_dict[net.inputs[1]] = np.transpose(np.expand_dims(img2, 0), (0, 3, 1, 2))
    #
    # There is some non-deterministic nan-bug in caffe
    #
    i = 1
    while i<=5:
        i+=1
    
        net.forward(**input_dict)
    
        containsNaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()
    
            if has_nan:
                print('blob %s contains nan' % name)
                containsNaN = True
    
        if not containsNaN:
            print('%d: Succeeded.' % j)
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')

    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
    output_flow.append(np.copy(blob))
np.save(args.outputpath, np.stack(output_flow, axis=0))
