#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import h5py
import skvideo.io

parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('hdf5path', help='hdf5 file on which to run flownet')
parser.add_argument('batchsize', help='number of images to use in a batch', default=1, type=int)
parser.add_argument('num_frames', help='number of video frames', default=100, type=int)
parser.add_argument('outputpath', help='path to numpy array output')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')

args = parser.parse_args()
input_hdf5_file = h5py.File(args.hdf5path, 'r')
data, labels = input_hdf5_file['data'], input_hdf5_file['labels'][:]
height, width = input_hdf5_file['data'].shape[1:3]

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

vars['BATCH_SIZE'] = args.batchsize

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

imgs = [data[0]]
output_flow = []
hdf5_file = h5py.File(args.outputpath, 'w')
hdf5_file.create_dataset('data', [args.num_frames - 1, height, width, 2], 'f2')
for j in range(0, args.num_frames - 1, args.batchsize):
    start, end = j, min(j + args.batchsize, args.num_frames - 1)
    imgs = [imgs[-1]] + list(data[start:end])
    if len(imgs) < args.batchsize + 1:
        imgs = imgs + [imgs[-1]] * (args.batchsize + 1 - len(imgs))
    input_dict = {}
    input_dict[net.inputs[0]] = np.transpose(imgs[:-1], (0, 3, 1, 2))
    input_dict[net.inputs[1]] = np.transpose(imgs[1:], (0, 3, 1, 2))
    net.forward(**input_dict)

    blob = net.blobs['predict_flow_final'].data.transpose(0, 2, 3, 1)
    blob = blob[:end-start]
    hdf5_file['data'][j:j+end-start] = np.copy(blob).astype(np.float16)
    print(j, args.num_frames)
