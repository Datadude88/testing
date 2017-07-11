import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

net = caffe.Net('/home/tsawicki/Desktop/conv.prototxt', caffe.TEST)

print net.inputs
print net.blobs

[(k, v.data.shape) for k, v in net.blobs.items()]

net.params['conv'][0]


im = np.array(Image.open('/home/tsawicki/Desktop/cat_gray.jpg'))
im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

net.forward()

net.blobs['conv']

net.blobs['conv'].data[0,3]