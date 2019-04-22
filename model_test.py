# To test the difference between the new model and the origin model

import caffe
import numpy as np
import os

image_root = 'images'
test_image = ['{}.jpeg'.format(i) for i in range(4)]


def test_t(net, input_image="images/cat.png", output_tensor="fc5_"):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    
    image = caffe.io.load_image(input_image)
    transformed_image = transformer.preprocess('data', (image * 255 - 127.5) * 0.0078125 )
    net.blobs['data'].reshape(1, 3, 128, 128)
    net.blobs['data'].data[...] = transformed_image

    output = net.forward()
    return output[output_tensor].reshape(-1)


def cosine_dist(x, y):
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))


net = caffe.Net("./models/origin/TestModel.prototxt", "./models/origin/TestModel.caffemodel", caffe.TEST)
net2 = caffe.Net("./models/clear_idle_filters_delconv5_svd-130/TestModel.prototxt", "./models/clear_idle_filters_delconv5_svd-130/TestModel.caffemodel", caffe.TEST)

for img in test_image:
    output1 = test_t(net, os.path.join(image_root, img))
    output2 = test_t(net2, os.path.join(image_root, img))
    print(cosine_dist(output1, output2))
