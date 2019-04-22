import caffe
import numpy as np

# r = 240
r = 130

# net = caffe.Net("./models/no_bn/TestModel.prototxt", "./models/no_bn/TestModel.caffemodel", caffe.TEST)
net = caffe.Net("./models/clear_idle_filters_delconv5/TestModel.prototxt", "./models/clear_idle_filters_delconv5/TestModel.caffemodel", caffe.TEST)

weight, bias = net.params['fc5_']
# weight, bias = net.params['conv5_1_1b']

U, sigma, VT = np.linalg.svd(weight.data, full_matrices=False)

sigma[:r].sum()/sigma.sum()

# net2 = caffe.Net("./models/fc_svd/TestModel.prototxt", caffe.TEST)
net2 = caffe.Net("./models/clear_idle_filters_delconv5_svd-130/TestModel.prototxt", caffe.TEST)


for key in net2.params.keys():
    if key == 'fc5_1':
        net2.params[key][0].data[...] = np.dot(np.eye(r) * sigma[:r], VT[:r])
    elif key =='fc5_':
        net2.params[key][0].data[...] = U[:, :r]
        net2.params[key][1].data[...] = bias.data
    else:
        net2.params[key][0].data[...] = net.params[key][0].data
        net2.params[key][1].data[...] = net.params[key][1].data

# net2.save("./models/fc_svd/TestModel.caffemodel")
net2.save("./models/clear_idle_filters_delconv5_svd-130/TestModel.caffemodel")
print('New model has been saved in ./fc_svd')
