import caffe
import numpy as np

net = caffe.Net("./models/origin/TestModel.prototxt", "./models/origin/TestModel.caffemodel", caffe.TEST)

conv_with_bn = ['conv1_1_1', 'conv1_2_1', 'conv1_2_2', 'conv1_3_1']

new_params = {}
for conv_name in conv_with_bn:
    weight, bias = net.params[conv_name]
    weight = weight.data
    bias = bias.data
    channels = weight.shape[0]
    
    mean, var, scalef = net.params[conv_name + "_bn"]
    mean = mean.data
    var = var.data
    scalef = scalef.data
    
    scales, shift = net.params[conv_name + "_scale"]
    scales = scales.data
    shift = shift.data
    
    if scalef != 0:
        scalef = 1. / scalef
    mean *= scalef
    var *= scalef
    rstd = 1. / np.sqrt(var + 1e-5)
    
    new_weight = weight * rstd.reshape((channels,1,1,1)) * scales.reshape((channels, 1, 1, 1))
    new_bias = (bias - mean) * rstd * scales + shift
    
    new_params[conv_name] = new_weight, new_bias.reshape(-1)

net2 = caffe.Net("./models/no_bn/TestModel.prototxt", caffe.TEST)

for key in net2.params.keys():
    if key in new_params:
        net2.params[key][0].data[...] = new_params[key][0]
        net2.params[key][1].data[...] = new_params[key][1]
    else:
        net2.params[key][0].data[...] = net.params[key][0].data
        net2.params[key][1].data[...] = net.params[key][1].data

net2.save("./models/no_bn/TestModel.caffemodel")
print('New model has been saved in ./no_bn')
