import caffe
import numpy as np

net = caffe.Net("./models/no_bn/TestModel.prototxt", "./models/no_bn/TestModel.caffemodel", caffe.TEST)

class Pruner(object):
    def __init__(self, net):
        self._net = net
        self.conv_data = {}
        
    def _prune(self, conv_param, del_kernels=None, not_del_filters=False):
        weight, bias = conv_param
        weight = weight.data
        bias = bias.data
        origin_channels = weight.shape[0]
        
        # delete filters
        if not not_del_filters:
            abs_mean = np.abs(weight).mean(axis=(1,2,3))
            del_filters = np.where(abs_mean < 1e-10)[0]
            weight = np.delete(weight, del_filters, axis=0)
            bias = np.delete(bias, del_filters, axis=0)
        else:
            del_filters = np.array([])
        
        # delete kernels
        if del_kernels is not None:
            weight = np.delete(weight, del_kernels, axis=1)

        try:
            # print(del_filters)
            print(len(del_filters))
        except:
            # print(del_filters, 0)
            print(0)
        try:
            # print(del_kernels)
            print(len(del_kernels))
            print('**********')
        except:
            # print(del_kernels, 0)
            print(0)
            print('**********')
            
        return weight, bias, del_filters, origin_channels
    
    def prune_conv(self, name, bottom=None):
        if bottom is None:
            self.conv_data[name] = self._prune(self._net.params[name])
        else:
            self.conv_data[name] = self._prune(self._net.params[name], self.conv_data[bottom][2])
            
    def prune_concat(self, name, bottoms):
        offsets = [0] + [self.conv_data[b][3] for b in bottoms]
        for i in range(1, len(offsets)):
            offsets[i] += offsets[i-1]
        del_filters = [self.conv_data[b][2] + offsets[i] for i, b in enumerate(bottoms)]
        del_filters_new = np.concatenate(del_filters)
        self.conv_data[name] = self._prune(self._net.params[name], del_filters_new, not_del_filters=True)
        
    def save(self, new_model, output_weights):
        net2 = caffe.Net(new_model, caffe.TEST)
        for key in net2.params.keys():
            if key in self.conv_data:
                net2.params[key][0].data[...] = self.conv_data[key][0]
                net2.params[key][1].data[...] = self.conv_data[key][1]
            else:
                net2.params[key][0].data[...] = net.params[key][0].data
                net2.params[key][1].data[...] = net.params[key][1].data
        net2.save(output_weights)


pruner = Pruner(net)

pruner.prune_conv("conv1_1_1")
pruner.prune_conv("conv1_2_1")
pruner.prune_conv("conv1_2_2", "conv1_2_1")
pruner.prune_conv("conv1_3_1")
pruner.prune_conv("conv1_3_2", "conv1_3_1")
pruner.prune_conv("conv1_3_3", "conv1_3_2")

pruner.prune_concat("conv2_1", ("conv1_1_1", "conv1_2_2", "conv1_3_3"))
pruner.prune_conv("conv2_2", "conv2_1")
pruner.prune_conv("conv2_3", "conv2_2")
pruner.prune_conv("conv2_4", "conv2_3")
pruner.prune_conv("conv2_5", "conv2_4")
pruner.prune_conv("conv2_6", "conv2_5")
pruner.prune_conv("conv2_7", "conv2_6")
pruner.prune_conv("conv2_8", "conv2_7")

print("^^^^^^^^^^^^^Conv3^^^^^^^^^^^^^^^^")
pruner.prune_concat("conv3_1_1", ("conv2_2", "conv2_4", "conv2_6", "conv2_8"))
pruner.prune_concat("conv3_1_1b", ("conv2_2", "conv2_4", "conv2_6", "conv2_8"))
# pruner.prune_conv("conv3_1_2", "conv3_1_1")
# # pruner.prune_concat("conv3_2_1", ("conv3_1_2", "conv3_1_1b"))
# pruner.prune_conv("conv3_2_2", "conv3_2_1")
# # pruner.prune_concat("conv3_3_1", ("conv3_2_2", "conv3_2_1"))
# pruner.prune_conv("conv3_3_2", "conv3_3_1")
# # pruner.prune_concat("conv3_4_1", ("conv3_2_1", "conv3_2_2", "conv3_3_2"))
# pruner.prune_conv("conv3_4_2", "conv3_4_1")
# # pruner.prune_concat("conv3_5_1", ("conv3_4_2", "conv3_4_1"))
# pruner.prune_conv("conv3_5_2", "conv3_5_1")
# # pruner.prune_concat("conv3_6_1", ("conv3_5_2", "conv3_5_1"))
# pruner.prune_conv("conv3_6_2", "conv3_6_1")

# print("^^^^^^^^^^^^^Conv4^^^^^^^^^^^^^^^^")
# pruner.prune_concat("conv4_1_1", ("conv3_3_1", "conv3_5_1", "conv3_6_2", "conv3_6_1"))
# pruner.prune_concat("conv4_1_1b", ("conv3_3_1", "conv3_5_1", "conv3_6_2", "conv3_6_1"))
# pruner.prune_conv("conv4_1_2", "conv4_1_1")
# # pruner.prune_concat("conv4_2_1", ("conv4_1_2", "conv4_1_1b"))
# pruner.prune_conv("conv4_2_2", "conv4_2_1")
# # pruner.prune_concat("conv4_3_1", ("conv4_2_2", "conv4_2_1"))
# pruner.prune_conv("conv4_3_2", "conv4_3_1")
# # pruner.prune_concat("conv4_4_1", ("conv4_2_1", "conv4_2_2", "conv4_3_2"))
# print("^^^^^^^^^^********************^^^^^^^^^^^")
# pruner.prune_conv("conv4_4_2", "conv4_4_1")
# # pruner.prune_concat("conv4_5_1", ("conv4_4_2", "conv4_4_1"))
# pruner.prune_conv("conv4_5_2", "conv4_5_1")
# # pruner.prune_concat("conv4_6_1", ("conv4_5_2", "conv4_5_1"))
# pruner.prune_conv("conv4_6_2", "conv4_6_1")
#
# print("^^^^^^^^^^^^^Conv5^^^^^^^^^^^^^^^^")
# pruner.prune_concat("conv5_1_1", ("conv4_3_1", "conv4_5_1", "conv4_6_2", "conv4_6_1"))
# # pruner.prune_concat("conv5_1_1b", ("conv4_3_1", "conv4_5_1", "conv4_6_2", "conv4_6_1"))
# pruner.prune_conv("conv5_1_2", "conv5_1_1")
# # pruner.prune_concat("conv5_2_1", ("conv5_1_2", "conv5_1_1b"))
# pruner.prune_conv("conv5_2_2", "conv5_2_1")
# print("Warning!")
# pruner.prune_concat("fc5_1", ("conv5_2_2", "conv5_2_1", "conv5_2_2", "conv5_2_1"))

[(k, v[0].shape[0]) for k, v in pruner.conv_data.items() if v[0] is not None]


# You should modify the number of channels in new prototxt before save
pruner.save("./models/clear_idle_filters01/TestModel.prototxt", "./models/clear_idle_filters01/TestModel.caffemodel")
print('New prototxt and model has been saved in ./clear_idle_filters')

