import os
os.environ['GLOG_minloglevel'] = '2'

import caffe
import numpy as np


def Decompose(Mi, r, n, kw, kh):
    U, sigma, VT = np.linalg.svd(Mi, full_matrices=False)
    Ur = U[:, :r]
    Vr = VT[:r, :]
    Sr = np.eye(r) * sigma[:r]
    D = np.reshape(Vr, [r, 1, kw, kh])
    S_t = np.dot(Ur, Sr)
    S = np.reshape(S_t, [n, r, 1, 1])
    return D, S


def single_conv(T):
    r = 5
    n, c, kw, kh = T.shape
    list_d = np.zeros([c, r, 1, kw, kh])
    list_s = np.zeros([n, r, c, 1, 1])

    for i in range(c):
        Ti = T[:, i, :, :]
        Mi = np.reshape(Ti, [n, kw * kh])
        Di, Si = Decompose(Mi, r, n, kw, kh)
        list_d[i, :, :, :, :] = Di
        list_s[:, :, i, :, :] = Si

    Td = np.reshape(list_d, [r*c, 1, kw, kh])
    Ts = np.reshape(list_s, [n, r*c, 1, 1])
    return Td, Ts


def main():
    model_path = '/home/sjt/SJT/ZTE/Fighting/models/clear_idle_filters_delconv5_svd-130/'
    Output_path = '/home/sjt/SJT/ZTE/Fighting/models/DAC01/'

    net = caffe.Net(model_path + "TestModel.prototxt",
                    model_path + "TestModel.caffemodel", caffe.TEST)

    weight, bias = net.params['conv4_5_2']
    weight = weight.data

    Td, Ts = single_conv(weight)

    net2 = caffe.Net(Output_path + "TestModel.prototxt", caffe.TEST)
    for key in net2.params.keys():
        if key == 'depth_conv4_5_2':
            net2.params[key][0].data[...] = Td
        elif key == 'point_conv4_5_2':
            print(net2.params[key][0].data[...].shape)
            net2.params[key][0].data[...] = Ts
            net2.params[key][1].data[...] = bias.data
        else:
            net2.params[key][0].data[...] = net.params[key][0].data
            net2.params[key][1].data[...] = net.params[key][1].data

    net2.save(Output_path + "TestModel.caffemodel")
    print('New model has been saved in ./fc_svd')


if __name__ == '__main__':
    main()

