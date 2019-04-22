# Delete the conv5_3 -- conv5_6
import caffe


BEFORE_MODIFY_DEPLOY_NET = "./models/clear_idle_filters01/TestModel.prototxt"
AFTER_MODIFY_DEPLOY_NET = "./models/clear_idle_filters_delconv5/TestModel.prototxt"
BEFORE_MODIFY_CAFFEMODEL = "./models/clear_idle_filters01/TestModel.caffemodel"
AFTER_MODIFY_CAFFEMODEL = "./models/clear_idle_filters_delconv5/TestModel.caffemodel"

# 根据prototxt修改caffemodel
net = caffe.Net(AFTER_MODIFY_DEPLOY_NET, BEFORE_MODIFY_CAFFEMODEL, caffe.TEST)
net.save(AFTER_MODIFY_CAFFEMODEL)

print('Successful!')


