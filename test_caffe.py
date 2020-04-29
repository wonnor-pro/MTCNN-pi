import caffe

imglistfile = "./imglist.txt"
minsize = 20
caffe_model_path = "./model"
threshold = [0.6, 0.7, 0.7]
factor = 0.709
    
caffe.set_mode_cpu()
# PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)



print("so far good")
