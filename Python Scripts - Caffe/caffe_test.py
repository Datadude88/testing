import caffe
import matplotlib.pyplot as plt
import os
files = '/home/tsawicki/caffe2.7/caffe/examples/mnist/'
solver_path = files + 'lenet_solver.prototxt'

#NN_map = caffe.Net(NN_path,caffe.TRAIN)
os.chdir('/home/tsawicki/caffe2.7/caffe/')
solver = caffe.get_solver(solver_path)
train_loss = []
train_avg_loss = []
train_acc = []
train_acc_total = 0
train_acc = []
train_avg_acc = []

for i in range(1,5001):
    solver.step(1)
    train_loss += [solver.net.blobs['loss'].data.tolist()]
    
plt.plot(range(1,5001),train_loss)
plt.show()