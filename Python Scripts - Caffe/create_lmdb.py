import numpy as np
import caffe
import pandas as pd
import lmdb
from skimage import color, io

DATA_PREFIX='/home/tsawicki/Documents/Python Input/Out/'
DATA_FILE_TRAIN= DATA_PREFIX + '3Views_train.csv'
DATA_FILE_VAL = DATA_PREFIX + '3Views_val.csv'

fileread = pd.read_csv(DATA_FILE_TRAIN, sep=',', header=None)
fileread2 = pd.read_csv(DATA_FILE_VAL, sep=',', header=None)
train = fileread[(fileread[2]==0) | (fileread[2]==2) | (fileread[2]==4)].reset_index(drop=True)
val = fileread2[(fileread2[2]==0) | (fileread2[2]==2) | (fileread2[2]==4)].reset_index(drop=True)

numTrainSamples = len(train)
numTestSamples = len(val)
IMAGE_SIZE = 216

xTrain = np.zeros(shape = [numTrainSamples, 1, IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8)
yTrain = np.zeros(shape = [numTrainSamples])
xTest = np.zeros(shape = [numTestSamples, 1, IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8)
yTest = np.zeros(shape = [numTestSamples])

for i in range(numTrainSamples):
    xtemp = io.imread(DATA_PREFIX + train[1][i] + '.jpg')
    ytemp = train[2][i]
    if len(xtemp.shape) == 3:
        temp2 = color.rgb2gray(xtemp)
    else:
        temp2 = xtemp.astype('float32')
        temp2 /= 255.0
    xTrain[i,0,:,:] = temp2 
    if ytemp > 0:
        yTrain[i] = ytemp
        
for i in range(numTestSamples):
    xtemp = io.imread(DATA_PREFIX + val[1][i] + '.jpg')
    ytemp = val[2][i]
    if len(xtemp.shape) == 3:
        temp2 = color.rgb2gray(xtemp)
    else:
        temp2 = xtemp.astype('float32')
        temp2 /= 255.0
    xTest[i,0,:,:] = temp2 
    if ytemp > 0:
        yTest[i] = ytemp
    
map_size1 = xTrain.nbytes * 10
map_size2 = xTest.nbytes * 10
    
env1 = lmdb.open(DATA_PREFIX + '../train_lmdb', map_size=map_size1)
with env1.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(numTrainSamples):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = xTrain.shape[1]
        datum.height = xTrain.shape[2]
        datum.width = xTrain.shape[3]
        datum.data = xTrain[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(yTrain[i])
        str_id = '{:0>8d}'.format(i)
        txn.put(str_id, datum.SerializeToString())
        
env2 = lmdb.open(DATA_PREFIX + '../test_lmdb', map_size=map_size2)
with env2.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(numTestSamples):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = xTest.shape[1]
        datum.height = xTest.shape[2]
        datum.width = xTest.shape[3]
        datum.data = xTest[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(yTest[i])
        str_id = '{:0>8d}'.format(i)
        txn.put(str_id, datum.SerializeToString())



#------------- reading a lmdb file ------------------
env = lmdb.open(DATA_PREFIX + '../test_lmdb', readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')
    
datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label

#env = lmdb.open(DATA_PREFIX + '../test_lmdb', readonly=True)
#with env.begin() as txn:
#    cursor = txn.cursor()
#    for key, value in cursor:
#        print(key, value)
        
        
       





 
#second method        
lmdb_env = lmdb.open(DATA_PREFIX + '../test_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    full_data = caffe.io.datum_to_array(datum)
    for l, d in zip(label, full_data):
            print l, d       
    