
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io as sio
import numpy as np
import time

def myconv2d(x,w,dilation):
    accRateX = dilation[0]
    accRateY = dilation[1]
    [sizeX1,sizeX2,sizeX3] = np.shape(x) 
    [sizeW1,sizeW2,sizeW3,sizeW4] = np.shape(w) 
    numItX = sizeX1 - ((sizeW1-1)*accRateX)
    numItY = sizeX2 - ((sizeW2-1)*accRateY)
    output = np.zeros([numItX, numItY, sizeW4])
    for k in range(sizeW4):
        for i in range(numItX):
            for j in range(numItY):
                            xkernelend = i+((sizeW1-1)*accRateX)
                            ykernelend = j+((sizeW2-1)*accRateY)
                            kernelX = x[i:xkernelend+1:accRateX,j:ykernelend+1:accRateY,:]
                            summation = sum(sum(sum(np.multiply(kernelX,w[:,:,:,k]))))  
                            output[i,j,k] = output[i,j,k] + summation                
    return output                        
        
def ReLU(x):
    return np.where(x<0, 0, x) 

def learning(autocalibration,recArea,accRate,maximumIteration,learningRate,sizeW1,sizeW2,sizeW3):
    config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session(config=config)
    
    [autocalibrationX, autocalibrationY, autocalibrationZ] = np.shape(autocalibration)
    autocalibrationLearning = autocalibration
    autocalibrationLearning = np.reshape(autocalibrationLearning,[1,autocalibrationX, autocalibrationY, autocalibrationZ])
    [recArea_0,recAreaX,recAreaY,recAreaZ] = np.shape(recArea)
    
    [b1x, b1y, autocalibrationZ, N1] = sizeW1
    [b2x, b2y, N1, N2] = sizeW2
    [b3x, b3y, N2, target_Z] = sizeW3
    
    inputAutocalibration = tf.placeholder(tf.float32, [1, autocalibrationX, autocalibrationY, autocalibrationZ])                                  
    inputRecArea = tf.placeholder(tf.float32, [1, recAreaX,recAreaY,recAreaZ])         
    
    Input = tf.reshape(inputAutocalibration, [1, autocalibrationX, autocalibrationY, autocalibrationZ])         

    initW1 = tf.truncated_normal(sizeW1, stddev=0.1,dtype=tf.float32)
    Wconv1 = tf.Variable(initW1 ,name = 'W1') 
    lconv1 = tf.nn.convolution(Input, Wconv1,padding='VALID',dilation_rate = [1,accRate])
    hconv1 = tf.nn.relu(lconv1) 

    initW2 = tf.truncated_normal(sizeW2, stddev=0.1,dtype=tf.float32)
    Wconv2 = tf.Variable(initW2 ,name = 'W2')
    lconv2 = tf.nn.convolution(hconv1, Wconv2,padding='VALID',dilation_rate = [1,accRate])
    hconv2 = tf.nn.relu(lconv2)

    initW3 = tf.truncated_normal(sizeW3, stddev=0.1,dtype=tf.float32)
    Wconv3 = tf.Variable(initW3 ,name = 'W3')
    hconv3 = tf.nn.convolution(hconv2, Wconv3,padding='VALID',dilation_rate = [1,accRate])

    error_norm = tf.norm(inputRecArea - hconv3)     
    train_step = tf.train.AdamOptimizer(learningRate).minimize(error_norm)

    init = tf.global_variables_initializer()
    sess.run(init)

    print('Initial Norm Error',sess.run(error_norm,feed_dict={inputAutocalibration: autocalibrationLearning, inputRecArea: recArea})    )
    for i in range(maximumIteration):
        sess.run(train_step, feed_dict={inputAutocalibration: autocalibrationLearning, inputRecArea: recArea})

    error = sess.run(error_norm,feed_dict={inputAutocalibration: autocalibrationLearning, inputRecArea: recArea})
    w1 = sess.run(Wconv1)
    w2 = sess.run(Wconv2)
    w3 = sess.run(Wconv3)
    sess.close()
    return [w1,w2,w3,error]  

                       
maximumIteration = 1000
learningRate = 3e-3

RAKI_PART1 = sio.loadmat('RAKI_PART1.mat')

b1x= int(np.squeeze(RAKI_PART1['b1x']))
b1y= int(np.squeeze(RAKI_PART1['b1y']))

b2x= int(np.squeeze(RAKI_PART1['b2x']))
b2y= int(np.squeeze(RAKI_PART1['b2y']))

b3x= int(np.squeeze(RAKI_PART1['b3x']))
b3y= int(np.squeeze(RAKI_PART1['b3y']))

N1 = int(np.squeeze(RAKI_PART1['N1']))
N2 = int(np.squeeze(RAKI_PART1['N2']))

accRate = int(np.squeeze(RAKI_PART1['accRate']))

autocalibration = RAKI_PART1['autocalibration']
kspace = RAKI_PART1['kspace']
[Xdim,Ydim,numCoils] = np.shape(kspace)
autocalibration = np.float32(autocalibration)  

[autocalibrationX, autocalibrationY, autocalibrationZ] = np.shape(autocalibration)
numRealCoils = 2*numCoils

w1_allchannels = np.zeros([b1x, b1y, numRealCoils, N1, numRealCoils],dtype=np.float32)
w2_allchannels = np.zeros([b2x, b2y, N1, N2, numRealCoils],dtype=np.float32)
w3_allchannels = np.zeros([b3x, b3y, N2, accRate - 1, numRealCoils],dtype=np.float32) 

sizeW1 = [b1x, b1y, numRealCoils, N1]
sizeW2 = [b2x, b2y, N1, N2]
sizeW3 = [b3x, b3y, N2, accRate - 1]

learning_start = time.time() 
recAreaArr = RAKI_PART1['recAreaArr']
for i in range(autocalibrationZ):
    print('Coil #',i+1)
    recArea = recAreaArr[i,:,:,:]
    [recAreaX,recAreaY,recAreaZ] = np.shape(recArea)
    recArea = np.reshape(recArea,[1,recAreaX,recAreaY,recAreaZ])

    [w1,w2,w3,error]=learning(autocalibration,recArea,accRate,maximumIteration,learningRate,sizeW1,sizeW2,sizeW3)
    w1_allchannels[:,:,:,:,i] = w1
    w2_allchannels[:,:,:,:,i] = w2
    w3_allchannels[:,:,:,:,i] = w3                               
    time_channel_end = time.time()
    print('Norm Error = ',error)
    print('')

sio.savemat('weights.mat',{'w1_allchannels':w1_allchannels,'w2_allchannels':w2_allchannels,'w3_allchannels':w3_allchannels})
learning_end = time.time();
print('Learning Time:',(learning_end - learning_start),'seconds')

