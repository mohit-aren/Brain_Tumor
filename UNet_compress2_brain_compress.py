import numpy as np
import pandas as pd

import json
import sys
from PIL import Image, ImageOps

#from skimage.io import imread
#from matplotlib import pyplot as plt
import random

import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] ='mode=FAST_RUN,device=cpu'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

#from tensorflow.keras import models
from keras.optimizers import SGD
#from tensorflow.keras.layers import Input, ZeroPadding2D
#from tensorflow.keras.layers import Activation, Flatten, Reshape
#from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
#from tensorflow.keras import utils as  np_utils
#from keras.applications import imagenet_utils

path = 'results/'
img_w = 256
img_h = 256
n_labels = 4

Tumor = [255,255,255]
Tumor1 = [104,104,104]
Tumor2 = [198,198,198]
Unlabelled = [0,0,0]


n_train = 89
n_test = 6
n_val = 5

def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

def label_map1(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            #print(labels[r][c])
            if(labels[r][c] >= Tumor[0] -10 and labels[r][c] <= Tumor[0] + 10):
                label_map[r, c, 0] = 1
            elif(labels[r][c] >= Tumor1[0] -10 and labels[r][c] <= Tumor1[0] + 10):
                label_map[r, c, 1] = 1
            elif(labels[r][c] >= Tumor2[0] -10 and labels[r][c] <= Tumor2[0] + 10):
                label_map[r, c, 2] = 1
            elif(labels[r][c] >= Unlabelled[0] -10 and labels[r][c] <= Unlabelled[0] + 10):
                label_map[r, c, 3] = 1
    return label_map


import os
import imageio

def prep_data1(mode):
    data = []
    label = []
    
    folder_path = 'data' # path + mode

    #images_path = os.listdir(folder_path)
    images_path = os.listdir("data/images")

    if(mode == 'train'):
        n = 50
    elif(mode == 'val'):
        n = 45
    else:
        n = 10
    for index, image in enumerate(images_path):

        index += 1
        filename = os.path.join("data/images", image)
    
        print(index, filename)
        if(index > 50 and mode == 'train'):
            continue
        elif((index < 51 or index > 95) and mode =='val'):
            continue
        elif((index < 96 or index > 105) and mode == 'test'):
            continue
        
        #truth_file = filename.split('.png')
        #truth_file = filename.split('.png')
        
        tfile = os.path.join("data/GT", image.replace('_s', ''))
    
        
        print(tfile)
        if(filename == ""):
            break
        #img1 = Image.open(filename)
        img1 = Image.open(filename)
        '''
        w, h = img1.size
        start = (w-256)//2
        s_h = (h-256)//2
        
        new_im = img1.crop((start, s_h, start+256, s_h+256))
        '''
        new_size = tuple([240, 240])
        
        # create a new image and paste the resized on it
        
        #new_im = img1.resize((256, 256))
        new_im = Image.new('L', (256, 256))
        new_im.paste(img1, ((256-new_size[0])//2,
                            (256-new_size[1])//2))



        #img2 = Image.open(tfile)
        img2 = Image.open(tfile)
        '''
        #new_im1 = img2.resize((256, 256))
        new_im1 = img2.convert('RGB')
        new_im1 = new_im1.crop((start, s_h, start+256, s_h+256))
        '''
        #new_size = tuple([544, 512])
        
        # create a new image and paste the resized on it
        
        new_im1 = Image.new('L', (256, 256))
        new_im1.paste(img2, ((256-new_size[0])//2,
                            (256-new_size[1])//2))


        #index += 1
        # create a new image and paste the resized on it
        

        #img, gt = [imread(path + mode + '/' + filename + '.png')], imread(path + mode + '-colormap/' + filename + '.png')
        
        img, gt = [np.array(new_im,dtype=np.uint8)], np.array(new_im1,dtype=np.uint8)
        data.append(np.reshape(img,(256, 256,1)))
        label.append(label_map1(gt))
        sys.stdout.write('\r')
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    #print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048544, label.nbytes / 1048544))

    return data, label
    

def prep_data(mode):
    assert mode in {'test', 'train'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
    n = n_train if mode == 'train' else n_test
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(np.reshape(img,(256,256,1)))
        label.append(label_map(gt))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label

"""
def plot_results(output):
    gt = []
    df = pd.read_csv(path + 'test.csv')
    for i, item in df.iterrows():
        gt.append(np.clip(imread(path + item[1]), 0, 1))

    plt.figure(figsize=(15, 2 * n_test))
    for i, item in df.iterrows():
        plt.subplot(n_test, 4, 4 * i + 1)
        plt.title('Ground Truth')
        plt.axis('off')
        gt = imread(path + item[1])
        plt.imshow(np.clip(gt, 0, 1))

        plt.subplot(n_test, 4, 4 * i + 2)
        plt.title('Prediction')
        plt.axis('off')
        labeled = np.argmax(output[i], axis=-1)
        plt.imshow(labeled)

        plt.subplot(n_test, 4, 4 * i + 3)
        plt.title('Heat map')
        plt.axis('off')
        plt.imshow(output[i][:, :, 1])

        plt.subplot(n_test, 4, 4 * i + 4)
        plt.title('Comparison')
        plt.axis('off')
        rgb = np.empty((img_h, img_w, 3))
        rgb[:, :, 0] = labeled
        rgb[:, :, 1] = imread(path + item[0])
        rgb[:, :, 2] = gt
        plt.imshow(rgb)

    plt.savefig('result.png')
    plt.show()
"""

#########################################################################################################
from keras.layers import Input
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


#import skimage.io as io
#import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

def get_unet():
        inputs = Input((256,256, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(100, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(100, 1, activation='softmax')(conv9)
        conv10 = Reshape((256*256, 4))(conv10)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        return model


"""
with open('model_5l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())
"""

def get_unet_compress(layer1_filters,layer2_filters,layer3_filters,layer4_filters,layer5_filters):
        inputs = Input((256,256, 1))

        conv1 = Conv2D(layer1_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        conv1 = Conv2D(layer1_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(layer2_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(layer2_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(layer3_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(layer3_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(layer4_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(layer4_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(layer5_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(layer5_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(layer4_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], axis=3)
        conv6 = Conv2D(layer4_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(layer4_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(layer3_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], axis=3)
        conv7 = Conv2D(layer3_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(layer3_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(layer2_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], axis=3)
        conv8 = Conv2D(layer2_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(layer2_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(layer1_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], axis=3)
        conv9 = Conv2D(layer1_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(layer1_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(100, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(100, 1, activation='softmax')(conv9)
        conv10 = Reshape((256*256, 4))(conv10)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        return model


"""
with open('model_5l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())
"""

#autoencoder = get_unet()

print('Start')
#optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
#autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
layer1_filters = 64
layer2_filters = 128
layer3_filters = 256
layer4_filters = 512
layer5_filters = 1024

nlayer1_filters = 64
nlayer2_filters = 128
nlayer3_filters = 256
nlayer4_filters = 512
nlayer5_filters = 1024

#autoencoder = get_unet_compress(layer1_filters,layer2_filters,layer3_filters,layer4_filters,layer5_filters)


print( 'Compiled: OK')
#autoencoder.summary()

# Train model or load weights

train_data, train_label = prep_data1('train')
val_data, val_label = prep_data1('val')
nb_epoch = 20
batch_size = 2
#history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(val_data, val_label))
#autoencoder.save_weights('model_5l_weight_leaves_unet.1.hdf5')

#autoencoder.load_weights('model_5l_weight_ep50.hdf5')


#test_data, test_label = prep_data1('test')
test_data, test_label = val_data, val_label

def enure_binary(x):
    y = []
    for indx in range(0, len(x)):
        if(x[indx] < 0.5):
            y.append(0)
        else:
            y.append(1)
            
    return y
            
            
num_dimensions = 4096
        # print final results
class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            #print(num_dimensions)
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                

for withtrain in range(0,20):
    
    layer1_filters = nlayer1_filters
    layer2_filters = nlayer2_filters
    layer3_filters = nlayer3_filters
    layer4_filters = nlayer4_filters
    layer5_filters = nlayer5_filters

    autoencoder = get_unet_compress(layer1_filters,layer2_filters,layer3_filters,layer4_filters,layer5_filters)
    
    wt = 'UNet1_' + str(withtrain) + '.h5'
    
    autoencoder.load_weights(wt)
    score = autoencoder.evaluate(test_data, test_label, verbose=1)
    print( 'Test score:', score[0])
    print( 'Test accuracy:', score[1])
    
    ####################### 1st convolution layer with 64 filters
    print('1st convolution layer with 64 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[1].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    filters2, biases2 = autoencoder.layers[2].get_weights()
    filters3 = np.copy(filters)
    biases3 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,layer1_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[1].set_weights([filters1, biases1])


        filters3 = np.copy(filters2)
        biases3 = np.copy(biases2)
        
        for i in range(0,layer1_filters):
            if(x[i] < 0.5):
                biases3[i] = 0
                filters3[:, :, :, i] = 0
        
        autoencoder.layers[2].set_weights([filters3, biases3])

        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, layer1_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(layer1_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, layer1_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)


    A1 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    nlayer1_filters = new_num
    
    ####################### 1st convolution layer with 64 filters
    print('2nd convolution layer with 128 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[4].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    filters2, biases2 = autoencoder.layers[5].get_weights()
    filters3 = np.copy(filters)
    biases3 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,layer2_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[4].set_weights([filters1, biases1])


        filters3 = np.copy(filters2)
        biases3 = np.copy(biases2)
        
        for i in range(0,layer2_filters):
            if(x[i] < 0.5):
                biases3[i] = 0
                filters3[:, :, :, i] = 0
        
        autoencoder.layers[5].set_weights([filters3, biases3])

        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, layer2_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(layer2_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, layer2_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)

    A2 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    nlayer2_filters = new_num
    
 
    ####################### 1st convolution layer with 64 filters
    print('3rd convolution layer with 256 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[7].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    filters2, biases2 = autoencoder.layers[8].get_weights()
    filters3 = np.copy(filters)
    biases3 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,layer3_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[7].set_weights([filters1, biases1])


        filters3 = np.copy(filters2)
        biases3 = np.copy(biases2)
        
        for i in range(0,layer3_filters):
            if(x[i] < 0.5):
                biases3[i] = 0
                filters3[:, :, :, i] = 0
        
        autoencoder.layers[8].set_weights([filters3, biases3])

        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, layer3_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(layer3_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, layer3_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)

    A3 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    nlayer3_filters = new_num
    
    ####################### 1st convolution layer with 64 filters
    print('4th convolution layer with 512 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[10].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    filters2, biases2 = autoencoder.layers[11].get_weights()
    filters3 = np.copy(filters)
    biases3 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,layer4_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[10].set_weights([filters1, biases1])


        filters3 = np.copy(filters2)
        biases3 = np.copy(biases2)
        
        for i in range(0,layer4_filters):
            if(x[i] < 0.5):
                biases3[i] = 0
                filters3[:, :, :, i] = 0
        
        autoencoder.layers[11].set_weights([filters3, biases3])

        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, layer4_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(layer4_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, layer4_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)

    A4 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    nlayer4_filters = new_num
    
     
    ####################### 1st convolution layer with 64 filters
    print('5th convolution layer with 1024 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[14].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    filters2, biases2 = autoencoder.layers[15].get_weights()
    filters3 = np.copy(filters)
    biases3 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,layer5_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[14].set_weights([filters1, biases1])


        filters3 = np.copy(filters2)
        biases3 = np.copy(biases2)
        
        for i in range(0,layer5_filters):
            if(x[i] < 0.5):
                biases3[i] = 0
                filters3[:, :, :, i] = 0
        
        autoencoder.layers[15].set_weights([filters3, biases3])

        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, layer5_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(layer5_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, layer5_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)

    A5 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    nlayer5_filters = new_num
    
    
    #################Compression##############3
    
    autoencoder_compress = get_unet_compress(nlayer1_filters,nlayer2_filters,nlayer3_filters,nlayer4_filters,nlayer5_filters)

    print('Start')
    #optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    #autoencoder_compress.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print( 'Compiled: OK')
    autoencoder_compress.summary()
    
    for k in range(1, 2):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer1_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer1_filters):
            if(A1[j] == 1) :
                filters1[:, :, :, index1] = filters[:, :, :, j]
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
   
    for k in range(2, 3):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer1_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer1_filters):
            if(A1[j] == 1) :
                index2 = 0
                for l in range(layer1_filters):
                    if(A1[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(4, 5):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer2_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer2_filters):
            if(A2[j] == 1) :
                index2 = 0
                for l in range(layer1_filters):
                    if(A1[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(5, 6):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer2_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer2_filters):
            if(A2[j] == 1) :
                index2 = 0
                for l in range(layer2_filters):
                    if(A2[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(7, 8):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer3_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer3_filters):
            if(A3[j] == 1) :
                index2 = 0
                for l in range(layer2_filters):
                    if(A2[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(8, 9):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer3_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer3_filters):
            if(A3[j] == 1) :
                index2 = 0
                for l in range(layer3_filters):
                    if(A3[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(10, 11):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer4_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer4_filters):
            if(A4[j] == 1) :
                index2 = 0
                for l in range(layer3_filters):
                    if(A3[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(11, 12):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer4_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer4_filters):
            if(A4[j] == 1) :
                index2 = 0
                for l in range(layer4_filters):
                    if(A4[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    

    for k in range(14, 15):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer5_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer5_filters):
            if(A5[j] == 1) :
                index2 = 0
                for l in range(layer4_filters):
                    if(A4[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(15, 16):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer5_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer5_filters):
            if(A5[j] == 1) :
                index2 = 0
                for l in range(layer5_filters):
                    if(A5[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
   
    for k in range(18, 19):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer4_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer4_filters):
            if(A4[j] == 1) :
                index2 = 0
                for l in range(layer5_filters):
                    if(A5[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(20, 22):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer4_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer4_filters):
            if(A4[j] == 1) :
                index2 = 0
                for l in range(layer4_filters):
                    if(A4[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(23, 24):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer3_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer3_filters):
            if(A3[j] == 1) :
                index2 = 0
                for l in range(layer4_filters):
                    if(A4[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(25, 27):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer3_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer3_filters):
            if(A3[j] == 1) :
                index2 = 0
                for l in range(layer3_filters):
                    if(A3[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(28, 29):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer2_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer2_filters):
            if(A2[j] == 1) :
                index2 = 0
                for l in range(layer3_filters):
                    if(A3[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(30, 32):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer2_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer2_filters):
            if(A2[j] == 1) :
                index2 = 0
                for l in range(layer2_filters):
                    if(A2[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])

    for k in range(33, 34):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer1_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer1_filters):
            if(A1[j] == 1) :
                index2 = 0
                for l in range(layer2_filters):
                    if(A2[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(35, 37):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = layer1_filters, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(layer1_filters):
            if(A1[j] == 1) :
                index2 = 0
                for l in range(layer1_filters):
                    if(A1[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
        
    history = autoencoder_compress.fit(train_data, train_label, batch_size=1, nb_epoch=1000, verbose=1, validation_data=(val_data, val_label))
    
    wt = 'UNet1_' + str(withtrain+1) + '.h5'

    autoencoder_compress.save_weights(wt)
    
    # Model visualization
    #from keras.utils.visualize_util import plot
    #plot(autoencoder, to_file='model.JPG', show_shapes=True)
    
    #test_data, test_label = prep_data1('test')
    score = autoencoder_compress.evaluate(test_data, test_label, verbose=1)
    print( 'Test score:', score[0])
    print( 'Test accuracy:', score[1])
