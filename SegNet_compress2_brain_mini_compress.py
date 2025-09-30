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
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.layers import Input, ZeroPadding2D
#from tensorflow.keras.layers import Activation, Flatten, Reshape
#from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate  as merge
#from tensorflow.keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose

#from tensorflow.keras import utils as  np_utils
#from keras.applications import imagenet_utils

path = 'results/'
img_w = 240
img_h = 240
n_labels = 4

Tumor = [255,255,255]
Tumor1 = [104,104,104]
Tumor2 = [198,198,198]
Unlabelled = [0,0,0]


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]


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
        start = (w-240)//2
        s_h = (h-240)//2
        
        new_im = img1.crop((start, s_h, start+240, s_h+240))
        '''
        #new_size = tuple([544, 512])
        
        # create a new image and paste the resized on it
        
        #new_im = img1.resize((240, 240))
        #new_im = Image.new("RGB", (544, 512))
        #new_im.paste(img1, ((544-new_size[0])//2,
                            #(512-new_size[1])//2))



        #img2 = Image.open(tfile)
        img2 = Image.open(tfile)
        '''
        #new_im1 = img2.resize((240, 240))
        new_im1 = img2.convert('RGB')
        new_im1 = new_im1.crop((start, s_h, start+240, s_h+240))
        '''
        #new_size = tuple([544, 512])
        
        # create a new image and paste the resized on it
        
        #new_im1 = Image.new("RGB", (544, 512))
        #new_im1.paste(img2, ((544-new_size[0])//2,
                            #(512-new_size[1])//2))


        #index += 1
        # create a new image and paste the resized on it
        

        #img, gt = [imread(path + mode + '/' + filename + '.png')], imread(path + mode + '-colormap/' + filename + '.png')
        
        img, gt = [np.array(img1,dtype=np.uint8)], np.array(img2,dtype=np.uint8)
        data.append(np.reshape(img,(240, 240,1)))
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

    plt.savefig('result.JPG')
    plt.show()
"""

#########################################################################################################
"""
def SegNet(input_shape=(480, 384, 3), classes=13):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(classes, 1, 1, border_mode="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model
"""

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


def SegNet(input_shape=(240, 240, 1), classes=4):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(classes, 1, 1, padding="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model

def SegNet_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters, input_shape=(240, 240, 1), classes=4):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(layer1_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(layer2_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(layer3_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(layer4_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(layer5_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(layer6_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(layer7_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(layer8_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(classes, 1, 1, padding="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model

"""
with open('model_5l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())
"""

autoencoder = SegNet()

print('Start')
optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print( 'Compiled: OK')
autoencoder.summary()

# Train model or load weights

train_data, train_label = prep_data1('train')
val_data, val_label = prep_data1('val')
nb_epoch = 100
batch_size = 2
#history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(val_data, val_label))
autoencoder.load_weights('model_5l_weight_lungs_segnet.hdf5')

#autoencoder.load_weights('model_5l_weight_ep50.hdf5')


layer1_filters = 64
layer2_filters = 128
layer3_filters = 256
layer4_filters = 512
layer5_filters = 512
layer6_filters = 256
layer7_filters = 128
layer8_filters = 64
test_data, test_label = prep_data1('test')
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

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 64):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,64):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[1].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,64):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[1].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A1 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer1_filters = new_num

####################### 1st convolution layer with 128 filters
print('1st convolution layer with 128 filters')
A = []
Acc = []

arr = autoencoder.evaluate(test_data, test_label, verbose=1)
print(arr)

filters, biases = autoencoder.layers[5].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 128):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,128):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[5].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,128):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[5].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A2 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer2_filters = new_num

####################### 1st convolution layer with 256 filters
print('1st convolution layer with 256 filters')
A = []
Acc = []

arr = autoencoder.evaluate(test_data, test_label, verbose=1)
print(arr)

filters, biases = autoencoder.layers[9].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 256):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,256):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[9].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,256):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[9].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A3 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer3_filters = new_num

####################### 1st convolution layer with 512 filters
print('1st convolution layer with 512 filters')
A = []
Acc = []

arr = autoencoder.evaluate(test_data, test_label, verbose=1)
print(arr)

filters, biases = autoencoder.layers[13].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 512):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,512):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[13].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,512):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[13].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A4 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer4_filters = new_num

####################### 2nd convolution layer with 512 filters
print('2nd convolution layer with 512 filters')
A = []
Acc = []

arr = autoencoder.evaluate(test_data, test_label, verbose=1)
print(arr)

filters, biases = autoencoder.layers[16].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 512):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,512):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[16].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,512):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[16].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A5 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer5_filters = new_num

####################### 2nd convolution layer with 256 filters
print('2nd convolution layer with 256 filters')
A = []
Acc = []

arr = autoencoder.evaluate(test_data, test_label, verbose=1)
print(arr)

filters, biases = autoencoder.layers[20].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 256):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,256):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[20].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,256):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[20].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A6 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer6_filters = new_num

####################### 2nd convolution layer with 128 filters
print('2nd convolution layer with 128 filters')
A = []
Acc = []

arr = autoencoder.evaluate(test_data, test_label, verbose=1)
print(arr)

filters, biases = autoencoder.layers[24].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 128):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,128):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[24].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,128):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[24].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A7 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer7_filters = new_num

####################### 2nd convolution layer with 64 filters
print('2nd convolution layer with 64 filters')
A = []
Acc = []

arr = autoencoder.evaluate(test_data, test_label, verbose=1)
print(arr)

filters, biases = autoencoder.layers[28].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 64):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,64):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[28].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,64):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        autoencoder.layers[28].set_weights([filters1, biases1])
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A8 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer8_filters = new_num

#################Compression##############3

autoencoder_compress = SegNet_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters)

print('Start')
optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
autoencoder_compress.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
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
    n_filters, ix = 64, 1
    
    index1 = 0
    # plot each channel separately
    for j in range(64):
        if(A1[j] == 1) :
            filters1[:, :, :, index1] = filters[:, :, :, j]
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights([filters1, biases1])

for k in range(2,5):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    
    A_1 = autoencoder.layers[k].get_weights()
    A_2 = autoencoder_compress.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(64):
        if(A1[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights(A_2)


for k in range(5, 6):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    filters, biases = autoencoder.layers[k].get_weights()
    filters1, biases1 = autoencoder_compress.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 128, 1
    
    index1 = 0
    # plot each channel separately
    for j in range(128):
        if(A2[j] == 1) :
            index2 = 0
            for l in range(64):
                if(A1[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights([filters1, biases1])

for k in range(6,9):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    
    A_1 = autoencoder.layers[k].get_weights()
    A_2 = autoencoder_compress.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(128):
        if(A2[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights(A_2)

for k in range(9, 10):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    filters, biases = autoencoder.layers[k].get_weights()
    filters1, biases1 = autoencoder_compress.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(A3[j] == 1) :
            index2 = 0
            for l in range(128):
                if(A2[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights([filters1, biases1])

for k in range(10,13):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    
    A_1 = autoencoder.layers[k].get_weights()
    A_2 = autoencoder_compress.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(A3[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights(A_2)

for k in range(13, 14):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    filters, biases = autoencoder.layers[k].get_weights()
    filters1, biases1 = autoencoder_compress.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 512, 1
    
    index1 = 0
    # plot each channel separately
    for j in range(512):
        if(A4[j] == 1) :
            index2 = 0
            for l in range(256):
                if(A3[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights([filters1, biases1])

for k in range(14,16):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    
    A_1 = autoencoder.layers[k].get_weights()
    A_2 = autoencoder_compress.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(512):
        if(A4[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights(A_2)

for k in range(16, 17):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    filters, biases = autoencoder.layers[k].get_weights()
    filters1, biases1 = autoencoder_compress.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 512, 1
    
    index1 = 0
    # plot each channel separately
    for j in range(512):
        if(A5[j] == 1) :
            index2 = 0
            for l in range(512):
                if(A4[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights([filters1, biases1])

for k in range(17,20):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    
    A_1 = autoencoder.layers[k].get_weights()
    A_2 = autoencoder_compress.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(512):
        if(A5[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights(A_2)

for k in range(20, 21):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    filters, biases = autoencoder.layers[k].get_weights()
    filters1, biases1 = autoencoder_compress.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(A6[j] == 1) :
            index2 = 0
            for l in range(512):
                if(A5[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights([filters1, biases1])

for k in range(21,24):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    
    A_1 = autoencoder.layers[k].get_weights()
    A_2 = autoencoder_compress.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(A6[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights(A_2)

for k in range(24, 25):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    filters, biases = autoencoder.layers[k].get_weights()
    filters1, biases1 = autoencoder_compress.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 128, 1
    
    index1 = 0
    # plot each channel separately
    for j in range(128):
        if(A7[j] == 1) :
            index2 = 0
            for l in range(256):
                if(A6[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights([filters1, biases1])

for k in range(25,28):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    
    A_1 = autoencoder.layers[k].get_weights()
    A_2 = autoencoder_compress.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(128):
        if(A7[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights(A_2)

for k in range(28, 29):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    filters, biases = autoencoder.layers[k].get_weights()
    filters1, biases1 = autoencoder_compress.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 64, 1
    
    index1 = 0
    # plot each channel separately
    for j in range(64):
        if(A8[j] == 1) :
            index2 = 0
            for l in range(128):
                if(A7[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights([filters1, biases1])

for k in range(29,31):
    if(len(autoencoder.layers[k].get_weights()) == 0):
        continue
    
    A_1 = autoencoder.layers[k].get_weights()
    A_2 = autoencoder_compress.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(64):
        if(A8[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    autoencoder_compress.layers[k].set_weights(A_2)

#train_data, train_label = prep_data1('train')
nb_epoch = 100
batch_size = 8
history = autoencoder_compress.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
autoencoder_compress.save_weights('model_5l_weight_compress_mini.hdf5')

# Model visualization
#from keras.utils.visualize_util import plot
#plot(autoencoder, to_file='model.JPG', show_shapes=True)

#test_data, test_label = prep_data1('test')
score = autoencoder_compress.evaluate(test_data, test_label, verbose=1)
print( 'Test score:', score[0])
print( 'Test accuracy:', score[1])

#output = autoencoder.predict(test_data, verbose=1)
#output = output.reshape((output.shape[0], img_h, img_w, n_labels))

#plot_results(output)
