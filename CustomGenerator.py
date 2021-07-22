from tensorflow.keras.utils import Sequence
import cv2 as cv
import numpy as np
import json
from keras.preprocessing.image import load_img
from random import randint,randrange,random

def rot():
    angle=randint(0,50)
    p_rot = (randint(-1,1)*randint(1,50),randint(-1,1)*randint(1,50))
    scale=1-random()*0.1
    def f (image):
        height,width = image.shape[:2]
        rotation_matrix =  cv.getRotationMatrix2D((width//2 + p_rot[0],height//2 + p_rot[1]),angle,scale)
        rotated_image = cv.warpAffine(image,rotation_matrix,(width,height)) 
        return rotated_image
    return f

def brightness():
    decrease_b=bool(randint(0,1))
    bright_constant=randint(10,80)
    def f(image):
        bright = np.ones(image.shape,dtype='uint8')*bright_constant
        if decrease_b:
            return cv.subtract(image,bright)
        return cv.add(image,bright)
    return f

def flips():
    flip_mode=randint(-1,1)
    to_flip = bool(randint(0,1))

    def f(image):
        if to_flip:
            return cv.flip(image,flip_mode)
        return image

    return f


class CustomDataGen(Sequence):
    def __init__(self, x_paths,
                 y_paths,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=True, load_images_func=None,data_augmentation=False):
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.read_image = load_images_func
        self.data_augmentation = data_augmentation

        self.n = len(self.x_paths)

    def on_epoch_end(self):
        pass

    def __load_image(self,image_path,shape,gray=False,mods=None):
        # print("Loading image:",image_path)
        if gray:
            if 'tif' in image_path:
                image = np.array(load_img(image_path,color_mode='grayscale'))
            else:
                image = cv.imread(image_path,0)
            
        else:
            # opencv don't read tif images
            if 'tif' in image_path:
                image = np.array(load_img(image_path))
            else:
                image = cv.imread(image_path)
                image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        
        image = cv.resize(image, shape)
        if mods is not None:
            for f in mods:
                image = f(image)

        image = image / 255
        # print("image shape", image.shape)
        # cv.imshow('Image',image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return image

    # def __load_images(self,image_paths,shape,gray=False):
    #     return np.array([self.__load_image(str(path),shape,gray)for path in image_paths])

    def __getitem__(self, index):
        batches_x = self.x_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batches_y = self.y_paths[index * self.batch_size:(index + 1) * self.batch_size]

        if self.read_image is None:
            # X, y = self.__load_images(batches_x,self.input_size[0:2],gray=self.input_size[-1]==1), self.__load_images(batches_y,self.input_size[0:2],gray=True)
            X ,y = [],[]
            shape = self.input_size[:2]
            # print("Input_shape in getItem",shape)
            if self.data_augmentation:
                for x_path,y_path in zip(batches_x,batches_y):
                    rotation_f = rot()
                    flips_f = flips()
                    brightness_f = brightness()
                    y_mod = [rotation_f,flips_f]
                    x_mod = [rotation_f,flips_f,brightness_f]
                    X.append(self.__load_image(x_path,shape,gray=self.input_size[-1]==1,mods=x_mod))
                    y.append(self.__load_image(y_path,shape,gray=True,mods=y_mod))
            else:
                for x_path,y_path in zip(batches_x,batches_y):
                    X.append(self.__load_image(x_path,shape,gray=self.input_size[-1]==1))
                    y.append(self.__load_image(y_path,shape,gray=True))


        else:
            batch = [self.read_image(img_path,label_path,self.input_size[0:2]) for img_path,label_path in zip(batches_x,batches_y)]

            X,y = zip(*batch)
        X,y = np.array(X),np.array(y)

        return X, y

    def __len__(self):
        return self.n // self.batch_size


