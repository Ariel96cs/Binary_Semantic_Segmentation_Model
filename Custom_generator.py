from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import json

class CustomDataGen(Sequence):
    def __init__(self, x_paths,
                 y_paths,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=True, load_images_func=None):
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.read_image = load_images_func

        self.n = len(self.x_paths)

    def on_epoch_end(self):
        pass

    def __load_image(self,image_path,shape,gray=False):
        if gray:
            image = cv2.imread(image_path,0)
            image = image // 255
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, shape)
        return image

    def __load_images(self,image_paths,shape,gray=False):
        return np.array([self.__load_image(path,shape,gray)for path in image_paths])

    def __getitem__(self, index):
        batches_x = self.x_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batches_y = self.y_paths[index * self.batch_size:(index + 1) * self.batch_size]

        if self.read_image is None:
            X, y = self.__load_images(batches_x,self.input_size[0:2]), self.__load_images(batches_y,self.input_size[0:2],gray=True)
        else:
            batch = [self.read_image(img_path,label_path,self.input_size[0:2]) for img_path,label_path in zip(batches_x,batches_y)]

            X,y = zip(*batch)
            X,y = np.array(X),np.array(y)

        return X, y

    def __len__(self):
        return self.n // self.batch_size
