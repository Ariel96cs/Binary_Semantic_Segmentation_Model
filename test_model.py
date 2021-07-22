import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
# from model.iou_loss import IoU
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import time

INPUT_SIZE = 224
def execute_video(video_path,gray):
    cap = cv2.VideoCapture(str(video_path))

    while(True):
        # Capture frame-by-frame
        ret, img = cap.read()
        h,w = img.shape[:2]
        img = cv2.resize(img, (INPUT_SIZE,INPUT_SIZE))
        if gray:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img / 255.0

        input_batch=np.array([img])
        predict = model.predict(input_batch)

        output = predict[0]
        #output0 = cv2.resize(output, (w,h))
        cv2.imshow("img", img)
        # print(output[output>0.1])
        #cv2.imshow("img2", img2)
        cv2.imshow("mks", output)
        
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def load_single_img(img_path):
    image=tf.keras.preprocessing.image.load_img(str(img_path),target_size=(256,256))            
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr=input_arr/255.
    #input_arr=input_arr 
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    return input_arr

use_cpu=True
if(use_cpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


# model = tf.keras.models.load_model('model_checkpoint_tf.h5')
#model = load_model('unet_model_whole_100epochs.h5', compile=False)
#model.compile(optimizer=Adam(1e-4), loss=IoU, metrics=['binary_accuracy'])

# model.summary()

import numpy as np
import cv2
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('-d','--dataset_videos',type=str)
parser.add_argument('-m','--model_path',type=str)
parser.add_argument('-g','--gray',help="Gray scaled model input",type=bool,default=False)

args = parser.parse_args()
model = tf.keras.models.load_model(args.model_path)

for video_path in Path(args.dataset_videos).iterdir():
    try:
        print("Loading :",video_path)
        execute_video(video_path,args.gray)
        break
    except:
        continue

