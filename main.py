

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


from pathlib import Path
from Binary_instance_seg_model import BInstSeg
from argparse import ArgumentParser
import cv2
import numpy as np
import json
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img
from os import makedirs


def read_image(img_path, label_path, image_shape):
    # print("image path->",img_path)
    # image = cv2.imread(img_path)
    image = np.array(load_img(img_path))
    # print("Image.shape:",image.shape)

    mask = np.zeros(image.shape, dtype=np.uint8)
    quad = json.load(open(label_path, 'r'))
    coords = np.array(quad['quad'], dtype=np.int32)
    cv2.fillPoly(mask, coords.reshape(-1, 4, 2), color=(255,255,255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2))
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    mask = cv2.threshold(mask, 0,255, cv2.THRESH_BINARY)[1]
    mask = cv2.resize(mask, image_shape)
    mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
    image = cv2.resize(image, image_shape)
    return image, mask

def generate_masks_images_from_json(x_paths,y_paths,size):
    for image_path,mask_path in tqdm(list(zip(x_paths,y_paths))):
        _,mask= read_image(image_path,mask_path.replace('.png','.json'),size)
        cv2.imwrite(mask_path,mask)
        # print(mask_path)

def load_paths(dataset_path):
    x_paths = []
    y_paths = []
    for type_docs_per_nation_folder in Path(dataset_path).iterdir():
        if not type_docs_per_nation_folder.is_file():
            images_folder = Path(f"{str(type_docs_per_nation_folder)}/images/")
            for diff_background_doc_folder in images_folder.iterdir():
                if not diff_background_doc_folder.is_file():
                    x_paths += list(diff_background_doc_folder.iterdir())
    x_paths = [str(i) for i in x_paths]
    y_paths = [i.replace('/images/','/ground_truth/').replace('.tif','.png') for i in x_paths]

    return x_paths,y_paths

def main():
    parser = ArgumentParser(description="Script for training Unet Model for mask generator")
    parser.add_argument("-d","--dataset_path",help="Dataset path",type=str,required=True)
    parser.add_argument("-m","--model_path",help="Pretrained model path to load",type=str)
    parser.add_argument('-ck','--model_checkpoint',help="Model checkpoint filepath",type=str,default='./checkpoints/')
    parser.add_argument("-i","--input_model_shape", help="Input model shape", type=int, default=224)
    parser.add_argument("-iepoch","--initial_epoch", help="Epoch index to restart training", type=int,default=0)
    parser.add_argument('-g','--grayscaled_model_input',help="Train a grayscaled input model",default=False,type=bool)
    parser.add_argument('-G', '--generate_masks_from_json', help="Generate masks images from ground truth json", default=False,
                        type=bool)
    parser.add_argument('-tf','--transfer_learning',help='Train model for transferLearning with specified dataset path for unsupervised learning',type=str)
    # parser.add_argument('-gpu','--gpu_memory_limit',help='Limit gpu memory disponibility',type=int)

    args = parser.parse_args()

        
    

    channels = 1 if args.grayscaled_model_input else 3
    model_input_shape = (args.input_model_shape,args.input_model_shape,channels)
    dataset_path = args.dataset_path
    model_path = args.model_path
    generate_masks = args.generate_masks_from_json
    if not Path(args.model_checkpoint).exists():
        makedirs(args.model_checkpoint)

    model = BInstSeg(model_input_shape)
    if model_path is not None:
        model.load_model(model_path)
        
    else:
        model.build_model(nodes=16)

    # model.compile_model()
    print("Loading dataset")
    x_paths = []
    y_paths = []

    if args.transfer_learning is not None:
        x_paths = list(Path(args.transfer_learning).iterdir())
        y_paths = x_paths
        print("Training for transfer learning")
        model.compile_model(loss_function='mse',show_metrics=False)
        model.train_model(x_paths,y_paths,early_stopping_patience=10,check_point_name="model_checkpoint_tf.h5"
                      ,use_custom_generator_training=True)
        model.load_model('model_checkpoint_tf.h5',compile=False)
    if not model.model_is_compiled():
        model.compile_model('dice_loss')
        
    print("Loading segmentation dataset")
    train_folder = Path(f'{dataset_path}/train/')
    val_folder = Path(f'{dataset_path}/val/')
    if train_folder.exists() and val_folder.exists():
        x_train,y_train = load_paths(train_folder)
        x_val,y_val = load_paths(val_folder)
        print("Training segmentation model")
        model.train_model(x_train,y_train,early_stopping_patience=100,checkpoint_filepath=args.model_checkpoint
                        ,use_custom_generator_training=True,batch_size=64,epochs=400,initial_epoch=args.initial_epoch,
                        x_val=x_val,y_val=y_val)

    else:
        x_paths,y_paths = load_paths(dataset_path)

        if generate_masks:
            print("Generating masks images")
            generate_masks_images_from_json(x_paths,y_paths,model_input_shape[0:2])
            return

        print("Training segmentation model")
        # model.compile_model()
        model.train_model(x_paths,y_paths,early_stopping_patience=10,checkpoint_filepath=args.model_checkpoint
                        ,use_custom_generator_training=True,epochs=200,initial_epoch=args.initial_epoch)

if __name__ == '__main__':
    main()

