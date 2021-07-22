import scipy
import numpy as np
from PIL import ImageEnhance
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img,array_to_img

class RandomAugmetationGen(ImageDataGenerator):
    def __init__(self, input_shape, featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None):
        """Generate batches of tensor image data with real-time data augmentation.

   The data will be looped over (in batches).

  Arguments:
      model_input: Tuple, image.shape
      featurewise_center: Boolean.
          Set input mean to 0 over the dataset, feature-wise.
      samplewise_center: Boolean. Set each sample mean to 0.
      featurewise_std_normalization: Boolean.
          Divide inputs by std of the dataset, feature-wise.
      samplewise_std_normalization: Boolean. Divide each input by its std.
      zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
      zca_whitening: Boolean. Apply ZCA whitening.
      rotation_range: Int. Degree range for random rotations.
      width_shift_range: Float, 1-D array-like or int
          - float: fraction of total width, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-width_shift_range, +width_shift_range)`
          - With `width_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `width_shift_range=[-1, 0, +1]`,
              while with `width_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      height_shift_range: Float, 1-D array-like or int
          - float: fraction of total height, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-height_shift_range, +height_shift_range)`
          - With `height_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `height_shift_range=[-1, 0, +1]`,
              while with `height_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      brightness_range: Tuple or list of two floats. Range for picking
          a brightness shift value from.
      shear_range: Float. Shear Intensity
          (Shear angle in counter-clockwise direction in degrees)
      zoom_range: Float or [lower, upper]. Range for random zoom.
          If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
      channel_shift_range: Float. Range for random channel shifts.
      fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
          Default is 'nearest'.
          Points outside the boundaries of the input are filled
          according to the given mode:
          - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
          - 'nearest':  aaaaaaaa|abcd|dddddddd
          - 'reflect':  abcddcba|abcd|dcbaabcd
          - 'wrap':  abcdabcd|abcd|abcdabcd
      cval: Float or Int.
          Value used for points outside the boundaries
          when `fill_mode = "constant"`.
      horizontal_flip: Boolean. Randomly flip inputs horizontally.
      vertical_flip: Boolean. Randomly flip inputs vertically.
      rescale: rescaling factor. Defaults to None.
          If None or 0, no rescaling is applied,
          otherwise we multiply the data by the value provided
          (after applying all other transformations).
      preprocessing_function: function that will be applied on each input.
          The function will run after the image is resized and augmented.
          The function should take one argument:
          one image (Numpy tensor with rank 3),
          and should output a Numpy tensor with the same shape.
      data_format: Image data format,
          either "channels_first" or "channels_last".
          "channels_last" mode means that the images should have shape
          `(samples, height, width, channels)`,
          "channels_first" mode means that the images should have shape
          `(samples, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      validation_split: Float. Fraction of images reserved for validation
          (strictly between 0 and 1).
      dtype: Dtype to use for the generated arrays.
        """
        super().__init__(featurewise_center=featurewise_center,
        samplewise_center=samplewise_center, 
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=samplewise_std_normalization,
        zca_whitening=zca_whitening, 
        zca_epsilon=zca_epsilon, 
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range, 
        zoom_range=zoom_range, 
        channel_shift_range=channel_shift_range, 
        fill_mode=fill_mode,
        cval=cval, horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=rescale,
        preprocessing_function=preprocessing_function, 
        data_format=data_format, validation_split=validation_split, dtype=dtype)

        self.input_shape = input_shape

    def generate_random_transformation_f(self,seed=None):
        params = self.get_random_transform(self.input_shape, seed)
        def transformation_funct(apply_brightness_mod=True):
            if not apply_brightness_mod:
                params['brightness'] = None
            print("Checking brightness:",params['brightness'])
            def f(image):
                return self.apply_transform(image, params)
            return f
        return transformation_funct


if __name__ == "__main__":
    from pathlib import Path
    from argparse import ArgumentParser
    import cv2 as cv

    parser = ArgumentParser(description="Main method for testing RandomAugmentationGenerator")
    parser.add_argument('-i','--image_path',help="Image input path",type=str)
    parser.add_argument('-mask','--mask_path',help="Mask input path",type=str)
    parser.add_argument('--iter',help='Number of modifications to check',type=int,default=15)
    args = parser.parse_args()
    input_shape = (224,224)


    # data_gen_args = dict(#featurewise_center=True,
    #                      #featurewise_std_normalization=True,
    #                      rotation_range=90,
    #                      width_shift_range=0.1,
    #                      height_shift_range=0.1,
    #                      zoom_range=0.2)
    # image_datagen = ImageDataGenerator(
    #                     rotation_range=50,
    #                      width_shift_range=0.1,
    #                      height_shift_range=0.1,
    #                      zoom_range=0.05,rescale=1./255,shear_range=5
    #                      )
    # # mask_datagen = ImageDataGenerator(rotation_range=10,
    # #                      width_shift_range=0.1,
    # #                      height_shift_range=0.1,
    # #                      zoom_range=0.1)

    # # Provide the same seed and keyword arguments to the fit and flow methods
    # seed = 0
    # # image_datagen.fit(images, augment=True, seed=seed)
    # # mask_datagen.fit(masks, augment=True, seed=seed)

    # image_generator = image_datagen.flow_from_directory(
    #     args.image_path,
    #     class_mode=None,
    #     seed=seed)
    # mask_generator = image_datagen.flow_from_directory(args.mask_path,class_mode=None,seed=seed)
    # # mask_generator = mask_datagen.flow_from_directory(
    # #     args.mask_path,
    # #     class_mode=None,
    # #     seed=seed,interpolation='bilinear')

    # # combine generators into one which yields image and masks
    # train_generator = zip(image_generator, mask_generator)

    # for images, masks in train_generator:
    #     for image,mask in zip(images,masks):
    #         image = np.array(image)

    #         print(mask.shape)
    #         # print("Image",image)
    #         # print("Mask",mask)
    #         cv.imshow('img',image)
    #         cv.imshow('mask',mask)
    #         mask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
    #         mask = np.array(mask,dtype='uint8')
    #         cv.imshow('cut',cv.bitwise_and(image, image, mask=mask))
    #         k = cv.waitKey(0) & 0xFF
    #         if k == ord('q'):
    #             exit(0)

    # cv.destroyAllWindows()

    augm = RandomAugmetationGen(input_shape,rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest',
                                    brightness_range=(10,80),)

    img = img_to_array(load_img(args.image_path,color_mode='grayscale')) *1./255
    
    # img = cv.resize(img,input_shape[:2])

    mask = img_to_array(load_img(args.mask_path,color_mode='grayscale'),dtype='uint8') 
    # mask = cv.resize(mask,input_shape[:2])

    cv.imshow('original img',img)
    cv.imshow('original mask',mask)
    cv.imshow('original cut',cv.bitwise_and(img, img, mask=mask))

    for _ in range(args.iter):
        modification_set = augm.generate_random_transformation_f()
        modification_fx = modification_set(apply_brightness_mod=True)
        modification_fy = modification_set(apply_brightness_mod=False)

        
        new_img = modification_fx(img.copy())
        new_img = cv.resize(new_img,input_shape[:2])

        new_mask = modification_fy(mask.copy())
        new_mask = cv.resize(new_mask,input_shape[:2])

        cv.imshow('transform cut',cv.bitwise_and(new_img, new_img, mask=new_mask))
        k = cv.waitKey(0) & 0xFF
        if k == ord('q'):
            exit(0)
        

    cv.destroyAllWindows()









