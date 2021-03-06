from tensorflow.keras.layers import Conv2D,Dropout,MaxPooling2D,Conv2DTranspose, BatchNormalization,Input,Lambda,concatenate,Add,LeakyReLU
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import efficientnet as eff
import tensorflow as tf
from CustomGenerator import CustomDataGen
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras import backend as K

mean_iou = MeanIoU(num_classes=2)

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

class BInstSeg:
    def __init__(self, input_shape,read_image_func=None):
        self.input_shape = input_shape
        self.model = None
        self.read_image_func = read_image_func
        self.dice_coefficient = False
    

    def build_unet_BackboneEffB4(self,input_shape=(512, 512, 3),dropout_rate=0.1,weights=None,show_summary=True):

        backbone = eff.EfficientNetB4(weights=weights,
                                include_top=False,
                                input_shape=input_shape)
        input = backbone.input
        start_neurons = 16

        conv4 = backbone.layers[342].output
        conv4 = LeakyReLU(alpha=0.1)(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(dropout_rate)(pool4)
        
        # Middle
        convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
        convm = residual_block(convm,start_neurons * 32)
        convm = residual_block(convm,start_neurons * 32)
        convm = LeakyReLU(alpha=0.1)(convm)
        
        deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(dropout_rate)(uconv4)
        
        uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = residual_block(uconv4,start_neurons * 16)
        uconv4 = residual_block(uconv4,start_neurons * 16)
        uconv4 = LeakyReLU(alpha=0.1)(uconv4)
        
        deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
        conv3 = backbone.layers[153].output
        uconv3 = concatenate([deconv3, conv3])    
        uconv3 = Dropout(dropout_rate)(uconv3)
        
        uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = residual_block(uconv3,start_neurons * 8)
        uconv3 = residual_block(uconv3,start_neurons * 8)
        uconv3 = LeakyReLU(alpha=0.1)(uconv3)

        deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
        conv2 = backbone.layers[92].output
        uconv2 = concatenate([deconv2, conv2])
            
        uconv2 = Dropout(0.1)(uconv2)
        uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
        uconv2 = residual_block(uconv2,start_neurons * 4)
        uconv2 = residual_block(uconv2,start_neurons * 4)
        uconv2 = LeakyReLU(alpha=0.1)(uconv2)
        
        deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
        conv1 = backbone.layers[90].output
        uconv1 = concatenate([deconv1, conv1])
        
        uconv1 = Dropout(0.1)(uconv1)
        uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
        uconv1 = residual_block(uconv1,start_neurons * 2)
        uconv1 = residual_block(uconv1,start_neurons * 2)
        uconv1 = LeakyReLU(alpha=0.1)(uconv1)
        
        uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
        uconv0 = Dropout(0.1)(uconv0)
        uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
        uconv0 = residual_block(uconv0,start_neurons * 1)
        uconv0 = residual_block(uconv0,start_neurons * 1)
        uconv0 = LeakyReLU(alpha=0.1)(uconv0)
        
        uconv0 = Dropout(dropout_rate/2)(uconv0)
        output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv0)    
        
        model = Model(input, output_layer)
        self.model = model
        if show_summary:
            model.summary()

        return model

    def build_unet_BackboneEffBX(self,img_input_shape,nodes,weights=None,backbone_c=eff.EfficientNetB4, show_summary=True):
        backbone = backbone_c(include_top=False,input_shape=img_input_shape,weights=weights)
        inputs = backbone.inputs

        c5 = [layer for layer in backbone.layers if layer.name == 'top_activation'][0].output
        c4 = [layer for layer in backbone.layers if layer.name == 'block6a_expand_activation'][0].output
        c3 = [layer for layer in backbone.layers if layer.name == 'block4a_expand_activation'][0].output
        c2 = [layer for layer in backbone.layers if layer.name == 'block3a_expand_activation'][0].output
        c1 = [layer for layer in backbone.layers if layer.name == 'block2a_expand_activation'][0].output
        c0 = backbone.layers[2].output


        # DECODER Unet

        u6 = Conv2DTranspose(nodes*8, (2, 2), strides=(2, 2), padding='same',name='decoder_inputconv2dt')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(nodes*4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(nodes*2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(nodes, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)

        u10 = Conv2DTranspose(nodes, (2, 2), strides=(2, 2), padding='same',name='last_block_transponse')(c9)
        u10 = concatenate([u10, c0], axis=3)
        c10 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u10)
        c10 = BatchNormalization()(c10)
        c10 = Dropout(0.1)(c10)
        c10 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c10)
        c10 = BatchNormalization()(c10)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c10)

        model = Model(inputs=[inputs], outputs=[outputs])
        self.model = model

        if show_summary:
            model.summary()
        
        return model

    def build_model_MobileNetV2Encoder(self,show_summary=True,nodes=8):
        print('Buildding model with encoder as MobileNetV2')
        mobileNet = MobileNetV2(input_shape=self.input_shape,include_top=False)
        inputs = mobileNet.inputs

        c5 = [layer for layer in mobileNet.layers if layer.name == 'block_16_project_BN'][0].output
        c4 = [layer for layer in mobileNet.layers if layer.name == 'block_12_add'][0].output
        c3 = [layer for layer in mobileNet.layers if layer.name == 'block_5_add'][0].output
        c2 = [layer for layer in mobileNet.layers if layer.name == 'block_2_add'][0].output
        c1 = [layer for layer in mobileNet.layers if layer.name == 'expanded_conv_project_BN'][0].output


        # DECODER Unet

        u6 = Conv2DTranspose(nodes*8, (2, 2), strides=(2, 2), padding='same',name='decoder_inputconv2dt')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(nodes*4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(nodes*2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(nodes, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)

        u10 = Conv2DTranspose(nodes, (2, 2), strides=(2, 2), padding='same',name='last_block_transponse')(c9)
#         u10 = concatenate([u9, c1], axis=3)
        c10 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u10)
        c10 = BatchNormalization()(c10)
        c10 = Dropout(0.1)(c10)
        c10 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c10)
        c10 = BatchNormalization()(c10)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c10)

        model = Model(inputs=[inputs], outputs=[outputs])
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

        if show_summary:
            model.summary()

        self.model = model
        return model


    def build_model(self,show_summary=True,nodes=8):
        # Build U-Net model
        inputs = Input(self.input_shape)
        #inputs = Lambda(lambda x: x / 255)(inputs)
        
        # ENCODER
        c1 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        c3 = BatchNormalization()(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        c4 = BatchNormalization()(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(nodes*16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(nodes*16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)
        c5 = BatchNormalization()(c5)

        # DECODER
        u6 = Conv2DTranspose(nodes*8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(nodes*4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(nodes*2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(nodes, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

        if show_summary:
            model.summary()

        self.model = model
        return model

    def train_model(self, x_train, y_train, early_stopping_patience=None, reduce_lr_callback=True,epochs=60, checkpoint_filepath='./checkpoints/',
                    save_best_only=True,validation_split=0.1, verbose=1,
                    batch_size=32, use_custom_generator_training=False,
                    save_distribution=True, initial_epoch=0,
                    x_val=None,y_val=None):
        callbacks = []
        print('Batch size:',batch_size)
        if early_stopping_patience is not None:
            early_stopping = EarlyStopping(patience=early_stopping_patience,verbose=verbose)
            callbacks.append(early_stopping)
        if reduce_lr_callback:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=0.001)
            callbacks.append(reduce_lr)
        if checkpoint_filepath is not None:
            name = 'semantic_segmentation_model_{epoch:02d}-{val_loss:.4f}_dice_coeff-{val_dice_coeff:.4f}_binary_acc_{val_binary_accuracy:.4f}.h5'
            save_model_path = f'{checkpoint_filepath}/{name}'
            check_pointer = ModelCheckpoint(save_model_path, verbose=verbose,save_best_only=save_best_only)

            callbacks.append(check_pointer)
        if use_custom_generator_training:
            if x_val is None or y_val is None:
                X_train,X_val,y_train,y_val = train_test_split(x_train,y_train,test_size=validation_split,shuffle=True,
                                                            random_state=42)
            else:
                X_train,X_val,y_train,y_val = x_train,x_val,y_train,y_val

            if save_distribution:
                print("saving train and validation distribution")
                with open(f'{checkpoint_filepath}/train_val_distribution.json','w') as file:
                    json.dump({'X_train':X_train,'X_val':X_val},file)

            print("Train:",len(X_train))
            print("Validation:", len(X_val))
            traingen = CustomDataGen(X_train,y_train,batch_size,self.input_shape,load_images_func=self.read_image_func,data_augmentation=True)
            valgen = CustomDataGen(X_val, y_val, batch_size, self.input_shape,load_images_func=self.read_image_func)

            history = self.model.fit(traingen, validation_data=valgen,epochs=epochs,batch_size=batch_size,
                                     callbacks=callbacks,initial_epoch=initial_epoch)
        else:
            history = self.model.fit(x_train,y_train,validation_split=validation_split,batch_size=batch_size,
                                     epochs=epochs,
                                     callbacks=callbacks,initial_epoch=initial_epoch)
        return history

    def compile_model(self,loss_function='binary_crossentropy',show_metrics=True):
        print("Compiling model")
        metrics = []
        if show_metrics:
            metrics.append('binary_accuracy')
            # metrics.append(MeanIoU(num_classes=2))
            metrics.append(self.dice_coeff)
        if loss_function == 'dice_loss':
            self.dice_coefficient = True
            self.model.compile(optimizer='adam', loss=self.bce_dice_loss, metrics=metrics)
        else:
            self.model.compile(optimizer='adam', loss=loss_function, metrics=metrics)
        
    def load_model(self, model_path):
        self.model = load_model(model_path,custom_objects={'bce_dice_loss':self.bce_dice_loss,'dice_coeff':self.dice_coeff})
        return self.model
    def model_is_compiled(self):
        return self.model._is_compiled

    def dice_coeff(self,y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    def dice_loss(self,y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss
    
    def bce_dice_loss(self,y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss


