from tensorflow.keras.layers import Conv2D,Dropout,MaxPooling2D,Conv2DTranspose, BatchNormalization,Input,Lambda,concatenate
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from CustomGenerator import CustomDataGen
from sklearn.model_selection import train_test_split
import json

mean_iou = MeanIoU(num_classes=2)

class BInstSeg:
    def __init__(self, input_shape,read_image_func=None):
        self.input_shape = input_shape
        self.model = None
        self.read_image_func = read_image_func

    def build_model(self,show_summary=True,nodes=8):
        # Build U-Net model
        inputs = Input(self.input_shape)
        # s = Lambda(lambda x: x / 255)(inputs)
        
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

    def train_model(self, x_train, y_train, early_stopping_patience=None, epochs=60, checkpoint_filepath='./checkpoints/',
                    save_best_only=True,validation_split=0.1, verbose=1,
                    batch_size=32, use_custom_generator_training=False,
                    save_distribution=True, initial_epoch=0,
                    x_val=None,y_val=None):
        callbacks = []
        if early_stopping_patience is not None:
            early_stopping = EarlyStopping(patience=early_stopping_patience,verbose=verbose)
            callbacks.append(early_stopping)
        if checkpoint_filepath is not None:
            name = 'instance_segmentation_model_{epoch:02d}-{val_loss:.4f}.h5'
            check_pointer = ModelCheckpoint(f'{checkpoint_filepath}/{name}', verbose=verbose,save_best_only=save_best_only)
            callbacks.append(check_pointer)
        if use_custom_generator_training:
            if x_val is None or y_val is None:
                X_train,X_val,y_train,y_val = train_test_split(x_train,y_train,test_size=validation_split,shuffle=True,
                                                            random_state=42)
            else:
                X_train,X_val,y_train,y_val = x_train,x_val,y_train,y_val

            if save_distribution:
                print("saving train and validation distribution")
                with open('train_val_distribution.json','w') as file:
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
        metrics = []
        if show_metrics:
            metrics.append('binary_accuracy')
        self.model.compile(optimizer='adam', loss=loss_function, metrics=metrics)
        
    def load_model(self, model_path):
        self.model = load_model(model_path)
        return self.model


