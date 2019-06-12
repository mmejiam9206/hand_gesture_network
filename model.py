
def get_linear_model(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_one_layer_nn(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_mult_layer_nn(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_lenet(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), activation='tanh',padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(16, (5,5), activation='tanh'))
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(120, (5,5), activation='tanh'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation='tanh'))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_alexnet(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(5,5), activation='relu',padding='valid', input_shape=(32, 32, 3), strides = (1, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation = 'relu'))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation = 'relu'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(9216, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(1000, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(num_class, activation = 'softmax'))
    return model 
