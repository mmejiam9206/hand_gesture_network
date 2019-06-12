import tensorflow as tf

def getlinearmodel(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_alexnet(num_class_alexnet):
    #Instantiate an empty model
    model = tf.keras.Sequential()
    # 1st Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(5,5), activation='relu',padding='valid', input_shape=(32, 32, 3), strides = (1, 1)))
    # Max Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='valid'))
    # 2nd Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation = 'relu'))
    # Max Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='valid'))
    # 3rd Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation = 'relu'))
    # 4th Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation = 'relu'))
    # 5th Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation = 'relu'))
    # Max Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='valid'))
    # Passing it to a Fully Connected layer
    model.add(tf.keras.layers.Flatten())
    # 1st Fully Connected Layer
    #ESTA PRIMERA FC LA AGREGUÈ YO PORQUE CREO QUE FALTA
    # 1st Fully Connected Layer
    model.add(tf.keras.layers.Dense(9216, activation = 'relu'))
    # Add Dropout to prevent overfitting
    model.add(tf.keras.layers.Dropout(0.4))
    # 2nd Fully Connected Layer
    model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
    # Add Dropout
    model.add(tf.keras.layers.Dropout(0.4))
    # 2nd Fully Connected Layer
    model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
    # Add Dropout
    model.add(tf.keras.layers.Dropout(0.4))
    # 3rd Fully Connected Layer
    model.add(tf.keras.layers.Dense(1000, activation = 'relu'))
    # Add Dropout
    model.add(tf.keras.layers.Dropout(0.4))
    # Output Layer
    model.add(tf.keras.layers.Dense(num_class_alexnet, activation = 'softmax'))
    return model