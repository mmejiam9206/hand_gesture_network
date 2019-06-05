import tensorflow as tf

def getlinearmodel(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model


