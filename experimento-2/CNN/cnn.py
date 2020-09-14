from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.models import load_model


def runCNN():
    for epochAmount in range(1, 6):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # input image rwo and column
        input_img_row = x_train[0].shape[0]
        input_img_cols = x_train[0].shape[1]

        # reshape the input image to one dimension
        x_train = x_train.reshape(
            x_train.shape[0], input_img_row, input_img_cols, 1)
        x_test = x_test.reshape(
            x_test.shape[0], input_img_row, input_img_cols, 1)

        input_shape = (input_img_row, input_img_cols, 1)

        # set all input image as a type float32
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        # normalize the input data

        x_train = x_train/255
        x_test = x_test/255

        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classes = y_train.shape[1]

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3),
                         activation="relu", input_shape=input_shape))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(units=128, activation="relu",))

        model.add(Dropout(0.5))

        model.add(Dense(units=num_classes, activation="softmax",))

        model.compile(optimizer=SGD(0.01),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        csv_logger = CSVLogger(
            'log-{}.csv'.format(epochAmount), append=True, separator=';')

        model.fit(x=x_train,
                  y=y_train,
                  batch_size=35,
                  epochs=epochAmount,
                  verbose=2,
                  validation_data=(x_test, y_test),
                  callbacks=[csv_logger])

        model.evaluate(x_test, y_test, callbacks=[csv_logger])

        model.save("cnn_model-{}.h5".format(epochAmount))


runCNN()
