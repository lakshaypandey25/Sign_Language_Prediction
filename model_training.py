from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import SGD
from data_loader import load_data

# Define the model
def create_model(input_shape=(28, 28, 1), num_classes=35):  # Updated num_classes to 35
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Compile and train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=128):
    model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    accuracy = model.evaluate(X_test, y_test, batch_size=32)
    print("Accuracy: ", accuracy[1])
    model.save('ISL_CNN_Model.h5')

# Example usage:
if __name__ == '__main__':
    dataset_path = 'C:/Users/lakshay pandey/Desktop/AI DA/dataset/Indian'  # Updated dataset path
    X_train, X_test, y_train, y_test = load_data(dataset_path)
    model = create_model()
    train_model(model, X_train, y_train, X_test, y_test)
