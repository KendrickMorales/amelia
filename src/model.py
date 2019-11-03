# import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50


class BruiseDetector():
    def __init__(self, input_shape):
        super(BruiseDetector, self).__init__()
        resnet = ResNet50(weights='imagenet', input_shape=input_shape,
                          include_top=False)
        x = resnet.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(101, activation='softmax')(x)

        self.model = Model(resnet.input, x)

        self.model.compile(optimizer='SGD',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit_gen(self, train_gen, batch_size=16, test_gen=None, epochs=100, steps_per_epoch=10):
        return self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=test_gen,
            validation_steps=1)
    
    def fit(self, X, y, batch_size=16, test_gen=None, epochs=100):
        return self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        return self.model.summary()

    def save(self, fname):
        return self.model.save(fname)

    def save_weights(self, fname):
        return self.model.save_weights(fname)

    def load_weights(self, fname):
        return self.model.load_weights(fname)


if __name__ == "__main__":
    model = BruiseDetector()

    model.summary()
