from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam



class Model:
    def __init__(self, input_shape, num_classes=10, kernel_size=(5,5), kernel_size2=(3,3), ndense=128, nout1=28, nout2=28):
        self.__model = Sequential()
        self.__model.add(Conv2D(nout1, kernel_size, 
                            padding='valid', 
                            input_shape=input_shape,
                            activation='tanh'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))
        self.__model.add(Conv2D(nout2, kernel_size2, padding='valid', activation='tanh'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))
        self.__model.add(Flatten())
        self.__model.add(Dense(ndense, activation='tanh'))
        self.__model.add(Dense(num_classes, activation='softmax'))
        
        
    def train(self, X_train, y_train, batch_size, epochs, lr, decay, X_val, y_val):
        opt = Adam(learning_rate=lr, decay=decay)
        self.__model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
        self.hist_target = self.__model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                shuffle=True,verbose=0)
    
    def eval(self):
        print('Training accuracy = %f'%self.hist_target.history['accuracy'][-1])
        print('Validation accuracy = %f'%self.hist_target.history['val_accuracy'][-1])
        
    def predict(self, X_test):
        return self.__model.predict(X_test)
    
    def save(self, filepath):
        self.__model.save(filepath)
        