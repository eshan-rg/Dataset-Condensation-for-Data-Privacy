import tensorflow.keras.datasets as tfds
import numpy as np

(X_train, y_train), (X_test, y_test) = tfds.mnist.load_data()

X_target = X_train[: X_train.shape[0] // 2]
X_attack = X_train[X_train.shape[0] // 2 :]
y_target = y_train[: y_train.shape[0] // 2]
y_attack = y_train[y_train.shape[0] // 2 :]

X_target = X_target.reshape(X_target.shape[0], 28, 28, 1)
X_attack = X_attack.reshape(X_attack.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_target.shape, X_attack.shape, y_target.shape, y_attack.shape)

np.save("data/X_target.npy", X_target)
np.save("data/X_attack.npy", X_attack)
np.save("data/y_target.npy", y_target)
np.save("data/y_attack.npy", y_attack)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)
