# CIFAR-10 Image Classification with CNN in TensorFlow/Keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)

#  Load data 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

#  Visualise first 10 images
plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])], fontsize=8)
    plt.axis('off')
plt.tight_layout(); plt.show()

#  Normalise
x_train = x_train.astype('float32')/255.0
x_test  = x_test.astype('float32')/255.0

# Build model
model = Sequential([
    Conv2D(32,3,activation='relu',input_shape=(32,32,3)),
    BatchNormalization(),
    Conv2D(32,3,activation='relu'),
    MaxPooling2D(2), Dropout(0.25),

    Conv2D(64,3,activation='relu'),
    BatchNormalization(),
    Conv2D(64,3,activation='relu'),
    MaxPooling2D(2), Dropout(0.25),

    Flatten(),
    Dense(512,activation='relu'), Dropout(0.5),
    Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Train
history = model.fit(x_train, y_train,
                    epochs=15, batch_size=64,
                    validation_data=(x_test, y_test))

#  Accuracy/Loss plots
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Accuracy'); plt.xlabel('epoch'); plt.ylabel('acc'); plt.legend(); plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.show()

#  Evaluate & predict (--> y_pred defined **here**)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal test accuracy: {test_acc*100:.2f}%")

y_pred_probs   = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)   # <- y_pred_classes ready

# Colour‑coded visual (green correct, red wrong)
plt.figure(figsize=(15,6))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i]); plt.axis('off')
    true = class_names[int(y_test[i])]
    pred = class_names[y_pred_classes[i]]
    colour = 'green' if pred == true else 'red'
    plt.title(f"Pred: {pred}\nTrue: {true}", color=colour, fontsize=9)
plt.tight_layout(); plt.show()
