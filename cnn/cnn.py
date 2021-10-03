import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, image
from keras import layers
from keras import models
from keras.layers import Dropout
from tensorflow.keras import optimizers
from keras.models import load_model

# from ann_visualizer.visualize import ann_viz;

train_dir = '../data/train/'
validation_dir = '../data/validation/'
model_file_name = 'cat_dog_model.h5'
 
 
def init_model():
    model = models.Sequential()
 
    KERNEL_SIZE = (3, 3)
 
    model.add(layers.Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
 
    model.add(layers.Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
 
    model.add(layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
 
    model.add(layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
 
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
 
    model.add(Dropout(0.5))
 
    model.add(layers.Dense(1, activation='sigmoid'))
 
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-3),
                  metrics=['accuracy'])
 
    return model
 
 
def fig_loss(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
 
 
def fig_acc(history):
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
 
 
def fit(model):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
 
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=128,
        class_mode='binary')
 
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=256,
        class_mode='binary')
 
    history = model.fit_generator(
        train_generator,
        # steps_per_epoch=,
        epochs=11,
        validation_data=validation_generator,
        # validation_steps=,
    )
 
    model.save(model_file_name)
 
    fig_loss(history)
    fig_acc(history)
 
 
def predict():
    model = load_model(model_file_name)
    print(model.summary())
 
    img_path = '../data/test/cat/cat.5001.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = img_tensor / 255
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # 其形状为 (1, 150, 150, 3)

    result = model.predict(img_tensor)
    print("the prediction result for this picture is: ", result)

    plt.imshow(img_tensor[0])
    plt.show()
 
    #result = model.predict(img_tensor)
    #print(result)
 
CLASS_NAMES = ["Cat", "Dog"]

def predict_sigle(model, img_path):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = img_tensor / 255
    img_tensor = np.expand_dims(img_tensor, axis=0)

    prediction = model.predict(img_tensor)

    plt.imshow(img_tensor[0])
    plt.show()
    
    if prediction[0] > 0.5:
        predict_class = CLASS_NAMES[1]
    else:
        predict_class = CLASS_NAMES[0]

    return predict_class

 
# 画出count个预测结果和图像
def fig_predict_result(model, count):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        '../data/test/',
        target_size=(150, 150),
        batch_size=256,
        class_mode='binary')
 
    text_labels = []
    plt.figure(figsize=(30, 20))
    # 迭代器可以迭代很多条数据，但我这里只取第一个结果看看
    for batch, label in test_generator:
        pred = model.predict(batch)
        for i in range(count):
            true_reuslt = label[i]
            print(true_reuslt)
            if pred[i] > 0.5:
                text_labels.append('dog')
            else:
                text_labels.append('cat')
 
            # 4列，若干行的图
            plt.subplot(count / 4 + 1, 4, i + 1)
            plt.title('This is a ' + text_labels[i])
            imgplot = plt.imshow(batch[i])
 
        plt.show()
 
        # 可以接着画很多，但是只是随机看看几条结果。所以这里停下来。
        break
 
 
if __name__ == '__main__':
    #model = init_model()
    #model = load_model(model_file_name)
    #fit(model)
 
    # 利用训练好的模型预测结果。
    #predict()

    # img_path = './data/test/cat/cat.4153.jpg'
    # predict_class = predict_sigle(model, img_path)
    # print(predict_class)
    
    model = load_model(model_file_name)
    # ann_viz(model, title="CNN")
    top_layer = model.layers[0]
    plt.imshow(top_layer.get_weights()[0][:, :, :, 0].squeeze(), cmap='gray')
    #随机查看10个预测结果并画出它们
    fig_predict_result(model, 10)
