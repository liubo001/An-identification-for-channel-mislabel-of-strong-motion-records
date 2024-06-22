import os
import time
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from ResNet18 import ResNet18
import glob
from PIL import Image
from keras.backend import set_session
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.models import Sequential
from keras_preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta
from tensorflow.keras.applications import VGG19, resnet_v2, InceptionV3, inception_resnet_v2
import keras.backend as K
from keras import layers

def load_dataset(path_name):
    img_path = glob.glob(path_name + "\\" + "*.png")
    img_path = np.array(img_path)
    return img_path

def read_shape(path_name,shapes):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_shape(full_path,shapes)
        else:  # 文件
            if dir_item.endswith('.png'):
                img = Image.open(full_path)
                if img.mode != 'RGB':
                    img = img.convert("RGB")
                img = img_to_array(img)
                shape = img.shape
                shapes.append(shape)
                break
    print(shapes)
    return shapes
class LossHistory(keras.callbacks.Callback):#定义一个类，用于记录训练过程中的数据，继承自callback

    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 按照batch来追加数据
    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('binary_accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_binary_accuracy'))

    def on_epoch_end(self, epoch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('binary_accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_binary_accuracy'))

    def loss_plot(self, loss_type,model_name):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'k', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_loss[loss_type], 'r', label='val loss')
        plt.title(model_name)
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置

        plt.savefig("./result_data_resnet/%s.png"%(model_name))
        np.savetxt("./result_data_resnet/%s_train_acc_epoch.txt" %(model_name),self.accuracy['epoch'],fmt="%s")
        np.savetxt("./result_data_resnet/%s_train_loss_epoch.txt" %(model_name),self.losses['epoch'],fmt="%s")
        np.savetxt("./result_data_resnet/%s_test_acc_epoch.txt" %(model_name),self.val_acc['epoch'],fmt="%s")
        np.savetxt("./result_data_resnet/%s_test_loss_epoch.txt" %(model_name),self.val_loss['epoch'],fmt="%s")

        plt.show()

class Dataset:
    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.val_path = val_path

    def load(self):#加载数据，并进行分类
        train_Mpaths = load_dataset(self.train_path)
        val_Mpaths = load_dataset(self.val_path)
         # 输出训练集、验证集、测试集的数量
        print(train_Mpaths.shape[0], 'train samples')
        print(val_Mpaths.shape[0], 'val samples')
        self.train_Mpaths = train_Mpaths
        self.val_Mpaths = val_Mpaths


class Model_Triplet:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, drop_out, shapes):
        self.base_network = Sequential()
        # self.base_model = resnet_v2.ResNet50V2(weights="imagenet", include_top=False,
        #                                        input_shape=(shapes[0][0], int((shapes[0][1])), 3))
        # self.base_model = VGG19(weights="imagenet", include_top=False,
        #                                        input_shape=(shapes[0][0], int((shapes[0][1])), 3))
        # self.base_model = inception_resnet_v2.InceptionResNetV2(weights="imagenet", include_top=False,
        #                         input_shape=(shapes[0][0], int((shapes[0][1])), 3))
        self.base_model = ResNet18((224, 224, 3), 3)
        # self.base_model = ResNet34((224, 224, 3), 3)
        # self.base_model.summary()
        self.base_network.add(self.base_model)
        self.base_network.add(Flatten())


        input_a = layers.Input(shape=(shapes[0][0], int((shapes[0][1])), 3))
        input_p = layers.Input(shape=(shapes[0][0], int((shapes[0][1])), 3))
        processed_a = self.base_network(input_a)
        processed_p = self.base_network(input_p)
        
        l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        l1_distance = l1_distance_layer([processed_a, processed_p])
        out = Dense(128, activation='relu')(l1_distance)
        out = Dropout(drop_out)(out)
        out = Dense(1, activation='sigmoid')(out)
        self.model = Sequential()
        self.model = Model([input_a, input_p], out)
        self.model.summary()

    def train(self, dataset, batch_size, nb_epoch, history, model_name):
        # 控制显存使用
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.99
        set_session(tf.compat.v1.Session(config=config))

        # sgd = SGD(learning_rate=0.01, decay=10e-6, momentum=0.98, nesterov=True)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0)
        # adadelta = Adadelta(learning_rate=0.01, rho=0.95, epsilon=1e-07, decay=0.0010)
        # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-06, amsgrad=True)
        self.model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['binary_accuracy'])#categorical
        model_path = "./model_resnet/%s_" % model_name
        filename = model_path + "{epoch:02d}_{val_loss:.4f}.h5"
        checkpoint = ModelCheckpoint(filepath=filename, monitor="val_loss", mode="min",
                                     save_weights_only=False, save_best_only=False, verbose=1, period=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=5, restore_best_weights=True)
        callback_lists = [history, checkpoint]

        def train_generator():
            while 1:
                row = np.random.randint(0, len(dataset.train_Mpaths)-1, batch_size)
                x_paths = dataset.train_Mpaths[row]
                x_anchors, x_positives, x_negatives, labels0, labels1, labels00 = [], [], [], [], [], []
                for i, x_path in enumerate(x_paths):
                    img = Image.open(x_path)
                    if img.mode != 'RGB':
                        img = img.convert("RGB")
                    img = img_to_array(img)
                    RSN = str(x_path).split("\\")[-1]
                    x_p = ".\\H2\\" + RSN
                    img_p = Image.open(x_p)
                    if img_p.mode != 'RGB':
                        img_p = img_p.convert("RGB")
                    img_p = img_to_array(img_p)
                    label1 = 1
                    x_n = ".\\UD\\" + RSN
                    img_n = Image.open(x_n)
                    if img_n.mode != 'RGB':
                        img_n = img_n.convert("RGB")
                    img_n = img_to_array(img_n)
                    label0 = 0

                    x_anchors.append(img)
                    x_positives.append(img_p)
                    x_negatives.append(img_n)
                    labels0.append(label0)
                    labels00.append(label0)
                    labels1.append(label1)
                x10 = x_anchors
                # print(len(x10))#10
                x10.extend(x_anchors)#EW-UD,EW-NS
                x1_EW_EW = x10
                # print(len(x1_EW_EW))#_EW_EW))#20
                x1_EW_EW.extend(x_negatives)#EW-UD,EW-NS,UD-NS
                x1_EW_EW_UD = x1_EW_EW
                # print(len(x1_EW_EW_UD))#30
                x20 = x_negatives
                # print(len(x20))#10
                x20.extend(x_positives)
                x2_UD_NS = x20
                # print(len(x2_UD_NS))#20
                x2_UD_NS.extend(x_positives)
                x2_UD_NS_NS = x2_UD_NS
                # print(len(x2_UD_NS_UD))#30
                x1_EW_EW_UD = np.array(x1_EW_EW_UD)
                x2_UD_NS_NS = np.array(x2_UD_NS_NS)
                x = [x1_EW_EW_UD, x2_UD_NS_NS]
                y = np.concatenate([labels0, labels1, labels00], axis=0)
                # print(len(y))#30
                y = np.array(y)
                yield x, y

        def val_generator():
            while 1:
                row = np.random.randint(0, len(dataset.val_Mpaths)-1, batch_size)
                x_paths = dataset.val_Mpaths[row]
                x_anchors, x_positives, x_negatives, labels0, labels1, labels00 = [], [], [], [], [], []
                for i, x_path in enumerate(x_paths):
                    img = Image.open(x_path)
                    if img.mode != 'RGB':
                        img = img.convert("RGB")
                    img = img_to_array(img)
                    RSN = str(x_path).split("\\")[-1]
                    x_p = ".\\H2\\" + RSN
                    img_p = Image.open(x_p)
                    if img_p.mode != 'RGB':
                        img_p = img_p.convert("RGB")
                    img_p = img_to_array(img_p)
                    label1 = 1
                    x_n = ".\\UD\\" + RSN
                    img_n = Image.open(x_n)
                    if img_n.mode != 'RGB':
                        img_n = img_n.convert("RGB")
                    img_n = img_to_array(img_n)
                    label0 = 0

                    x_anchors.append(img)
                    x_positives.append(img_p)
                    x_negatives.append(img_n)
                    labels0.append(label0)
                    labels00.append(label0)
                    labels1.append(label1)
                x10 = x_anchors
                x10.extend(x_anchors)  # EW-UD,EW-NS
                x1_EW_EW = x10
                x1_EW_EW.extend(x_negatives)  # EW-UD,EW-NS,UD-NS
                x1_EW_EW_UD = x1_EW_EW
                x20 = x_negatives
                x20.extend(x_positives)
                x2_UD_NS = x20
                x2_UD_NS.extend(x_positives)
                x2_UD_NS_NS = x2_UD_NS
                x1_EW_EW_UD = np.array(x1_EW_EW_UD)
                # print(x1_EW_EW_UD)
                x2_UD_NS_NS = np.array(x2_UD_NS_NS)
                x = [x1_EW_EW_UD, x2_UD_NS_NS]
                y = np.concatenate([labels0, labels1, labels00], axis=0)
                y = np.array(y)
                yield x, y

        self.model.fit(train_generator(),
                       epochs=nb_epoch,
                       steps_per_epoch=int(len(dataset.train_Mpaths) / batch_size),
                       validation_data=val_generator(),
                       validation_steps=int(len(dataset.val_Mpaths) / batch_size),
                       callbacks=[early_stopping, callback_lists])


if __name__ == '__main__':
    if not os.path.exists("./result_data_resnet"):  # 如果不存在这个文件夹或目录，就执行下一步
        os.mkdir("./result_data_resnet")  # 创建新目录
    if not os.path.exists("./model_resnet"):
        os.mkdir("./model_resnet")
    if not os.path.exists("./test"):
        print("请补充数据集")
    history = LossHistory()
    train_path = ".\\H1"
    val_path = ".\\val"

    shapes = []
    shapes = read_shape(train_path, shapes)
    dataset = Dataset(train_path=train_path, val_path=val_path)

    dropout, batch_size, nb_epoch = 0.5, 64, 100

    model_name = "model"
    dataset.load()
    model = Model_Triplet()
    model.build_model(dataset, drop_out=dropout, shapes=shapes)

    start = time.time()
    model.train(dataset, batch_size, nb_epoch, history, model_name=model_name)
    end = time.time()

    print("总耗时=")
    print(end - start)
    history.loss_plot('epoch', model_name=model_name)  # 画图并且记录训练过程
