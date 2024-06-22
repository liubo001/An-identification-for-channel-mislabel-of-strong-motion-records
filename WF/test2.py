import os
import time
from keras.models import load_model
import numpy as np
import glob
from PIL import Image
from keras_preprocessing.image import img_to_array
from keras import backend as K

def load_dataset(path_name):
    img_path = glob.glob(path_name + "\\" + "*.png")
    img_path = np.array(img_path)
    return img_path

class Dataset:
    def __init__(self, train_path):
        self.train_path = train_path

    def load(self):#加载数据，并进行分类
        test_Mpaths = load_dataset(self.train_path)
        print(test_Mpaths.shape[0], 'test samples')

        self.test_Mpaths = test_Mpaths

class Model_Triplet:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = load_model(model_path)
        self.model.summary()

    def evaluate(self, dataset, batch_size):
        def generator(out):
            true_names = []
            a = 1
            j = 0
            while 1:
                x_paths = dataset.test_Mpaths[j:((j + batch_size))]
                j = (j + batch_size)
                test_names = []
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

                    test_names.append(str(x_path).split("\\")[-1])
                if out:
                    true_names.extend(test_names)
                    print(a)
                    a = a + 1

                with open("./predicts/ground_name.txt", "w") as f:
                    truth_name_out = "\n".join(true_names)
                    f.write(truth_name_out)

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
                x2_UD_NS_NS = np.array(x2_UD_NS_NS)
                x = [x1_EW_EW_UD, x2_UD_NS_NS]
                y = np.concatenate([labels0, labels1, labels00], axis=0)
                y = np.array(y)
                if len(y) == 0:
                    break
                yield x, y

        predicts = self.model.predict(generator(out=True), steps=int((dataset.test_Mpaths.shape[0]) / batch_size),
                                      callbacks=None, max_queue_size=10,
                                      workers=1, use_multiprocessing=False, verbose=0)

        np.savetxt("./predicts/predicts_probability.txt", predicts, fmt="%s")


if __name__ == '__main__':
    valid_path = "./test"
    dataset = Dataset(valid_path)

    model_name = "model_10_0.1387"
    batch_size = 1

    model_path = "./model_resnet/%s.h5" % model_name

    pred_path = "./predicts"
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    dataset.load()
    model = Model_Triplet()
    model.load_model(model_path)

    start = time.time()
    model.evaluate(dataset, batch_size)
    end = time.time()