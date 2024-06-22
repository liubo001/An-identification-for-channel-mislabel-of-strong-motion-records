from tensorflow import keras
from tensorflow.keras import layers

INPUT_SIZE = 224
CLASS_NUM = 3

# stage_name=2,3,4,5;  block_name=a,b,c
def ConvBlock(input_tensor, num_output, stride, stage_name, block_name):
    filter1, filter2 = num_output

    x = layers.Conv2D(filter1, 3, strides=stride, padding='same', name='res'+stage_name+block_name+'_branch2a')(input_tensor)
    x = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch2a')(x)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_branch2a_relu')(x)

    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same', name='res'+stage_name+block_name+'_branch2b')(x)
    x = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch2b')(x)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_branch2b_relu')(x)

    shortcut = layers.Conv2D(filter2, 1, strides=stride, padding='same', name='res'+stage_name+block_name+'_branch1')(input_tensor)
    shortcut = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch1')(shortcut)

    x = layers.add([x, shortcut], name='res'+stage_name+block_name)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_relu')(x)

    return x

def IdentityBlock(input_tensor, num_output, stage_name, block_name):
    filter1, filter2 = num_output

    x = layers.Conv2D(filter1, 3, strides=(1, 1), padding='same', name='res'+stage_name+block_name+'_branch2a')(input_tensor)
    x = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch2a')(x)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_branch2a_relu')(x)

    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same', name='res'+stage_name+block_name+'_branch2b')(x)
    x = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch2b')(x)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_branch2b_relu')(x)

    shortcut = input_tensor

    x = layers.add([x, shortcut], name='res'+stage_name+block_name)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_relu')(x)

    return x

def ResNet18(input_shape, class_num):
    input = keras.Input(shape=input_shape, name='input')

    # conv1
    x = layers.Conv2D(64, 7, strides=(2, 2), padding='same', name='conv1')(input)  # 7×7, 64, stride 2
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool1')(x)   # 3×3 max pool, stride 2

    # conv2_x
    x = ConvBlock(input_tensor=x, num_output=(64, 64), stride=(1, 1), stage_name='2', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=(64, 64), stage_name='2', block_name='b')

    # conv3_x
    x = ConvBlock(input_tensor=x, num_output=(128, 128), stride=(2, 2), stage_name='3', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=(128, 128), stage_name='3', block_name='b')

    # conv4_x
    x = ConvBlock(input_tensor=x, num_output=(256, 256), stride=(2, 2), stage_name='4', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=(256, 256), stage_name='4', block_name='b')

    # conv5_x
    x = ConvBlock(input_tensor=x, num_output=(512, 512), stride=(2, 2), stage_name='5', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=(512, 512), stage_name='5', block_name='b')

    # average pool, 1000-d fc, softmax
    x = layers.AveragePooling2D((7, 7), strides=(1, 1), name='pool5')(x)
    # x = layers.Flatten(name='flatten')(x)
    # x = layers.Dense(class_num, activation='softmax', name='fc1000')(x)

    model = keras.Model(input, x, name='resnet18')
    # model.summary()
    return model

