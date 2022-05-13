import  tensorflow as tf
# from    keras import layers, Sequential



class bottleneckBlock(tf.keras.layers.Layer):
    # 残差模块
    def __init__(self, filter_num, stride=1):
        super(bottleneckBlock, self).__init__()
        # 第一个卷积单元
        self.conv1 = tf.keras.layers.Conv2D(filter_num, (1, 1), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        # 第二个卷积单元
        self.conv2 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        # 第三个卷积单元
        self.conv3 = tf.keras.layers.Conv2D(filter_num * 4, (1, 1), strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()


        if stride != 1 or filter_num != filter_num * 4:# 通过1x1卷积完成shape匹配        #if have better algorithm, fix it
            # Fix(maybe √) wrong -> self.inplanes != planes * block.expansion
            # inplanes:输入block之前的通道数
            # planes:在block中间处理的时候的通道数（这个值是输出维度的1/4)
            # planes * block.expansion:输出的维度 (block.expansion=4)
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filter_num * 4, (1, 1), strides=stride))
        else:# shape匹配，直接短接
            self.downsample = lambda x:x

    def call(self, inputs, training=None):

        # [b, h, w, c]，通过第一个卷积单元
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 通过第三个卷积单元
        out = self.conv3(out)
        out = self.bn3(out)
        # 通过identity模块
        identity = self.downsample(inputs)

        # print("identity.shape",identity.shape)
        # print("out.shape",out.shape)

        # 3条路径输出直接相加                                         Fix(√)  #Wrong -> need to fit shape(identity,out)
        output = tf.keras.layers.add([out, identity])
        output = tf.nn.relu(output) # 激活函数

        return output


class ResNet(tf.keras.Model):
    # 通用的ResNet实现类
    def __init__(self, layer_dims, num_classes=10): # [3, 4, 6, 3]
        super(ResNet, self).__init__()
        # 根网络，预处理
        self.stem = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Activation('relu'),
                                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        # 堆叠4个Block，每个block包含了多个BasicBlock,设置步长不一样
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)


        # 通过Pooling层将高宽降低为1x1
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):
        # 通过根网络
        x = self.stem(inputs)
        # 一次通过4个模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 通过池化层
        x = self.avgpool(x)
        # 通过全连接层
        x = self.fc(x)

        return x



    def build_resblock(self, filter_num, blocks, stride=1):
        # 辅助函数，堆叠filter_num个BasicBlock
        res_blocks = tf.keras.Sequential()
        # 只有第一个BasicBlock的步长可能不为1，实现下采样
        res_blocks.add(bottleneckBlock(filter_num, stride))

        for _ in range(1, blocks):#其他BasicBlock步长都为1
            res_blocks.add(bottleneckBlock(filter_num, stride=1))

        return res_blocks



def resnet50():
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([3, 4, 6, 3])