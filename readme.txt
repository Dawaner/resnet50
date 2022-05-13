
代码中仍有不严谨的地方，请谨慎观看
具体说明内容请看.doc

数据集：cifar10
库：tensorflow

实验环境:
CPU:i7-7700hq
GPU:GTX 1050
RAM:2*8G 2400Hz

Python环境:
Python3.8.0
tensorflow-gpu 2.8.0

驱动：
CUDA 11.2
cuDNN 8.1
NVIDIA 512.15

------------------------------------------------------

文件：
实验1,2
resnet50_train.py,resnet_50.py

实验3
pretained.py

实验4
fashion_resnet50.py

------------------------------------------------------

说明：
# self.inplanes != planes * block.expansion
# inplanes:输入block之前的通道数
# planes:在block中间处理的时候的通道数（这个值是输出维度的1/4)
# planes * block.expansion:输出的维度 (block.expansion=4)
在tensorflow中，没有参数能获得inplanes，因此很难验证通道数是否匹配。
但是resnet50，每次都是不匹配的，因此理论上通过三次卷积单元后进行shape匹配是能后解决问题的。
建议学习resnet50及以上的使用pytorch。

------------------------------------------------------

参考链接：
https://www.freesion.com/article/51081367268/
https://zhuanlan.zhihu.com/p/353235794
https://zhuanlan.zhihu.com/p/374448655

------------------------------------------------------
