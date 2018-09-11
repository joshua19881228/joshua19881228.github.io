---
title: "Training MTCNN"
category: "Computer Vision"
tag: ["Face Detection", "Face Alignment"]
---

# MTCNN训练记录 #

最近尝试使用Caffe复现MTCNN，感觉坑很大，记录一下训练过程，目前还没有好的结果。网上也有很多童鞋在尝试训练MTCNN，普遍反映使用TensorFlow可以得到比较好的结果，但是使用Caffe不是很乐观。

## 已经遇到的问题现象 ##

* 2018.09.11：目前只训练了12net，召回率偏低。

    以[blankWorld/MTCNN-Accelerate-Onet](https://github.com/blankWorld/MTCNN-Accelerate-Onet)为baseline，blankWorld在FDDB上的测试性能如下图

    ![FDDB Result](https://raw.githubusercontent.com/blankWorld/MTCNN-Accelerate-Onet/master/img/mtcnn-fddb.jpg "FDDB Result"){: .center-image .image-width-480}

    这个效果很不错，但是我自己生成样本后训练12net，召回率有明显下降。性能对比如下图

    ![FDDB 12net Compare](/img/TrainMTCNN/12net_fddb.png "FDDB 12net Compare"){: .center-image .image-width-480}

    暂且不管12net的测试结果为什么会这么差，两个模型的性能差距是可以反映的。

## 训练记录 ##

 * 2018.09.11

    **训练数据生成** 

    参考[AITTSMD/MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)提供的prepare_data进行数据生成。数据集情况如下表

    | Training Set        | Positive      | Negative      | Part          | Landmark      |
    | :----------:        | :------:      | :-------:     | :--:          | :------:      |
    |**Number of Images** | 156728/189530 | 470184/975229 | 156728/547211 | 313456/357604 |
    |**Validation Set**     | Positive | Negative  | Part  | Landmark |
    |**Number of Images** | 10000    | 10000     | 10000 | 10000    |

    其中Pos:Neg:Part:Landmark = 1:3:1:2，样本比例参考原作的比例。Pos、Neg、Part来自于[WiderFace](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html)，Landmark来自于[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。其中正样本进行了人工的数据筛选，筛选的原因是根据WiderFace生成的正样本，有很多都是质量很差的图像，包含人脸大面积遮挡或十分模糊的情况。之前召回率很差的性能来自没有经过筛选的训练集，因为使用了OHEM，只有loss值在前70%的样本才参与梯度计算，感觉如果质量差的样本占比较大，网络学习到的特征是错误的，那些质量好的图像可能得不到充分的学习。

    **训练参数设置**

    初始训练参数如下

    ```
    type:"Adam"    
    momentum: 0.9
    momentum2:0.999
    delta:1e-8
    base_lr: 0.01
    weight_decay: 0.0005    
    batch_size: 256
    ```

    [//]: <> (训练路径在62服务器的/data2/zxli/CODE/caffe_multilabel/examples/mtcnn_12net/下，模型models_20180907，数据data_20180907，记录train_20180911。图像数据存储在/data2/zxli/GIT/mtcnn-caffe/prepare_data/12_20180905/)

    第一轮训练在75000次迭代(17.5个epoch)时停止，测试记录如下

    ```
    I0911 10:16:25.019253 21722 solver.cpp:347] Iteration 75000, Testing net (#0)
    I0911 10:16:28.057858 21727 data_layer.cpp:89] Restarting data prefetching from start.
    I0911 10:16:28.072748 21722 solver.cpp:414]     Test net output #0: cls_Acc = 0.4638
    I0911 10:16:28.072789 21722 solver.cpp:414]     Test net output #1: cls_loss = 0.096654 (* 1 = 0.096654 loss)
    I0911 10:16:28.072796 21722 solver.cpp:414]     Test net output #2: pts_loss = 0.008529 (* 0.5 = 0.0042645 loss)
    I0911 10:16:28.072801 21722 solver.cpp:414]     Test net output #3: roi_loss = 0.0221648 (* 0.5 = 0.0110824 loss)
    ```

    **注意：分类测试结果是0.4638是因为测试集没有打乱，1-10000为pos样本，10001-20000为neg样本，20001-30000为part样本，30001-40000为landmark样本。因此，实际分类正确率应该是0.9276**
    
    降低学习率至0.001，训练135000次迭代(31.5个epoch)时停止，测试记录如下

    ```
    I0911 13:14:36.482010 23543 solver.cpp:347] Iteration 135000, Testing net (#0)
    I0911 13:14:39.629933 23660 data_layer.cpp:89] Restarting data prefetching from start.
    I0911 13:14:39.645612 23543 solver.cpp:414]     Test net output #0: cls_Acc = 0.4714
    I0911 13:14:39.645649 23543 solver.cpp:414]     Test net output #1: cls_loss = 0.0765401 (* 1 = 0.0765401 loss)
    I0911 13:14:39.645656 23543 solver.cpp:414]     Test net output #2: pts_loss = 0.00756469 (* 0.5 = 0.00378234 loss)
    I0911 13:14:39.645661 23543 solver.cpp:414]     Test net output #3: roi_loss = 0.0201988 (* 0.5 = 0.0100994 loss)
    ```

    实际分类正确率是0.9428。