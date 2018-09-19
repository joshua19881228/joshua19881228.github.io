---
title: "Training MTCNN"
category: "Computer Vision"
tag: ["Face Detection", "Face Alignment"]
---

[//]: <> (# 训练步骤 #)
[//]: <> (1. 测试负样本，)

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

    |                   | Positive      | Negative      | Part          | Landmark      |
    | :----------:      | :------:      | :-------:     | :--:          | :------:      |
    | **Training Set**  | 156728/189530 | 470184/975229 | 156728/547211 | 313456/357604 |
    |**Validation Set** | 10000         | 10000         | 10000         | 10000         |

    其中Pos:Neg:Part:Landmark = 1:3:1:2，样本比例参考原作的比例。Pos、Neg、Part来自于[WiderFace](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html)，Landmark来自于[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。其中正样本进行了人工的数据筛选，筛选的原因是根据WiderFace生成的正样本，有很多都是质量很差的图像，包含人脸大面积遮挡或十分模糊的情况。之前召回率很差的性能来自没有经过筛选的训练集，因为使用了OHEM，只有loss值在前70%的样本才参与梯度计算，感觉如果质量差的样本占比较大，网络学习到的特征是错误的，那些质量好的图像可能得不到充分的学习。

    **训练参数设置**

    初始训练参数如下

    ```protobuf
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

    ```txt
    I0911 10:16:25.019253 21722 solver.cpp:347] Iteration 75000, Testing net (#0)
    I0911 10:16:28.057858 21727 data_layer.cpp:89] Restarting data prefetching from start.
    I0911 10:16:28.072748 21722 solver.cpp:414]     Test net output #0: cls_Acc = 0.4638
    I0911 10:16:28.072789 21722 solver.cpp:414]     Test net output #1: cls_loss = 0.096654 (* 1 = 0.096654 loss)
    I0911 10:16:28.072796 21722 solver.cpp:414]     Test net output #2: pts_loss = 0.008529 (* 0.5 = 0.0042645 loss)
    I0911 10:16:28.072801 21722 solver.cpp:414]     Test net output #3: roi_loss = 0.0221648 (* 0.5 = 0.0110824 loss)
    ```

    **注意：分类测试结果是0.4638是因为测试集没有打乱，1-10000为pos样本，10001-20000为neg样本，20001-30000为part样本，30001-40000为landmark样本。因此，实际分类正确率应该是0.9276**

    降低学习率至0.001，训练135000次迭代(31.5个epoch)时停止，测试记录如下

    ```vim
    I0911 13:14:36.482010 23543 solver.cpp:347] Iteration 135000, Testing net (#0)
    I0911 13:14:39.629933 23660 data_layer.cpp:89] Restarting data prefetching from start.
    I0911 13:14:39.645612 23543 solver.cpp:414]     Test net output #0: cls_Acc = 0.4714
    I0911 13:14:39.645649 23543 solver.cpp:414]     Test net output #1: cls_loss = 0.0765401 (* 1 = 0.0765401 loss)
    I0911 13:14:39.645656 23543 solver.cpp:414]     Test net output #2: pts_loss = 0.00756469 (* 0.5 = 0.00378234 loss)
    I0911 13:14:39.645661 23543 solver.cpp:414]     Test net output #3: roi_loss = 0.0201988 (* 0.5 = 0.0100994 loss)
    ```

    实际分类正确率是0.9428。训练260000次迭代后停止，测试记录如下

    ```vim
    I0911 16:58:47.514267 28442 solver.cpp:347] Iteration 260000, Testing net (#0)
    I0911 16:58:50.624385 28448 data_layer.cpp:89] Restarting data prefetching from start.
    I0911 16:58:50.639556 28442 solver.cpp:414]     Test net output #0: cls_Acc = 0.471876
    I0911 16:58:50.639595 28442 solver.cpp:414]     Test net output #1: cls_loss = 0.0750447 (* 1 = 0.0750447 loss)
    I0911 16:58:50.639602 28442 solver.cpp:414]     Test net output #2: pts_loss = 0.0074394 (* 0.5 = 0.0037197 loss)
    I0911 16:58:50.639608 28442 solver.cpp:414]     Test net output #3: roi_loss = 0.0199694 (* 0.5 = 0.00998469 loss)
    ```

    实际分类正确率是0.943752。

    **问题：** 训练结果看似还可以，但是召回率很低，在阈值设置为0.3的情况下，召回率也才将将达到90%。阈值要设置到0.05，才能达到97%-98%的召回率，ROC曲线如下图。严格来说这个测试并不严谨，应该用检测器直接在图像中进行检测，但是为了方便，我直接用val集上的性能画出了ROC曲线，其中的FDDB曲线是将的人脸区域截取出来进行测试得到的。

    ![12net 1st ROC](/img/TrainMTCNN/12net_roc_1st.png "12net 1st ROC"){: .center-image .image-width-480}

* 2018.09.14

    使用上述12net在WiderFace上提取正负样本，提取结果如下：

    | Thresholed | Positive | Negative | Part   |
    | :--------: | :------: | :------: | :---:  |
    | 0.05       | 85210    | 36745286 | 632861 |
    | 0.5        | 66224    | 6299420  | 354350 |

* 2018.09.17
  
    准备24net的训练样本。由于生成12net检测到的正样本数目有限，训练24net的pos样本包含两部分，一部分是训练12net的正样本，一部分是经过筛选的12net检测到的正样本；neg样本和part样本全部来自12net的难例；landmark与12net共用样本。经过采样后达到样本比例1:3:1:2，样本数目如下表：

    |                    | Positive | Negative | Part   | Landmark |
    | :----------:       | :------: | :------: | :--:   | :------: |
    | **Training Set**   | 225172   | 675516   | 225172 | 313456   |
    | **Validation Set** | 10000    | 10000    | 10000  | 10000    |

    [//]: <> (训练路径在62服务器的/data2/zxli/CODE/caffe_multilabel/examples/mtcnn_24net/下，模型models_20180917，数据data_20180916，记录train_20180917。图像数据存储在/data2/zxli/GIT/mtcnn-caffe/prepare_data/24_20180914/)

    训练过程与12net类似，学习率从0.01下降到0.0001，最终的训练结果如下

    ```vim
    I0917 15:19:00.631140 36330 solver.cpp:347] Iteration 70000, Testing net (#0)
    I0917 15:19:03.305665 36335 data_layer.cpp:89] Restarting data prefetching from start.
    I0917 15:19:03.317827 36330 solver.cpp:414]     Test net output #0: cls_Acc = 0.481501
    I0917 15:19:03.317865 36330 solver.cpp:414]     Test net output #1: cls_loss = 0.0479137 (* 1 = 0.0479137 loss)
    I0917 15:19:03.317874 36330 solver.cpp:414]     Test net output #2: pts_loss = 0.00631254 (* 0.5 = 0.00315627 loss)
    I0917 15:19:03.317879 36330 solver.cpp:414]     Test net output #3: roi_loss = 0.0179083 (* 0.5 = 0.00895414 loss)
    ```

    实际分类正确率是0.963。ROC曲线如下图，同样使用val集上的性能画出曲线。

    ![24net 1st ROC](/img/TrainMTCNN/24net_roc_1st.png "24net 1st ROC"){: .center-image .image-width-480}

* 2018.09.18

    使用24net在WiderFace上提取正负样本，提取结果如下：

    | Thresholed | Positive | Negative | Part   |
    | :--------: | :------: | :------: | :---:  |
    | 0.5, 0.5   | 86396    | 83212    | 225285 |

    利用以上数据生成48net的训练样本，由于24net生成的样本数量有限，结合前两次训练所用的数据，生成训练集：

    |                    | Positive | Negative | Part   | Landmark |
    | :----------:       | :------: | :------: | :--:   | :------: |
    | **Training Set**   | 283616   |  850848  | 283616 | 567232   |
    | **Validation Set** | 10000    | 10000    | 10000  | 10000    |