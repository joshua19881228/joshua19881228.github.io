---
title: "Reading Note: $S^3FD$: Single Shot Scale-invariant Face Detector"
category: ["Computer Vision"]
tag: ["Reading Note", "Face Detection"]
---

**TITLE**: S$S^3FD$: Single Shot Scale-invariant Face Detector

**AUTHOR**: Shifeng Zhang, Xiangyu Zhu, Zhen Lei, Hailin Shi, Xiaobo Wang, Stan Z. Li

**ASSOCIATION**: Chinese Academy of Sciences

**FROM**: [arXiv:1708.05237](https://arxiv.org/abs/1708.05237)

## CONTRIBUTION ##

1. Proposing a scale-equitable face detection framework with a wide range of anchor-associated layers and a series of reasonable anchor scales so as to handle dif- ferent scales of faces well.
2. Presenting a scale compensation anchor matching strategy to improve the recall rate of small faces.
3. Introducing a max-out background label to reduce the high false positive rate of small faces.
4. Achieving state-of-the-art results on AFW, PASCAL face, FDDB and WIDER FACE with real-time speed.
2.

## METHOD ##

There are mainly three reasons that why the performance of anchor-based detetors drop dramatically as the objects becoming smaller:

1. **Biased Framework.** Firstly, the stride size of the lowest anchor-associated layer is too large, thus few features are reliable for small faces. Secondly, anchor scale mismatches receptive field and both are too large to fit small faces.
2. **Anchor Matching Strategy.** Anchor scales are discrete but face scale is continuous. Those faces whose scale distribute away from anchor scales can not match enough anchors, such as tiny and outer face.
3. **Background from Small Anchors.** Small anchors lead to sharp increase in the number of negative anchors on the background, bringing about many false positive faces.

The architecture of Single Shot Scale-invariant Face Detector is shown in the following figure.

![Framework](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20180120_S3FD.png "Framework"){: .center-image .image-width-640}
