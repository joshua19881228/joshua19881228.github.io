---
title: "Reading Note:  Be Your Own Prada: Fashion Synthesis with Structural Coherence"
category: ["Computer Vision"]
tag: ["Reading Note", "CNN", "GAN", "Fashion"]
---

**TITLE**: Be Your Own Prada: Fashion Synthesis with Structural Coherence

**AUTHOR**: Shizhan Zhu, Sanja Fidler, Raquel Urtasun, Dahua Lin, Chen Change Loy

**ASSOCIATION**: The Chinese University of Hong Kong, University of Toronto, Vector Institute, Uber Advanced Technologies Group

**FROM**: [ICCV2017](http://personal.ie.cuhk.edu.hk/~ccloy/files/iccv_2017_fashiongan.pdf)

## CONTRIBUTION ##

A method that can generate new outfits onto existing photos is developped so that it can 

1. retain the body shape and pose of the wearer,
2. roduce regions and the associated textures that conform to the language description, 
3. Enforce coherent visibility of body parts.

## METHOD ##

Given an input photograph of a person and a sentence description of a new desired outfit, the model first generates a segmentation map $\tilde{S}$ using the generator from the first GAN. Then the new image is rendered with another GAN, with the guidance from the segmentation map generated in the previous step. At test time, the final rendered image is obtained with a forward pass through the two GAN networks. The workflow of this work is shown in the following figure.

![Framework](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20171031_DeepFashion.png "Framework"){: .center-image .image-width-640}

The first generator $G_{shape}$ aims to generate the semantic segmentation map $\tilde{S}$ by conditioning on the spatial constraint $\downarrow m(S_0)$, the design coding $\textbf{d}$, and the Gaussian noise $\textbf{z}_S$.