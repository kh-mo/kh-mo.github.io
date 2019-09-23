---
layout: post
title: Pruning Research
category: Non-Category
---

Neural Network가 computer vision, speech recognition, natural language processing 분야에서 좋은 성능을 보이고 있습니다.
각 분야에서 주어지는 task에 딥러닝을 적용하여 성능을 높이는 문제 이외에도 왜 이 모델이 성능을 잘 내는지 판단하는 문제(explainable AI), 더 효율적인 모델은 없는지 탐색하는 문제(AutoML, pruning, distilation)도 존재합니다.
이 포스트에서는 효율적인 모델을 탐색하기 위한 pruning의 여러 접근방법론을 살펴보려고 합니다.

## pruning의 효율성 문제

많은 페이퍼에서 언급하는 pruning문제를 풀어야 하는 이유는 딥러닝 모델이 너무 크다는 점입니다(over-parameterization).
물론 이는 사실입니다.
최근(18~19년도) sota 성능의 딥러닝 모델들은 대규모 GPU 시스템을 필요로 하는 경우가 많습니다.
On device, embedded system에 적용하기에는 부담스러운 것이 사실입니다.

=> 수많은 paper의 introduction에서 언급하는 문제점

## Approach

[Learning both weights and connections for efficient neural network, NIPS 2015](https://arxiv.org/pdf/1506.02626.pdf)

네트워크를 학습하고 중요하지 않은 연결을 끊은 다음에 재학습(retrain)을 시킨다.
장점 : sparse한 모델을 얻을 수 있다.
단점 : 재학습 비용이 크다.

[ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression, ICCV 2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_ThiNet_A_Filter_ICCV_2017_paper.pdf)

장점 : 모델이 차지하는 메모리 공간 내 크기를 효과적으로 줄일 수 있다(모델 사이즈가 작아진다).
단점 : 삭제하려는 channel을 greedy algorithm으로 선택하기 때문에 계산 복잡도가 매우 크다.

[Variational Convolutional Neural Network Pruning, CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.pdf)


## Research LAB


**Remove Redundancy(reduce network complexity)**

[paper](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37631.pdf) : A fixed-point implementation with 8-bit integer(vs 32-bit floating point) activations.

[paper](https://arxiv.org/abs/1404.0736) : The linear structure of the neural network by finding an appropriate low-rank approximation of the parameters.

[paper](https://arxiv.org/abs/1412.6115) : Compressed deep convnets using vector quantization.

**"Fully Connected" to "global average pooling"(reduce over-fitting)** 

[paper](https://arxiv.org/abs/1409.4842) : Reduce the number of parameters of neural networks by replacing FC to GAP.

**Network Pruning**

[paper](https://papers.nips.cc/paper/156-comparing-biases-for-minimal-network-construction-with-back-propagation.pdf) : Biased weight decay.

[paper](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf) : Optimal Brain Damage. Reduce the number of connections based on the Hessian of the loss function.

[paper](https://papers.nips.cc/paper/647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon.pdf) : Optimal Brain Surgeon. Reduce the number of connections based on the Hessian of the loss function.

[paper](https://arxiv.org/pdf/1504.04788.pdf) : Reduce model sizes by using a hash function to randomly group connection weights into hash bucket.


