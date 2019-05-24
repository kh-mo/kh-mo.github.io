---
layout: post
title: An Overview on Data Representation Learning From Traditional Feature Learning to Recent Deep Learning 번역
category: Representation-Learning
---

## Abstract

약 100년 전부터, 본질적인 데이터 구조를 학습하기 위해서, 많은 representation learning 방법론이 제안되어왔고, 그 안에는 linear와 nonlinear, supervised와 unsupervised가 포함되어 있습니다.
특히, deep architecture는 최근 representation learning에 광범위하게 적용되었고, image classification, object detection, speech recognition과 같은 많은 task에 최고의 결과를 가져다주었습니다. 
이 논문에서, 우리는 data representation learning 방법론들의 개발과정을 리뷰합니다.
특히, 우리는 전통적인 feature learning 알고리즘과 sota 딥러닝 모델을 조사했습니다.
Data representation의 역사와 유용한 resource와 toolbox가 소개될 겁니다.
마지막으로, 우리는 data representation learning에 관한 몇가지 흥미로운 연구 방향과 주목할만한 점을 언급하며 이 논문을 마무리합니다.

## Introduction

Artificial intelligence, bioinformatics, finance와 같은 많은 도메인에서 data representation learning은 이어 설계되는 classification, retrieval, recommendation task를 가능하게 하는 중요한 단계입니다.
일반적으로, 큰 규모의 application에 대해, 어떻게 본질적인 데이터 구조를 학습하고 데이터로부터 가치있는 정보를 발견할 수 있는지가 점점 더 중요하고 도전적인 과제가 됩니다.

약 100년 전부터, 많은 데이터 representation learning 방법론이 제안되어 왔습니다.
그 중에서도, PCA(principal component analysis)는 1901년에 K.Pearson이 제안했고, LDA(linear discriminant analysis)는 1936년에 R.Fisher가 제안했습니다.
PCA와 LDA는 선형 방법론입니다.
그렇기는 하지만, PCA는 unsupervised 방법론이고 LDA는 supervised 방법론입니다.
PCA와 LDA에 기반하여 kernel PCA, GDA(generalized discriminant analysis)와 같은 다양한 확장된 방법론들이 제안되었습니다.
2000년에는 머신러닝 커뮤니티에서 고차원 데이터의 본질적인 구조를 발견하는 manifold learning에 대한 연구가 시작되었습니다.
PCA나 LDA같은 이전의 global 접근법과는 달리, Isomap(isometric feature mapping)과 LLE(locally linear embedding)같은 manifold learning 방법론은 일반적으로 locality 기반입니다.
2006년에는 G.Hinton과 그의 공저자들이 성공적으로 deep neural network를 차원 축소에 적용했고 Deep Learning의 개념을 제안했습니다.
오늘날 높은 효과를 보이기에 deep learning 알고리즘은 인공지능을 넘어 많은 분야에서 사용되고 있습니다.

반면에 artificial neural network에 대한 연구는 많은 성공과 어려움을 겪어왔습니다.
1943년에 W.McCulloch와 W.Pitts가 신경망을 위한, 후속 연구에서 M-P 모델이라고 불리는 첫 artificial neuron, linear threshold을 만들었습니다.
이후에 D.Hebb은 Hebbian 이론이라고 불리는 신경 가소성(neural plasticity-새로운 환경에 뇌가 적응해가는 능력) 메카니즘에 기반한 hypothesis of learning을 제안했습니다.
근본적으로, M-P 모델과 Hebbian 이론은 neural network 연구와 인공지능 분야의 development of connectionism을 위한 상황을 조성했습니다.
1958년 F.Rosenblatt은 binary classification을 위한 two-layer neural network인 perceptron을 만들었습니다.
그러나 M.Minsky와 S.Papert는 perceptron이 XOR(exclusive-or) 문제를 풀 수 없다는 점을 지적했습니다.
1974년 P.Werbos가 MLP(multi-layer perceptrons)을 학습하기 위한 back propagation 알고리즘을 제안할때까지 neural network 연구는 침체되었습니다.
특히, 1986년 D.Rumelhart와 G.Hinton, R.Williams은 back propagation 알고리즘이 neural network의 hidden layer안에서 유용한 내부적 representation을 생성할 수 있다는 것을 발견했습니다.
Back propagation 알고리즘을 이용하면 이론적으로 neural network의 많은 layer를 학습할 수 있지만 2가지 중요한 문제점이 있습니다.
모델의 과적합(overfitting)과 기울기 발산(gradient diffusion)이 그 문제점입니다.
2006년 G.Hinton은 deep neuural network의 finetuing과 greedy layer별 pre-training을 수행하는 아이디어를 결합해 representation learning 연구에 돌파구를 시도했습니다.
Neural network community를 혼란스럽게한 이슈들은 따라서 해결되었습니다.
이후에 많은 deep learning 알고리즘들이 제안되었고 성공적으로 다양한 도메인에 적용되었습니다.

이 논문에서 우리는 전통적인 feature learning과 최근 deep learning 둘 모두에 관한 data representation learning의 개발과정을 리뷰합니다.
이 논문의 나머지는 다음과 같이 구성되어 있습니다.
2장에서는 선형 알고리즘, 선형 알고리즘의 kernel 확장 알고리즘, manifold learning 방법론과 같은 전통적인 feature learning을 다룹니다.
3장에서는 최근 deep learning의 진보와 중요 모델, tookbox에 대해 다룹니다.
4장에서는 data representation learning에 대한 흥미로운 연구 방향에 대해서 결론을 내립니다.

## Traditional feature learning

이번 장에서 우리는 분류기나 다른 예측기를 만들 때 유용한 정보를 훨씬 쉽게 추출할 수 있는 데이터 변형을 학습하는 것을 목적으로 하는 shallow 모델이 속한 전통적인 feature learning 알고리즘에 집중합니다.
이런 이유로 우리는 SIFT(scale-invariant feature transform), LBP(local binary pattern), HOG(histogram of oriented gradient)와 같은 image descriptor나 TF-IDF(term frequency-inverse document frequency)와 같은 document statistic 계열의 manual feature engineering 방법론을 고려하지 않을 것입니다.

그것들을 공식화하는 관점에서, 알고리즘은 일반적으로 선형-비선형, supervised-unsupervised, generative-discriminative, global-local과 같은 유형으로 구분됩니다.
예를들어 PCA는 선형이자 unsupervised, generative, global feature learning 방법론이고 반면에 LDA는 선형이자 supervised, discriminative, global 방법론입니다.
이번 장에서 우리는 feature learning 알고리즘들을 global인 것과 local한 것으로 범주화한 분류체계를 적용할 것입니다.
일반적으로 global 방법론은 학습된 feature space에서 데이터의 global information을 보존하려하고 local 방법론은 새로운 representation을 학습하는 동안 데이터 사이의 local similarity를 보존하는 것에 집중합니다.
예를들어 PCA와 LDA와 달리, LLE는 locality 기반의 feature learning 알고리즘입니다.
게다가 우리는 manifold learning으로 locality 기반의 feature learning을 부르는데 고차원 데이터에서 manifold structure hidden을 발견하기 때문입니다.

문헌에서 Van der Maaten, Postma, Van den Herik는 34개 feature learning 알고리즘의 코드를 포함한 차원축소를 위한 MATLAB toolbox를 제공합니다.
Yan et al.의 페이퍼에서 많은 차원 축소 알고리즘군을 하나의 공식으로 통합한 graph embedding이라고 알려진 일반적인 프레임워크를 제안합니다.
Zhong, Chherawala, Cheriet의 페이퍼는 handwriting recognition에 대한 3종류 supervised 차원축소 방법론을 비교했다.
그동안에 Zhong, Cheriet의 페이퍼는 tensor로서 입력 데이터를 고려하는 tensor representation learning 관점에서 프레임워크를 제안했는데 이는 많은 선형, kernel, tensor 차원축소 방법론을 한 학습 기준으로 통합했다.

### Global feature learning

