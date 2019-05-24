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
2006년에는 G. Hinton과 그의 공저자들이 성공적으로 deep neural network를 차원 축소에 적용했고 Deep Learning의 개념을 제안했습니다.
오늘날 높은 효과를 보이기에 deep learning 알고리즘은 인공지능을 넘어 많은 분야에서 사용되고 있습니다.
