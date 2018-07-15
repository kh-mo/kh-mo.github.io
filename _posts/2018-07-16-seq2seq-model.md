---
layout: post
title: sequence to sequence 모델
category: Translation
---

## Meta Information
* 제목 : Sequence to Sequence Learning with Neural Networks
* 2014.09 : arXiv submit
* 2014.12  : NIPS

## 연구 동기
이번에 소개할 논문은 시퀀스(sequence)가 있는 데이터를 처리하는 대표적인 딥러닝 모델인 sequence to sequence 모델(이하 seq2seq)입니다.
Deep Neural Network(DNN)가 레이블 된 데이터를 충분히 가지고 있을 경우 여러 AI 어플리케이션에서 높은 성능을 보이지만, 가장 큰 한계점으로 여겨진 부분은 입력과 출력이 고정된 크기의 벡터로 표현되어야 한다는 것입니다.
그러나 언어처럼 매 순간 입력 길이가 달라져 변동성이 큰 데이터는 DNN이 처리하기 매우 까다롭습니다.
즉, DNN을 이용해서 machine translation, speech recognition, question answering과 같은 문제를 푸는 것은 한계가 있습니다.
그렇기 때문에 이 논문에서는 시퀀스가 데이터를 가진 어떤 문제가 주어졌을 때, 이를 학습할 수 있는 방법론을 제안하는 것을 목적으로 했고 그것이 seq2seq 모델입니다.

## 모델 구조
시퀀스가 있는 정보를 처리하기 위해 이 논문이 사용한 아키텍처는 Long Short Term Memory(LSTM)입니다.
LSTM은 기존의 vanilla RNN보다 긴 입력 시퀀스를 잘 처리하고  vanishing gradient 문제도 상대적으로 많이 줄어든 아키텍처입니다.
Seq2Seq 모델은 LSTM 아키텍처 2개를 사용해 하나는 인코더(encoder), 하나는 디코더(decoder)로 사용하는 모델이 되겠습니다.
이런 구성을 취한 이유는 이 논문이 도전했던 문제가 입력과 출력 시퀀스 길이가 다른 machine translation 문제였기 때문입니다.
논문에서 나타난 그림을 보면 좀 더 이해에 도움이 될 것 같습니다.
![Figure1 of Sequence to Sequence Learning with Neural Networks](/public/img/seq2seq-model-figure1.jpg "이미지제목")
