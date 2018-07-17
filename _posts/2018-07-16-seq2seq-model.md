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

![](/public/img/seq2seq-model-figure1.JPG "Figure1 of Sequence to Sequence Learning with Neural Networks")

이 그림에서 입력 시퀀스는 A,B,C,\<EOS\>가 되며 출력 시퀀스는 W,X,Y,Z,\<EOS\>가 됩니다.
이는 우리가 어떤 언어를 번역할 때 단어가 1:1로 매핑되어 완벽하게 같은 시퀀스로 입력과 출력이 나타나지 않는 것으로 상상할 수 있습니다.
위의 그림에서 보듯 입력과 출력 시퀀스는 처리해야 할 길이가 다르기 때문에 입력 정보를 처리하는 인코더 LSTM과 출력 정보를 처리하는 디코더 LSTM을 사용합니다.
인코더가 매 타임스탭마다 정보를 하나씩 받아 처리하여 최종적으로 \<EOS\> 정보를 처리하면 LSTM 아키텍처는 입력 정보를 벡터 하나로 표현하게 됩니다.
디코더는 이 벡터를 받아 매 타임스탭마다 출력해야 할 단어를 예측합니다.
그리고 마지막으로 \<EOS\>를 출력하면 예측이 끝나게 됩니다.
이 과정은 한 시퀀스를 다른 시퀀스에 매핑시키는 종단간(end to end) 학습으로 이루어지게 됩니다.

이번엔 수식적으로 정의해 보겠습니다.
Seq2seq 모델을 학습하기 위해서 우리는 목적함수를 다음과 같이 정의할 수 있습니다.

$$p(y_1,...,y_{T'}|x_1,...,x_{T})$$

여기서 $(x_1,...,x_T)$는 입력 시퀀스를 $(y_1,...,y_{T'})$ 는 출력 시퀀스이며 T'과 T를 구별하는 것은 입력과 출력 시퀀스 길이가 다름을 의미합니다.
인코더 LSTM은 입력 시퀀스를 받아 고정된 길이의 벡터 v를 만듭니다.
