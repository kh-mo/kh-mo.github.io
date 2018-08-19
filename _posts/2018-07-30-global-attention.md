---
layout: post
title: 입력 시퀀스를 고려한 번역 모델, Global Attention
category: Translation
---

## Meta Information
* 제목 : Neural Machine Translation by Jointly Learning to Align and Translate
* 2014.09 : arXiv submit
* 2015.05 : ICLR

## 연구 동기
인코더 디코더 기반의 신경망 네트워크가 번역 모델에서 높은 성능을 낼 수 있는 가능성을 보이면서 더 나은 성능을 얻기 위한 여러 연구가 진행되었습니다.
오늘 소개할 논문은 디코더가 출력할 단어를 예측하는 과정에서 가장 관련성이 높은 입력 문장의 부분을 찾는 모델을 제안했습니다.
기본적인 인코더 디코더 모델은 RNN 아키텍처를 사용하여 입력 문장을 순차적으로 인코딩한 후, 고정된 길이의 벡터(fixed-length vector)를 출력합니다.
그리고 디코더는 이 벡터를 받아 번역 디코딩을 시작합니다.
그러나 고정된 길이의 벡터는 입력 문장의 모든 정보를 벡터 하나로 압축해야 하기 때문에 효율적이지 못합니다.
이 문제는 번역해야 하는 문장이 길어질수록 더 심화됩니다.
본 논문이 이를 해결하기 위해 사용한 방법은 디코더가 다음 단어를 예측할 때, 인코더에서 출력되는 모든 단어 정보 중 가장 관련성이 높은 단어를 찾아 사용하는 방법입니다.
이것이 전역 어텐션(global attention) 방법론입니다.

## 모델 구조
기존의 seq2seq 모델에서 사용된 context vector는 인코더의 마지막 hidden state였습니다.
그러나 이 벡터는 인코더 입력 정보를 한 벡터로 압축해야 하기 때문에 한계가 있습니다.
본 논문에서 제안하는 방식은 인코더의 마지막 hidden state만 쓰는 것이 아니라 인코더의 모든 hidden state를 사용합니다.
논문에서는 이를 다음과 같은 수식으로 표현합니다.

$$c = q(\{h_1,...,h_{T_x}\})$$

$h_1$은 인코더의 첫 번째 hidden state, $h_{T_x}$는 인코더의 $T_x$번째 hidden state를 의미합니다.
이 $T_x$개 인코더 hidden state는 각각 매 순간 입력된 단어에 대한 정보를 담고 있습니다.
디코더가 RNN의 타임 스탭에 따라 한 단어를 생성할 때 마다 이 모든 인코더의 state와 비교하여 얼마나 정보가 잘 매칭 되는지를 계산하는 것이 attention 모델입니다.
그 계산 수식은 아래와 같은 과정을 따르게 됩니다.

$$e_{ij} = a(s_{i-1}, h_j)$$

디코더가 $s_i$번 째 단어를 예측할 때 $s_{i-1}$번째 디코더 hidden state와 모든 j개 인코더 hidden state를 a라는 함수를 이용하여 매칭합니다.
이 a 함수를 alignment model 이라고 부르며 이것은 디코더와 인코더 hidden state 사이의 유사도를 구하는 개념으로 볼 수 있습니다.
이렇게 $e_{ij}$를 구했으면 이를 softmax 함수를 사용하여 0~1사이로 정규화하여 가중치 형태로 변화시킵니다.
다음과 같은 수식을 사용합니다.

$$\alpha_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x} exp(e_{ik})}$$

그리고 이 가중치들을 인코더의 hidden state들과 가중합을 해줌으로써 디코더 i번째 타임 스탭에 대한 context vector를 얻을 수 있습니다.
수식은 다음과 같습니다.

$$c_{i} = \sum_{j=1}^{T_x} \alpha_{ij}h_{j}$$

수식으로 설명한 모델 구조는 다소 이해하기 어려울 수 있기 때문에 그림으로 한번 더 살펴보겠습니다.
우선 본 논문에 나온 아키텍처를 설명하는 그림은 다음과 같습니다.

<center>
<img src="/public/img/global-attention-figure1.JPG" width="40%", alt="Figure1 of Neural Machine Translation by Jointly Learning to Align and Translate">
</center>

이 그림에서 $X_1, X_2, X_3, X_T$는 인코더 입력 단어들입니다.
그리고 바로 위 상자에서 $\overrightarrow{h_1}, \overleftarrow{h_1}$이 표현되는데 이는 정방향 RNN과 역방향 RNN을 의미합니다.
본 아키텍처는 bidirectional RNN을 사용하기 때문에 그림에서 이렇게 나타내고 있는 것입니다.
$\overrightarrow{h_1}, \overleftarrow{h_1}$을 합친 것이 위 수식에서 나타내고 있던 인코더의 $h_1$ hidden state 입니다.
이 hidden state들과 그림 맨 위의 $S_{t-1}$의 사이에

## 성능 개선을 위한 테크닉

## 데이터 셋

## 결과

## 블로거 의견
attention 개념이 NLP에 잘 적용된 대표적 논문으로서 가치를 지니고 있다고 생각합니다.
그리고 본 포스트에는 저의 주관적인 판단과 해석으로 작성되었기 때문에 부정확한 부분이 있을 수 있습니다.
자신만의 견해, 정확한 이해를 위해서 논문을 직접 읽어보시는 것을 권유드립니다.
감사합니다.
