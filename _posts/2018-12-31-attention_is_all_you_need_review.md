---
layout: post
title: attention is all you need 논문 리뷰
category: Non-Category
---

본 포스트는 attention is all you need를 리뷰한 포스트입니다.
잘못된 해석이나 이해가 포함될 수 있으니 첨언과 조언은 언제나 환영합니다.
포스트에 사용된 그림은 논문의 그림과 [이 블로그](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)를 참고했습니다.

## The Transformer

![](/public/img/attention_is_all_you_need_figure1.JPG "Figure1 of attention is all you need 논문 리뷰")

위 그림은 transformer 모델의 전체적인 구조를 보여줍니다.
기존의 seq2seq 모델들과는 달리 CNN과 RNN 아키텍처가 포함되어 있지 않은 구조라 다소 파격적입니다.
그러나 여러가지 합리적인 근거를 기반한 모델로 그 성능이 매우 뛰어납니다.
한 부분씩 살펴보도록 하겠습니다.

## Embedding

기존의 seq2seq 모델들과 동일하게 input, output token을 받아 $d_{model}$ 차원 벡터로 변환하는 것을 목적으로 하는 단계입니다.
pytorch로 다음과 같이 구현 가능합니다.

<script src="https://gist.github.com/kh-mo/72919f0ecb434a0fe27551f880394f4e.js"></script>

nn.Embedding 함수를 이용하여 $d_{model}$ 차원을 가진 단어 갯수만큼의 embedding matrix가 생성됩니다.
해당 embedding 클래스 객체에서 token값을 이용하여 단어에 대한 벡터를 얻을 수 있습니다.

## Positional Encoding

RNN 또는 CNN 기반 seq2seq 모델들은 자연스럽게 단어가 입력되는 순서 정보를 받을 수 있었습니다.
RNN은 단어 토큰이 하나씩 들어가는 구조였고 CNN도 단어 묶음이 순차적으로 들어갔습니다.
그러나 transformer 모델은 문장 전체를 입력으로 받습니다.
때문에 순차적으로 계산하는 과정을 손실하게됩니다.
이 손실된 정보를 보정해주기 위해 도입된 것이 positional encoding입니다.
사용된 수식은 다음과 같습니다.

$$ PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}}) $$

$$ PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}}) $$

여기서 pos는 문장 속 단어를 의미합니다.
한 문장에 단어가 10개 있다면 pos는 1부터 10까지 값을 가지게 됩니다(python 구현시 0부터 9까지).
그리고 i는 $d_{model}$ 차원 단어 벡터의 특정 포지션을 의미합니다.
논문에서 $d_{model}$은 512차원으로 구성되므로 i는 1부터 512까지 값을 가지게 됩니다.
수식을 따라 구현하면 pos by i 로 구성된 2차원 행렬을 만들 수 있고 이것이 positioanl encoding의 결과입니다.
이 행렬은 input sequence matrix와 동일한 크기를 가지기 때문에 요소들을 더하여 인코더와 디코더의 입력값으로 사용합니다.
아래와 같은 그림으로 표현할 수 있습니다. 

![](/public/img/attention_is_all_you_need_figure2.JPG "Figure2 of attention is all you need 논문 리뷰")

positional encoding 구현은 다음과 같습니다.

<script src="https://gist.github.com/kh-mo/6a774bba6ae97a507b80810351602584.js"></script>

## Multi-Head Attention

Transformer 논문을 처음 접할 때 가장 이해하기 어려운 부분은 V와 K와 Q가 무엇인지 파악하는 것입니다.
논문의 3.2.3절 설명에 따르면 인코더의 V, K, Q는 모두 동일한 값으로 positional encoding을 통해 들어온 입력값 또는 이전 레이어의 출력값이 됩니다.
디코더의 경우 Masked Multi-Head Attention의 V, K, Q는 인코더와 동일하게 positional encoding을 통한 입력값 또는 이전 레이어 출력값입니다.
하지만 디코더 레이어 블록의 중간에 있는 Multi-Head Attention의 경우 Q는 디코더 sublayer의 출력값이고 V, K는 인코더의 출력값입니다.
다소 복잡할 수 있기 때문에 주의를 기울여야 합니다.

인코더에서 Multi-Head Attention은 3개 입력값 Q, K, V를 받습니다.
실제로 같은 2차원 행렬인 Q, K, V는 각각 선형 연산을 통해 다른 2차원 행렬로 변환되게 됩니다.
각 Q, K, V에 대해 총 h번 선형 연산이 수행되는데, 그 결과 각각 다른 벡터값을 얻어 연산을 수행하게 됩니다.
이 과정은 parallel하게 이루어질 수 있습니다.
Q, K, V로 구성된 h개 집합은 Scaled Dot-Product Attention 연산을 통해 h개 head를 생성합니다.
Scaled Dot-Product Attention 연산은 나중에 설명하도록 하겠습니다.
h개 head는 결합되어 다시 선형 연산을 통해 $d_{model}$ 차원의 벡터를 가지게 됩니다.
이렇게 h번 연산을 수행하여 다시 결합하는 이유를 논문에서는 다음과 같이 설명하고 있습니다.

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

저는 이것을 h개의 여러 subspace로 입력 문장 정보를 입력하여 평균화하는 것이 더 풍부한 정보를 문장으로부터 추출할 수 있다고 해석하였습니다.
단일 subspace를 사용하는 것보다 더 많은 정보를 얻을 수 있는 구조라고 생각됩니다(그저 블로거의 뇌피셜일 뿐입니다).
이를 나타내는 수식과 그림은 아래와 같습니다.
 
![](/public/img/attention_is_all_you_need_figure3.JPG "Figure3 of attention is all you need 논문 리뷰")
