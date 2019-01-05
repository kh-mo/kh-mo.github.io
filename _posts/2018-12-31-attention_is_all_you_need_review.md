---
layout: post
title: attention is all you need 논문 리뷰
category: Non-Category
---

본 포스트는 attention is all you need를 리뷰한 포스트입니다.
잘못된 해석이나 이해가 포함될 수 있으니 첨언과 조언은 언제나 환영합니다.
포스트에 사용된 그림은 논문의 그림과 [블로그](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec), 그리고 [구글 자료](https://drive.google.com/file/d/0B8BcJC1Y8XqobGNBYVpteDdFOWc/view)를 참고했습니다.

## The Transformer

![](/public/img/attention_is_all_you_need_figure1.JPG "Figure1 of attention is all you need 논문 리뷰")

위 그림은 transformer 모델의 전체적인 구조를 보여줍니다.
기존의 seq2seq 모델들과는 달리 CNN과 RNN 아키텍처가 포함되어 있지 않은 구조라 다소 파격적입니다.
그러나 오로지 attention만으로 구성된 이 네트워크는 여러가지 합리적인 근거를 기반한 모델로 그 성능이 매우 뛰어납니다.
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
때문에 순차적으로 계산하는 과정을 손실하게 됩니다.
이 손실된 정보를 보정해주기 위해 도입된 것이 positional encoding입니다.
사용된 수식은 다음과 같습니다.

$$ PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}}) $$

$$ PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}}) $$

여기서 pos는 문장 속 단어를 의미합니다.
한 문장에 단어가 10개 있다면 pos는 1부터 10까지 값을 가지게 됩니다(python 구현 시 0부터 9까지).
그리고 i는 $d_{model}$ 차원 단어 벡터의 특정 포지션을 의미합니다.
논문에서 $d_{model}$은 512차원으로 구성되므로 i는 1부터 512까지 값을 가지게 됩니다.
수식을 따라 구현하면 pos by i 로 구성된 2차원 행렬을 만들 수 있고 이것이 positioanl encoding의 결과입니다.
이 행렬은 input sequence matrix와 동일한 크기를 가지기 때문에 요소들을 더하여 인코더와 디코더의 입력값으로 사용합니다.
아래와 같은 그림으로 표현할 수 있습니다. 

![](/public/img/attention_is_all_you_need_figure2.JPG "Figure2 of attention is all you need 논문 리뷰")

한 문장에 주어지는 단어의 갯수와 순서는 문장마다 다릅니다.
그러나 위의 수식에 따르면 단어 벡터의 차원은 항상 일정하고 단어의 position 또한 고정되어 있기 때문에 늘 같은 position encoding 벡터값을 가지게 됩니다.
다시 말하면 어떤 문장이 주어지더라도 첫번째 position encoding 벡터, 두번째 position encoding 벡터는 늘 일정한 값이 나타난다는 것입니다.
다만 그 방식에 주기를 가진 sin, cos 함수를 넣어 규칙성을 주었다고 할 수 있습니다.
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

pytorch 구현은 다음과 같습니다.

<script src="https://gist.github.com/kh-mo/e0a0116b01f3091c7b5146fd7abc3a66.js"></script>

## Scaled Dot-Product Attention

Attention은 특정 벡터가 다른 벡터와 얼마나 유사한지를 측정하여 가중합을 구하는 방식으로 진행됩니다.
대표적인 attention 사용 방식은 아래와 같습니다.

![](/public/img/attention_is_all_you_need_figure4.JPG "Figure4 of attention is all you need 논문 리뷰")

이 중 인코더에서는 왼쪽 아래 encoder self-attention 방식을 사용합니다.
인코더 입력 문장 속 단어들은 주변 모든 단어와의 유사도를 계산합니다.
이 연산은 내적(dot-product) 방식으로 수행됩니다.
수식 $Q{K^2}$은 내적을 통해 문장 속 모든 단어들의 유사도를 구하는 과정입니다.
논문에서는 이 방식이 훨씬 속도가 빠르고 효율적이기 때문에 사용했다고 언급하고 있습니다.

> dot-product attention is much faster and more space-efficient in practice

이 연산은 벡터의 내적이라 차원이 커질수록 그 값도 커지게 됩니다.

> assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, $ q \cdot k = \sum_{i=1}^{d_k} q_ik_i $, has mean 0 and variance $d_k$

따라서 $\sqrt{d_k}$로 정규화를 시켜줍니다.
Attention은 각 벡터에 일정 가중치를 곱해 가중합을 구하는 개념이기 때문에 가중치들을 확률값으로 변환하여 연산을 수행합니다.
이로서 아래와 같은 수식이 완성됩니다.

$$ Attention(Q, K, V) = softmax(\frac{Q{K^T}}{\sqrt{d_k}})V $$

그러나 디코더의 경우 문장을 생성하는 과정에서 뒤쪽 단어의 정보를 참조해서는 안됩니다.
위 그림의 오른쪽 아래를 참고하시기 바랍니다.
이것을 방지하기 위해서 앞쪽의 단어가 뒤에 나오는 단어와의 유사도를 구할 때 mask를 사용하여 참조를 방지합니다.
전체적인 연산 그림은 아래와 같습니다.

![](/public/img/attention_is_all_you_need_figure5.JPG "Figure5 of attention is all you need 논문 리뷰")

구현 과정은 아래와 같습니다.

<script src="https://gist.github.com/kh-mo/0776a177a5423eb039080e083e22f433.js"></script>

## Layer Normalization

## Feed-Forward Network

인코더, 디코더 레이어에서 multi-head attention 연산과 normalization 연산을 거치고나면 단순한 feed forward 연산을 수행하게 됩니다.
수식과 코드는 다음과 같습니다.

$$ FFN(x) = max(0, xW_1 + b_1)W_2+b_2 $$  

<script src="https://gist.github.com/kh-mo/24f1fbbbc9f4e3950d4be03d3fa367d3.js"></script>

## 

Transformer는 parallelizable하기 때문에 상당히 학습 시간이 짧습니다.
또한 당시 존재했던 번역 task의 sota를 달성할만큼 높은 성능도 보입니다.
앙상블이 아닌 single 모델로 달성한 성능이란 점도 주목할 요소 중 하나라고 생각합니다.
