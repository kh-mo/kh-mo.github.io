---
layout: post
title: a neural conversational model 논문 리뷰
category: Non-Category
---

본 포스트는 a neural conversational model을 리뷰한 포스트입니다.
잘못된 해석이나 이해가 포함될 수 있으니 첨언과 조언은 언제나 환영합니다.

## Conversational modeling

사람과 대화할 수 있는 기계를 만드려는 시도는 예전부터 지속되어 왔습니다.
비행기 티켓 예약같은 특정 도메인 문제를 풀려는 대화 시스템, 잡담을 나누는 chit-chat, 전반적인 대화를 나누는 general conversation 모델 시스템 등이 있습니다.
만들고자 하는 대화 시스템의 목적에 따라 rule 기반 대화 시스템은 나름 괜찮은 성능을 보여왔습니다.
그러나 대화 시나리오를 추가하거나 대화 결과가 변경되어야 할 경우 rule 기반 대화 시스템은 수정해야 할 공수가 너무 많이 드는 문제가 발생할 수 있습니다.
또한 대화를 위해 intent와 entity를 찾는 등 hand-crafted rule을 지정하는 것도 매우 번거로운 작업이 됩니다.
 

## seq2seq framework for conversational modeling 

context가 주어졌을 때, 확률적 모델링을 통해서 응답을 생성해내는 generative modeling 쪽에 가깝다.
후보군에서 고르는 문제는 ranking based model 쪽이다.
'greedy' inference approach는 예측한 단어를 다음 디코더 입력으로 사용하는 방식.

seq2seq는 간단하고 보편적이라 강점이 있다.
논문 모델에서 task 중심 데이터는 기계가 묻고 사람이 답하고 기계가 답을 준다.
일반 데이터에서는 사람이 묻고 기계는 짧게 답한다.


## 해결해야 할 문제점

대부분 nlp 문제가 그렇듯 대화 시스템을 올바르게 평가할 objective function을 정의하기가 어려움.
필요한 적절한 지식을 대화에서 추론해야 함.
domain specific 경우 도메인 지식을 잘 축적하고 그 안에서 정답을 찾아내야 함.
general domain 경우 common sense를 모델이 알고 있어야 함.
대화가 consistency 속성을 지니고 있어야 함.
사람의 대화는 일반적으로 장기간(메모리를 지니고 있어야)이며 다음 발화 예측보다는 정보 교환을 기반으로 한다.
사용중인 성능평가지표 : perplexity, 사람이 평가(4명이서 다수결)
짧은 답변, 불만족스러운 답변을 한다.

## 뇌피셜

번역 task는 한 문장과 다른 언어 문장 사이에 관계를 표현.
문장 by 문장 구조에서 어느정도 단어 배열에 대한 formulating의 정해짐.
그러나 대화는 문장이 주어지고 다음 문장이 나타날 때 확률은 사실상 무한대에 가까움.
더 큰 문장 확률분포가 주어지는데 이러면 어떻게 문제를 풀 수 있지?

번역의 경우 단어 배열에 대한 seq2seq 모델이 주어졌을 때, 특정 단어는 특정 의미로 번역되어야 하는 
external knowledge를 활용한 번역방식을 찾아보자.

그 정보가 곧 외부 정보를 입력으로 받아 처리하는 목적 지향 대화 시스템에 녹일 수 있는 정보라고 생각된다.

추가적으로 강화학습을 통해 특정 대화에는 특정 정답을 내도록 가이드 해줄 수 있을 것으로 기대한다.




















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

> dot-product attention is much faster and more space-efficient in practice.

이 연산은 벡터의 내적이라 차원이 커질수록 그 값도 커지게 됩니다.

> assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, $ q \cdot k = \sum_{i=1}^{d_k} q_ik_i $, has mean 0 and variance $d_k$.

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

## 정리하며...

지금까지 transformer의 아키텍쳐 building block과 그 의미들을 살펴보았습니다.
Transformer는 parallelizable하기 때문에 상당히 학습 시간이 짧고 당시 존재했던 번역 task의 sota를 달성할만큼 높은 성능도 보입니다.
앙상블이 아닌 single 모델로 달성한 성능이란 점도 주목할 요소 중 하나라고 생각합니다.
전체 아키텍처 구조를 작성한 코드는 [여기](https://github.com/kh-mo/Transformer)를 참고해주시기 바랍니다.
감사합니다.
