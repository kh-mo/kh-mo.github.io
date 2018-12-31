---
layout: post
title: attention is all you need 논문 리뷰
category: Non-Category
---

본 포스트는 attention is all you need를 리뷰한 포스트입니다.
잘못된 해석이나 이해가 포함될 수 있으니 첨언은 언제나 환영합니다.

## The Transformer

![](/public/img/attention_is_all_you_need_figure1.JPG "Figure1 of attention is all you need 논문 리뷰")

위 그림은 transformer 모델의 전체적인 구조를 표현하고 있습니다.
기존의 seq2seq 모델들과는 달리 CNN과 RNN 아키텍처가 포함되어 있지 않은 구조라 다소 파격적입니다.
그러나 성능이 뛰어나고 또 여러가지 합리적인 이유를 근거로 하여 해당 구조가 나왔다고 할 수 있습니다.
한 부분씩 살펴보도록 하겠습니다.

## Embedding

기존의 seq2seq 모델들과 동일하게 input, output token을 받아 $d_{model}$ 차원 벡터로 변환해줍니다.
pytorch로 다음과 같이 구현 가능합니다.

<script src="https://gist.github.com/kh-mo/72919f0ecb434a0fe27551f880394f4e.js"></script>

nn.Embedding 함수를 이용하여 $d_{model}$ 차원을 가진 단어 갯수만큼의 embedding matrix가 생성됩니다.
해당 embedding 클래스 객체에서 token값을 이용하여 단어에 대한 벡터를 얻을 수 있습니다.

## Positional Encoding

RNN 또는 CNN 기반 seq2seq 모델들은 자연스럽게 단어가 입력되는 순서 정보를 받을 수 있었습니다.
RNN은 단어 토큰이 하나씩 들어가는 구조였고 CNN도 단어 묶음이 순차적으로 들어가게 되었지요.
그러나 transformer 모델은 문장을 통채로 입력으로 받고 있습니다.
이 때문에 단어가 문장내에서 위치하는 position, 순서정보를 추가로 반영하고 싶어 사용한 것이 positional encoding입니다.
사용된 수식은 다음과 같습니다.

$$ PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}}) $$

$$ PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}}) $$

여기서 pos는 문장 속 단어를 의미합니다.
그리고 i는 $d_{model}$ 차원 단어 벡터의 특정 포지션을 의미합니다.
pos x i 로 구성된 2차원 행렬이 positioanl encoding의 결과입니다.
이 행렬은 input sequence matrix와 동일한 크기를 가지기 때문에 요소들을 더하여 결과값을 냅니다.
아래와 같은 그림으로 표현할 수 있습니다. 

![](/public/img/attention_is_all_you_need_figure2.JPG "Figure2 of attention is all you need 논문 리뷰")

positional encoding은 다음과 같이 구현할 수 있습니다.

<script src="https://gist.github.com/kh-mo/6a774bba6ae97a507b80810351602584.js"></script>
