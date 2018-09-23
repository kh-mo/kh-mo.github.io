---
layout: post
title: 소프트맥스와 시그모이드 그래디언트 비교
category: Non-Category
---

RNN에 어텐션(attention)을 적용한 것은 resnet 모델에서 skip-connection을 연결한 것과 동일한 메커니즘으로 보입니다.
그런데 레이어가 많이 쌓일수록 모델의 성능이 높아지는 skip-connection과는 달리 어텐션은 추가적인 RNN 레이어마다 연결을 하더라도 성능 향상이 크지 않다고합니다.
그 이유를 어텐션은 소프트맥스(softmax)를 통해 가중치를 계산하는데 이 때 그래디언트(gradient)량이 상당히 줄어들기 때문으로 추정하고 있습니다.
특히 소프트맥스는 비선형함수(nonlinear function)에 비해 그래디언트가 상당히 작아진다고 하여 실제로 계산해보고 싶은 생각이 들었습니다.
그래서 본 포스트에선 크로스 엔트로피 손실함수(cross entropy loss function)에 대해서 대표적인 비선형함수인 시그모이드(sigmoid)와 소프트맥스(softmax)의 그래디언트를 한 번 비교해보겠습니다.


## 관련 개념

해당 그래디언트를 계산하기 전에 몇가지 정의를 짚고 넘어가겠습니다.
소프트맥스, 시그모이드, 크로스 엔트로피 함수의 형태와 미분 공식 몇가지를 정의하고 넘어가겠습니다.


* 소프트맥스

소프트맥스 수식 형태는 다음과 같습니다.

$$ p_i = \frac{exp(x_i)}{\sum_{k} exp(x_k)} $$

전체 분류해야 할 범주의 갯수는 k개가 됩니다. 이 함수의 입력값은 $x_i$가 되고 출력값은 $p_i$가 됩니다.
$p_i$는 0 ~ 1 사이의 값, 즉 확률을 가지게 되며 모든 k개 $p_i$의 합은 1이 됩니다.
<br>

* 시그모이드

시그모이드 수식 형태는 다음과 같습니다

$$ p_i = \frac{1}{1+exp(-x)} $$

$x$ 값이 커질수록 $p_i$ 값은 1에 가까워지며, $x$ 값이 작아질수록 $p_i$ 값이 0에 가까워지게 됩니다.
<br>

* 크로스 엔트로피

크로스 엔트로피 수식 형태는 다음과 같습니다.

$$ L = - \sum_{i} {y_i}\log p_i $$

$y_i$ 는 실제 정답을 나타내는 벡터입니다.
예를 들어 3범주 분류 문제에서 첫번째 범주가 정답이라면 $y = [1, 0, 0]$ 이 되고 $y_1$ 은 1, $y_2$ 는 0, $y_3$ 은 0이 되는 것입니다.
<br>

* 미분 공식

그래디언트 수식을 계산하는데 필요한 몇가지 미분 공식을 정리하고 넘어가겠습니다.

$$ y = exp(x) \implies \frac{dy}{dx} = exp(x) $$

$$ y = \log x \implies \frac{dy}{dx} = \frac{1}{x} $$

$$ y = \frac{f(x)}{g(x)} \implies \frac{dy}{dx} = \frac{f'(x)g(x)-f(x)g'(x)}{g(x)^2} $$


## Softmax graident

이제 소프트맥스의 그래디언트를 구해보겠습니다.
소프트맥스 그래디언트는 모든 x 변수에 대한 계산을 분모에 두고 분자에는 i번째 x만을 다루는 수식으로 구성되어 있습니다.
즉, i번째 x에 대한 그래디언트 값과 i이외의 x에 대한 그래디언트 값이 다르게 나올 수 있기 때문에 이를 구분해서 계산해야 합니다.
먼저 x가 i번째 입력값인 경우에 대해 살펴보겠습니다.

$$ \frac{d{p_i}}{d{x_i}} = \frac{d{\frac{exp(x_i)}{\sum_{k} exp(x_k)}}}{d{x_i}}$$


## Sigmoid gradient

