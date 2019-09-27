---
layout: post
title: 오차의 최소화, Gradient Descent
category: Definition
---

딥러닝에서 사용되는 최적화 방법론인 gradient descent에 관한 내용입니다.
이 분야를 공부하는 사람이라면 누구나 다 알고 있는 내용이지만 그래도 한 번 정리하여 개념이 흔들릴 때 참고하고자 합니다.
이 포스트가 누군가에게 도움이 되기를 바라고 혹시 잘못된 내용이 있다면 첨언해 주시길 부탁드립니다.
포스팅 작성에 도움을 받은 참고 [위키독스](https://wikidocs.net/6998)와 [블로그](https://darkpgmr.tistory.com/133)는 링크를 걸어두었습니다.

## what is the gradient?

다변수 함수 $f$가 $n$개 변수로 이루어져 있다면 다음과 같이 표현할 수 있습니다.

$$ f(x_1, x_2, ..., x_n) $$

이 때 그래디언드(gradient)는 함수 $f$를 각 변수로 편미분(Partial derivative)한 값을 원소로 하는 벡터라고 정의할 수 있습니다.

$$ \nabla f=(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}) $$

```
### TMI

**미분(d)** - 순간적인 변화율을 나타내는 개념
**편미분($ \partial $)** - 다변수함수에서 하나의 변수에 주목하고 나머지 변수의 값을 고정시켜 놓고 그 변수로 미분하는 일
**전미분(D)** - 변수의 미소한 변화에 따른 다변함수의 함수값
```

이 벡터는 기울기가 가장 가파른 곳으로의 방향, $f$값이 가장 가파르게 증가하는 방향을 의미합니다.

예를 들어, $f(x, y) = x^2 + y^2 + xy$라고 하면 그래디언트는 $\nabla f=(2x+y, 2y+x)$입니다.
임의의 점 $(1, 3)$에서 함수 $f$ 값이 최대로 증가하는 방향은 $(5, 7)$이고 그 기울기(벡터의 크기, 증가의 가파른 정도)는 $\lVert (5, 7) \rVert = \sqrt{5^2 + 7^2} = \sqrt{74}$입니다.

## gradient descent

그래디언트는 함수값을 최대화시키는 방향을 의미한다는 점을 위에서 이야기했습니다.
그래디언트 방향을 반대로 하면 어떻게 될까요?
어떤 함수값을 최소화시키는 방향을 나타내게 될 것입니다.
함수값을 최소화시키는 방향으로 나아가 최적값을 찾는 방법론, 이것이 gradient descent입니다.
머신러닝 분야에서 gradient descent 방법론이 많이 사용되는데요.
그 이유는 analytic하게 답을 구할 수 없는 경우 점진적으로 문제를 풀어나가야 하며 gradient descent는 그 방식에 알맞는 알고리즘이기 때문입니다.
수식으로 알고리즘을 좀 더 살펴보겠습니다.

$$ x_{n+1} = x_n - \alpha \nabla_x f $$

$x_{n+1}$은 이전의 $x_n$에서 일정 값을 빼주는 것을 수식에서 확인할 수 있습니다.
우선 각 요소를 살펴보면 $\nabla_x f$는 함수 $f$를 변수 $x$로 편미분한 값이고 $\alpha$는 learning rate입니다.
전체 함수값을 최소화시키기 위해 $x$의 변화량을 나타내는 것이 $\nabla_x f$이고 그 비율을 조절하는 역할이 learning rate입니다.
Learning rate를 조절해서 변수 $x$가 급격하게 변해 optimal point를 지나치는 것을 방지할 수 있습니다.
현재 위치에서 그래디언트 반대 방향(음수 방향)으로 함수를 업데이트 했기 때문에 전체 함수는 최소값을 찾아가게 됩니다.
만약 함수 최대값을 찾아가기 위해 gradient ascent방법론을 수행하고자 한다면 다음과 같은 수식을 활용하면 됩니다.

$$ x_{n+1} = x_n + \alpha \nabla_x f $$

여기까지 어떻게 그래디언트 값을 활용해서 변수값을 업데이트 하는지 확인했습니다.
그러나 무한정 학습하는 것은 아닙니다.
Gradient descent 방법은 점진적으로 해를 찾아가는 numerical, iterative 방식이기 때문에 어느 시점에서 학습을 멈춰야합니다.
또 때로는 예기치 못하게 발산할 가능성도 있습니다.
따라서 다음과 같은 조건을 만족할 경우에만 변수를 업데이트하고 학습을 진행합니다.

$$ L(\theta + \nabla \theta) \leq L(\theta) $$

업데이트 된 변수를 기반으로 구한 함수 $L$이 업데이트 되기 이전의 변수로 구한 함수값보다 작거나 같을 경우에만 변수를 업데이트 합니다(gradient descent의 경우).
이 조건하에서 진행되는 학습은 결국 gradient descent방법론이 최적해를 찾아가는 과정입니다.

<script src="https://gist.github.com/kh-mo/fbecdd96c163b895b5123571fe63d8c1.js"></script>

## backpropagation



## pytorch backpropagation example
