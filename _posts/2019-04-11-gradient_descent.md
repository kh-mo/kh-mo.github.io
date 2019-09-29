---
layout: post
title: 오차의 최소화, Gradient Descent
category: Definition
---

딥러닝에서 사용되는 최적화 방법론인 gradient descent에 관한 내용입니다.
이 분야를 공부하는 사람이라면 누구나 다 알고 있는 내용이지만 그래도 한 번 정리하여 개념이 흔들릴 때 참고하고자 합니다.
이 포스트가 누군가에게 도움이 되기를 바라고 혹시 잘못된 내용이 있다면 첨언해 주시길 부탁드립니다.
포스팅 작성에 도움을 받은 참고 [위키독스](https://wikidocs.net/6998)와 [위키피디아](https://en.wikipedia.org/wiki/Backpropagation), [블로그](https://darkpgmr.tistory.com/133)는 링크를 걸어두었습니다.

## what is the gradient?

다변수 함수 $f$가 $n$개 변수로 이루어져 있다면 다음과 같이 표현할 수 있습니다.

$$ f(x_1, x_2, ..., x_n) $$

이 때 그래디언드(gradient)는 함수 $f$를 각 변수로 편미분(Partial derivative)한 값을 원소로 하는 벡터라고 정의할 수 있습니다.

$$ \nabla f=(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}) $$

이 벡터는 기울기가 가장 가파른 곳으로의 방향, $f$값이 가장 가파르게 증가하는 방향을 의미합니다.

예를 들어, $f(x, y) = x^2 + y^2 + xy$라고 하면 그래디언트는 $\nabla f=(2x+y, 2y+x)$입니다.
임의의 점 $(1, 3)$에서 함수 $f$ 값이 최대로 증가하는 방향은 $(5, 7)$이고 그 기울기(벡터의 크기, 증가의 가파른 정도)는 $\lVert (5, 7) \rVert = \sqrt{5^2 + 7^2} = \sqrt{74}$입니다.

## gradient method

그래디언트는 함수값을 최대화시키는 방향을 의미한다는 점을 위에서 이야기했습니다.
그래디언트 방향을 반대로 하면 어떻게 될까요?
함수값을 최소화시키는 방향을 나타내게 될 것입니다.
이 방향으로 나아가 최적값을 찾는 방법론을 우리는 gradient method라고 부릅니다.
많은 문제의 해답을 이 gradient method로 찾는 이유는 세상에 존재하는 많은 문제들이 analytic하게 답을 구하기 쉽지 않아 점진적으로 문제를 풀어나가야 하며 그 방식에 알맞는 알고리즘이기 때문입니다.
수식으로 알고리즘을 좀 더 살펴보겠습니다.

$$ x_{n+1} = x_n - \alpha \nabla_x f $$

$x_{n+1}$은 $x_n$에서 일정 값을 뺀 결과임을 수식에서 확인할 수 있습니다.
우선 각 요소를 살펴보면 $\nabla_x f$는 함수 $f$를 변수 $x$로 편미분한 값이고 $\alpha$는 learning rate입니다.
전체 함수값을 최소화시키기 위해 $x$의 변화량을 나타내는 것이 $\nabla_x f$이고 그 비율을 조절하는 역할이 learning rate입니다.
Learning rate를 조절해서 변수 $x$가 급격하게 변해 optimal point를 지나치는 것을 방지할 수 있습니다.
현재 위치에서 그래디언트 반대 방향(음수 방향)으로 함수를 업데이트 했기 때문에 전체 함수는 최소값을 찾아가게 됩니다.
만약 함수 최대값을 찾아가기 위해 gradient ascent방법론을 수행하고자 한다면 다음과 같은 수식을 활용하면 됩니다.

$$ x_{n+1} = x_n + \alpha \nabla_x f $$

다만 주어진 그래디언트가 양수, 음수인지에 따라 값은 얼마든지 반전될 수 있기 때문에 통칭해서 gradient method라고 불러도 무방합니다.
여기까지 어떻게 그래디언트 값을 활용해서 변수값을 업데이트 하는지 확인했습니다.
그러나 무한정 학습하는 것은 아닙니다.
Gradient method 방법은 점진적으로 해를 찾아가는 numerical, iterative 방식이기 때문에 어느 시점에서 학습을 멈춰야합니다.
또 때로는 예기치 못하게 발산할 가능성도 있습니다.
따라서 다음과 같은 조건을 만족할 경우에만 변수를 업데이트하고 학습을 진행합니다.

$$ L(\theta + \nabla \theta) < L(\theta) $$

업데이트 된 변수를 기반으로 구한 함수 $L$이 업데이트 되기 이전의 변수로 구한 함수값보다 작은 경우에만 변수를 업데이트 합니다(gradient descent의 경우).
아래 코드는 gradient descent방법론이 최적해를 찾아가는 과정입니다.

<script src="https://gist.github.com/kh-mo/fbecdd96c163b895b5123571fe63d8c1.js"></script>

## backpropagation

Backpropagation 알고리즘은 역전파라고 불리는 알고리즘입니다.
특정 작업을 수행하는 함수가 존재할 때, 해당 함수에 있는 파라미터값을 효과적으로 학습하는 반복적이고 재귀적인 알고리즘이라고 할 수 있습니다.
저는 평소에 딥러닝에 관심을 가지고 있기에 간단한 모델 예시를 통해 backpropagation 알고리즘을 알아보도록 하겠습니다.

![](/public/img/gradient_descent_figure1.JPG "Figure1 of gradient descent, 출처:https://en.wikipedia.org/wiki/Backpropagation")

해당 모델은 $x_1$부터 $x_n$까지 n개 입력값을 받아 $o_j$라는 결과값을 반환하는 함수입니다.
그리고 이 결과값을 내는데 사용되는 파라미터는 $w_{1j}$부터 $w_{nj}$까지 n개입니다.
파라미터와 입력값을 서로 곱해서 transfer function을 통해 그 합을 구하고 결과값 ${net}_j$를 얻습니다.
이 과정을 element-wise sum이라고 부르기도 합니다.
이 ${net}_j$를 특정 activation function에 입력값으로 주어 최종 결과물인 $o_j$를 반환하게 됩니다.

자 그러면 이제부터 backpropagation을 살펴보겠습니다.
해당 함수는 입력 $x$를 받아 결과 $o$를 반환하는데 이 결과가 특정 값에 가까워지기를 원한다고 하겠습니다.
그러나 랜덤한 값으로 초기화 된 $w$ 값으로부터는 원하는 결과를 얻지 못할 확률이 큽니다.
때문에 해당 값에 가까워 질 수 있도록 $w$값을 조절하는 알고리즘에 gradient method, backpropagation이 사용됩니다.
현재의 $w_{1j}$가 $o_j$에 미치는 영향도, 순간 기울기, 편미분 값은 $\frac{\partial {o_j}}{\partial w_{1j}}$입니다.
Gradient method 수식에 따르면 원하는 결과값 $o_j$를 얻기 위해 $w_{1j}$를 업데이트 하는 수식은 $w_{1j_{n+1}} = w_{1j_{n}} - \alpha \frac{\partial {o_j}}{\partial w_{1j}}$입니다.
해당 수식서 $\frac{\partial {o_j}}{\partial w_{1j}}$ 부분은 chain rule을 통해 여러 단계로 분할하는 것이 가능합니다.
위 모델 그래프 구조에 따라 $\frac{\partial {o_j}}{\partial w_{1j}} = \frac{\partial {o_j}}{\partial net_j} $\frac{\partial net_j}{\partial w_{1j}}$로 분할이 가능합니다.
$\frac{\partial {o_j}}{\partial net_j}$와 $\frac{\partial net_j}{\partial w_{1j}}$은 local gradient라고 불립니다.
각 단계의 local gradient값을 구해서 그 값을 모델의 결과로부터 입력값까지 전파시켜가는 이 과정을 backpropagation 알고리즘이라고 부릅니다.
아래 그림처럼 진행됩니다.

![](/public/img/gradient_descent_figure2.JPG "Figure2 of gradient descent")


## pytorch backpropagation example
