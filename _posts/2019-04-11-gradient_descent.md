---
layout: post
title: 오차의 최소화, Gradient Descent
category: Definition
---

## Example

실제 가지고 있는 데이터 x, y가 다음과 같은 튜플 데이터라고 가정하겠습니다.

(x, y) = (10, 21)

그리고 기울기가 3, 절편이 2인 임의의 함수 y = 3x + 2 로부터 데이터에 피팅되는 gradient descent 과정을 살펴보겠습니다.
x 데이터 값을 넣으면 함수는 32를 예측값으로 줄 것입니다.
그러나 실제 우리가 기대하는 결과는 21입니다.
따라서 11의 오차가 발생하는데 이 값을 이용해서 우리는 기울기와 절편을 조절할 것입니다.

m(n+1) = m(n) + gradient(m)

b(n+1) = b(n) + gradient(b)

기울기와 절편을 조절해 오차를 줄여가는 방법은 사실 많이 있을 수 있습니다.
그 중 1차 미분값을 이용하여 m, b의 변화가 실제 y를 예측하는데 미치는 정도를 사용하는 방식이 그래디언트 디센트 방식입니다.

우리는 1차 미분값을 사용하기 때문에 m과 b는 결국 y에 대해서 미분 가능해야 합니다.

error = y - hat(y) = y - mx - b

미분(error/m) = -x



## what is the gradient?

다변수 함수 $f$가 $n$개 변수로 이루어져 있다면 다음과 같이 표현할 수 있습니다.

$$ f(x_1, x_2, ..., x_n) $$

이 $f$ 함수의 그래디언트는 함수 $f$를 각 변수로 편미분한 값을 원소로 하는 벡터입니다.

$$ f(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}) $$


## gradient descent

## backpropagation



