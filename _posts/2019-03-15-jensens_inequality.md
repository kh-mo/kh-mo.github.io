---
layout: post
title: 젠센 부등식, Jensen's Inequality
category: Definition
---

젠센 부등식(Jensen's Inequality)은 특정 조건에서 특정 성질을 만족하는 부등식입니다.
함수 $f$의 형태와 가중치 조건에 따라 하한값을 결정할 수 있습니다.

## convex와 concave

흔히 볼록함수라고 알려진 convex 함수는 위키피디아에서 검색할 경우 다음과 같은 성질을 만족하는 함수로 정의되어 있습니다.

$$ \forall x_1, x_2 \in X, \forall t \in [0, 1] $$

$$ f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2) $$

풀이하자면 모든 $X$집합의 원소 중 $x_1$과 $x_2$가 주어지고 $t$가 0~1 사이의 값을 가질 때, 수식 $f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)$ 을 만족하는 함수를 convex 함수라고 합니다.
그림을 통해 다시 이해해 보겠습니다.

![](/public/img/jensens_inequality_figure1.JPG "Figure1 of jensens_inequality_figure")

이 그림에서 검은색 선은 대표적인 아래로 볼록한 이차함수이자 convex 함수입니다.
자주색 선은 ${x_1}$, ${x_2}$의 함수값을 직선으로 연결한 선입니다.
$t{x_1} + (1-t){x_2}$은 ${x_1}$과 ${x_2}$를 $(1-t):t$로 내분한 점이고 자주색 선에서 함수값도 $f(x_1)$과 $f(x_2)$를 $(1-t):t$로 내분한 값입니다.
그림에서도 확인할 수 있듯 convex 함수는 두 점 ${x_1}$, ${x_2}$가 주어지고 0~1사이 $t$값을 가질 때, 

$$ f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2) $$

이 수식을 만족합니다.
convex함수의 함수값이 늘 직선 아래에 위치하는 것입니다.
볼록한 방향이 반대인 concave 함수는 convex 함수에 음수를 취한 함수로 정의됩니다.
따라서 위 부등식의 등호 방향이 바뀌게 됩니다.

$$ f(tx_1 + (1-t)x_2) \geq tf(x_1) + (1-t)f(x_2)$$

## 다변수, Jensen's Inequality, 통계적 관점

앞선 수식에서는 변수가 $x_1$, $x_2$로 두 개 였습니다.
만약 변수가 더 추가된다면 어떻게 될까요?
함수가 convex이고 양수 가중치를 가진 다변수 $x$에 대해 Jensen's Inequality는 다음과 같이 정의될 수 있습니다.

$$ \forall x \in X, w_i > 0, \sum_{i=1}^N w_i=1 $$

$$ f(\sum_{i=1}^N {w_i}{x_i}) \leq \sum_{i=1}^N {w_i}f(x_i) $$

변수가 늘어남에 따라 더 고차원의 공간을 표현할 수 있으나 결국 성질은 동일합니다.
여기서 $w_i$에 대해 한 번 더 생각해보면 $w_i$는 0~1 사이값을 가지며 모든 $w_i$ 합은 1이 됩니다.
이는 $w_i$가 변수$x$에 대한 확률이라고 할 수 있고 $\sum_{i=1}^N {w_i}{x_i}$은 $x$에 대한 기대값으로 볼 수 있습니다.
우변의 경우 $f(x)$에 대한 기대값으로 볼 수 있습니다.

$$ f(E[x]) \leq E[f(x)] $$ 

요약하면 결국 이런 것입니다.

>
> f(x)가 convex일 때, $f(E[x]) \leq E[f(x)]$
>
> f(x)가 concave일 때, $f(E[x]) \geq E[f(x)]$
>

## Example

확률분포 $p(x)$를 다음과 같이 정의해봅시다.

$$ p(x) = \int\limits p(x|z)p(z)dz $$

확률분포를 convcave 함수로 만들기 위해 log 변환을 수행하겠습니다.
Log 함수가 앞서 설명했던 함수 $f$ 입니다.
그러면 아래처럼 변형됩니다.

$$ log(p(x)) = log(\int\limits p(x|z)p(z)dz) $$

$p(x)$ 는 0~1 사이값을 가지고 log는 대표적인 concave 함수이기 때문에 Jensen's Inequality가 적용될 수 있습니다.

$$ f(\lim_{n \to \infty}\sum_{i=1}^N {w_i}{x_i}) \geq \lim_{n \to \infty}\sum_{i=1}^N {w_i}f(x_i) $$

$$ log(p(x)) = log(\int\limits p(x|z)p(z))dz \geq \int\limits log(p(x|z))p(z)dz $$
