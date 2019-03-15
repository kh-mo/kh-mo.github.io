---
layout: post
title: 젠센 부등식, Jensen's Inequality
category: Non-Category
---

젠센 부등식(Jensen's Inequality)은 특정 조건에서 특정 성질을 만족하는 부등식입니다.
함수 $f$의 형태와 가중치 조건에 따라 하한값을 결정할 수 있습니다.

## convex와 concave

흔히 볼록함수라고 알려진 convex 함수는 위키피디아에서 검색할 경우 다음과 같은 성질을 만족하는 함수로 정의되어 있습니다.

$$ \forall x_1, x_2 \in X, \forall t \in [0, 1] $$

$$ f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2) $$

풀이하자면 모든 $X$집합의 원소 중 $x_1$과 $x_2$가 주어지고 $t$가 0~1 사이의 값을 가질 때, 해당 수식 $f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)$ 을 만족하는 함수를 convex 함수라고 합니다.
이 형태를 그림으로 표현하면 다음과 같이 볼 수 있습니다.

![](/public/img/jensens_inequality_figure1.JPG "Figure1 of jensens_inequality_figure")

$x_1$과 $x_2$ 사이에 있는 $tf(x_1) + (1-t)f(x_2)$의 함수값은 늘 직선 아래에 위치하게 됩니다.
함수 $f$가 직선인 경우에만 $f(tx_1 + (1-t)x_2)$와 $tf(x_1) + (1-t)f(x_2)$가 일치하게 됩니다.

오목함수라고 불리는 concave 함수는 convex 함수에 음수를 취한 함수로 정의됩니다.
따라서 위 부등식의 등호 방향이 바뀌게 됩니다. 

$$ f(tx_1 + (1-t)x_2) \geq tf(x_1) + (1-t)f(x_2)$$

## 다변수, Jensen's Inequality, 통계적 관점

앞선 수식에서는 변수가 $x_1$, $x_2$로 두 개 였습니다.
만약 변수가 더 추가된다면 어떻게 될까요?
함수가 convex이고 양수 가중치를 가진 다변수 $x$에 대해 Jensen's Inequality는 다음과 같이 정의될 수 있습니다.

$$ \forall x \in X, w_i > 0, \sum_{i=1}^N w_i=1 $$

$$ f(\sum_{i=1}^N {w_i}{x_i}) \leq \sum_{i=1}^N {w_i}f(x_i) $$

변수가 늘어남에 따라 더 고차원의 공간을 표현할 수 있으나 결국 성질은 동일하게 됩니다.
그리고 우리는 여기서 $w_i$에 대해 한 번 더 생각해 볼 수 있습니다.
$w_i$는 0~1 사이값을 가지며 모든 $w_i$ 합은 1이 됩니다.
이는 $x$ 변수에 대한 확률값으로 생각할 수 있습니다.
즉, $\sum_{i=1}^N {w_i}{x_i}$은 $x$에 대한 기대값으로 생각할 수 있습니다.
좌변도 똑같이 $f(x)$에 대한 기대값으로 생각할 수 있습니다.

$$ f(E[x]) \leq E[f(x)] $$ 

따라서 우리는 다음과 같이 정리할 수 있습니다.

>
> f(x)가 convex일 때, $f(E[x]) \leq E[f(x)]$
>
> f(x)가 concave일 때, $f(E[x]) \geq E[f(x)]$
>

## Example

확률분포 $p(x)$를 다음과 같이 정의해봅시다.

$$ p(x) = \int\limits p(x|z)p(z)dz $$

확률분포를 convcave 함수로 만들기 위해 log 변환을 수행하겠습니다.
그러면 아래처럼 변형됩니다.

$$ log(p(x)) = log(\int\limits p(x|z)p(z))dz $$

$p(x)$ 는 0~1 사이값을 가지고 log는 대표적인 concave 함수이기 때문에 Jensen's Inequality가 적용될 수 있습니다.

$$ log(\int\limits p(x|z)p(z)dz) \geq \int\limits log(p(x|z)p(z))dz $$

여기서 $log(p(x|z)p(z))$는 $E[x]$가 되는데 적분식이므로 모든 $p(z)$의 합은 1이 됩니다.
즉, $p(z)$를 가중치로 보는 또하나의 Jensen's Inequality가 적용되어 다음과 같이 정리할 수 있습니다.

$$ log(p(x|z)p(z)) \geq p(z)log(p(x|z)) $$

따라서 이를 다같이 정리하면 아래와 같이 정리할 수 있습니다.

$$ log(p(x)) = log(\int\limits p(x|z)p(z))dz \geq \int\limits log(p(x|z)p(z))dz \geq \int\limits p(z)log(p(x|z))dz $$


