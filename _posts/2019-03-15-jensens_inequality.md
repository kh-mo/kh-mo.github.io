---
layout: post
title: 젠센 부등식, Jensen's Inequality
category: Non-Category
---




젠센 부등식(Jensen's Inequality)은 특정 조건에서 특정 성질을 만족하는 부등식입니다.
함수 $f$가 어떤 형태인에 따라  그리고 함수 앞 숫자의 조건, $x$값에 따라 하한값을 결정할 수 있는 방정식입니다.


## convex와 concave

흔히 볼록함수라고 알려진 convex 함수는 위키피디아에서 검색할 경우 다음과 같은 성질을 만족하는 함수로 정의되어 있습니다.

$$ \forall x_1 \neq x_2 \in X, \forall t \in (0, 1): f(tx_1 + (1-t)x_2) < tf(x_1) + (1-t)f(x_2)$$

풀이하자면 모든 $X$집합의 원소 $x$ 중 $x_1$과 $x_2$가 같지않고 $t$가 0~1 사이의 값을 가질 때, 해당 수식 $f(tx_1 + (1-t)x_2) < tf(x_1) + (1-t)f(x_2)$ 을 만족하는 함수를 convex 함수라고 합니다.
이 형태를 그림으로 표현하면 다음과 같이 볼 수 있습니다.

![](/public/img/jensens_inequality_figure.JPG "Figure of jensens_inequality_figure")

$x_1$과 $x_2$ 사이에 있는 $tf(x_1) + (1-t)f(x_2)$의 함수값은 늘 직선 아래에 위치하게 됩니다.
만약 함수 $f$가 직선인 경우에만 $f(tx_1 + (1-t)x_2)$과 $tf(x_1) + (1-t)f(x_2)$이 일치하게 됩니다.

## 내분점에서 관계


결론
f(x)가 convex일 때, f(E[x]) <= E[f(x)]
f(x)가 concave일 때, f(E[x]) >= E[f(x)]
