---
layout: post
title: 젠센 부등식, Jensen's Inequality
category: Non-Category
---




젠센 부등식(Jensen's Inequality)은 특정 조건에서 특정 성질을 만족하는 부등식입니다.
함수 $f$가 어떤 형태인에 따라  그리고 함수 앞 숫자의 조건, $x$값에 따라 하한값을 결정할 수 있는 방정식입니다.


## convex와 concave

convex 함수는 볼록함수라고도 알려져 있습니다.
위키피디아에서 검색할 경우 convex 함수는 다음과 같은 성질을 만족하는 함수로 정의되어 있습니다.

$$ \forall x_1 \neq x_2 \in X, \forall t \in (0, 1): f(tx_1 + (1-t)x_2) < tf(x_1) + (1-t)f(x_2)$$



## 내분점에서 관계


결론
f(x)가 convex일 때, f(E[x]) <= E[f(x)]
f(x)가 concave일 때, f(E[x]) >= E[f(x)]
