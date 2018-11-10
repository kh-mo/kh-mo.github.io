---
layout: post
title: Covariance Matrix 이해하기
category: Non-Category
---

## Covariance Matrix 계산은 어떻게 하는가?

D차원 벡터로 이루어진 단어 n개가 주어진 2차원 행렬을 가정해보자.
다음과 같은 형태로 표현할 수 있을 것이다.

| -    | word1 | word2 | ....  | wordn |
| ---- | ----- | ----- | ----- | ----- |
| d1   | x11   | x12   | ....  | x1n   |
| d2   | x21   | x22   | ....  | x2n   |
| ...  | ...   | ...   | ....  | ...   |
| dD   | xD1   | xD2   | ....  | xDn   |


이 행렬에서 단어벡터는 한 열을 의미하게 된다.
단어 벡터 사이의 상관관계를 표현하는 공분산 행렬은 다음과 같이 계산할 수 있다.

$$cov(wordi,wordj) = \frac{1}{D-1} \sum_{N}^i \sum_{N}^j (wordi-u(wordi))^T(wordj-u(wordj))$$

여기서 u(wordi)는 wordi의 element 평균값으로 채워진 D차원 벡터이다. 









## Covariance Matrix란 무엇인가?

## Covariance Matrix의 성질은 무엇인가?

## Covariance Matrix 예시
