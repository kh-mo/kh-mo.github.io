---
title: 공분산과 공분산 행렬(covariance)
category: Notation
---

## 공분산이란?

공분산이란 2개 변수가 함께 변하는 정도(joint variability)를 측정하는 척도입니다.
두 변수가 있을 때, 한 변수값이 커지면서 다른 변수도 값이 증가하는 등 두 변수의 변화 경향성이 유사하다면 공분산은 양수(positive)입니다.
반대로 한 변수값이 커질 때 다른 변수값이 작아지는 반대 경향성을 보인다면 공분산은 음수(negative)입니다.
공분산은 이렇게 두 변수의 선형적 관계를 설명하는 지표 중 하나입니다.
수식으로는 다음과 같이 나타날 수 있습니다.

$$
\begin{align}
cov(x,y) &= E[(x-E[x])(y-E[y])] \\
&= E[xy-xE[y]-E[x]x+E[x]E[y]] \\
&= E[xy]-E[x]E[y]-E[x]E[y]+E[x]E[y] \\
&= E[xy]-E[x]E[y] \\
\end{align}
$$

$$cov(x,y)$$는 공분산을 타나내는 기호로 $$\sigma_{xy}$$ 또는 $$\sigma(x,y)$$로 표현하기도 합니다.
$$E[x]$$는 변수 x의 기대값을 의미합니다.

## 공분산 행렬

공분산 행렬(covariance matrix)는 변수들 사이의 공분산을 행렬 형태로 나타낸 것입니다.
공분산 행렬은 정방행렬(square matrix)이자 전치(transpose)를 시켰을 때 동일한 행렬이 나타나는 대칭행렬(symmetric matrix)인 특징이 있습니다.
또, 행렬의 대각항들은 단일 변수의 분산을 의미합니다.
벡터 $$X = \{x_1, x_2, ... , x_n\}$$가 각 변수들의 집합일 때, 공분산 행렬은 $$\Sigma$$ 또는 $$K_{XX}$$ 기호로 표현합니다.

## 계산 예시

위의 개념을 따라 예시를 통해 공분산과 공분산 행렬을 구해보겠습니다.
우선 좌표평면에 임의의 군집 두 개를 찍어보겠습니다.

![](/public/img/covariance_figure1.JPG "Figure1 of covariance")

빨간색 군집의 벡터 X는 $$\{x,y\}$$로 이루어진 x, y 축입니다.
총 4개 점, (1,2), (2,1), (2,3), (3,2)로 구성되어 있고 이 x,y의 공분산은 다음과 같이 계산됩니다.

$$
\begin{align}
cov(x,y) &= E[(x-E[x])(y-E[y])] \\
&= \frac{(1-2)(2-2) + (2-2)(1-2) + (2-2)(3-2) + (3-2)(2-2)}{4} \\
&= \frac{0}{4} = 0 \\
\end{align}
$$

같은 방식으로 x의 공분산과 y의 공분산을 각각 구해보겠습니다.

$$
\begin{align}
cov(x,x) &= E[(x-E[x])(x-E[x])] \\
&= \frac{(1-2)(1-2) + (2-2)(2-2) + (2-2)(2-2) + (3-2)(3-2)}{4} \\
&= \frac{2}{4} = \frac{1}{2} \\
\end{align}
$$

$$
\begin{align}
cov(y,y) &= E[(y-E[y])(y-E[y])] \\
&= \frac{(2-2)(2-2) + (1-2)(1-2) + (3-2)(3-2) + (2-2)(2-2)}{4} \\
&= \frac{2}{4} = \frac{1}{2} \\
\end{align}
$$

공분산 행렬 $$\Sigma$$는 다음과 같이 표현될 수 있습니다.

$$
\begin{align}
\Sigma &= \begin{pmatrix} cov(x,x) & cov(x,y)\\ cov(y,x) & cov(y,y) \end{pmatrix} \\
&= \begin{pmatrix} \frac{1}{2} & 0\\ 0 & \frac{1}{2} \end{pmatrix} \\
\end{align}
$$

코드는 다음과 같이 쓸 수 있습니다.

```
import numpy as np
import matplotlib.pyplot as plt

# data
x = [2,1,2,3,6,7,6,5]
y = [3,2,1,2,5,6,7,6]
color = [0,0,0,0,1,1,1,1]

# covariance matrix
np.cov(x[:4],y[:4], bias=True)

array([[0.5, 0. ],
       [0. , 0.5]])

# image
plt.scatter(x[:4],y[:4],c="red")
plt.scatter(x[4:],y[4:],c="blue")
plt.xlim(0,8)
plt.ylim(0,8)
plt.xlabel("x")
plt.ylabel("y").set_rotation(0)
plt.grid(linestyle='dotted')
```
