---
layout: post
title: Pruning Research
category: Non-Category
---

## Approach

[Learning both weights and connections for efficient neural network, NIPS 2015](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf)

![](/public/img/pruning_figure1.JPG "Figure1 of Learning both weights and connections for efficient neural network")

Weight pruning 계열의 논문입니다.
학습된 네트워크를 가져와서 일정한 기준(threshold)에 미치지 않는 가중치 연결을 끊고 그 상태의 네트워크를 재학습시키는 방법론입니다.
이 과정을 반복하면 sparse한 모델을 얻는 것이 가능해집니다.
다만 재학습 비용이 크다는 단점이 있습니다.

[Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration, CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)

![](/public/img/pruning_figure2.JPG "Figure1 of Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration")

Filter pruning 계열의 논문입니다.
Redundancy가 있는 filter를 제거하는 방법론을 취하며 이를 위해 geometric median이라는 기준을 사용합니다.
특정 filter와 다른 모든 필터간의 유클리디안 거리합을 구하여 그 값이 최소가 되는 필터들을 제거합니다.
수식으로는 다음과 같이 표현됩니다.

$$
\begin{align}
x^* \in argmin_{x \in R^d} f(x) \\
where f(x) \overset{\underset{\mathrm{def}}{}}{=} \sum_{i \in [1,n]} {\parallel x-a^{i} \parallel}_2 \\
\end{align}
$$

논문의 설명에 따르면 거리합이 최소가 되는 GM filter는 다른 필터들로 충분히 설명될 수 있는 필터이기 때문에 제거가 가능하다고 합니다.
이는 선형대수 관점에서 필터들이 생성하는 공간이 존재하고 선형결합으로 만들어질 수 있는 필터를 제거하여 공간을 생성(span)하는 필터들만 남기는 접근 방법론으로 해석을 할 수도 있습니다.
정확도의 손실이 많지 않고 FLOPs를 상당히 줄일 수 있다는 pruning 계열의 장점을 보입니다.
다만 filter를 zerorize하고 제거하지 않는다는 점, 또한 geometric median을 정확히 구할 수 없다는 점이 단점으로 여겨집니다.
