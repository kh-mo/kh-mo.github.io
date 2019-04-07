---
layout: post
title: 의미를 보존하는 공간, manifold
category: Definition
---

랜덤한 난수를 발생시켜 사진을 하나 만든다면 우리는 어떤 사진을 얻을 수 있을까요?
아마 아래와 같은 사진을 얻을 것입니다.

![](/public/img/manifold_learning_figure1.JPG "Figure1 of manifold_learning")

노이즈가 가득한 사진을 얻게 되었습니다.
수천 번을 반복해도 이와 같을 것입니다.
그런데 여기서 이상한 점은 분명 우리가 카메라로 찍는 사진도 어떤 숫자의 조합이기 때문에 난수 발생기를 이용해 사진을 만든다면 평소에 보는 그럴듯한 사진 또는 그것과 유사한 사진이 만들어져야 할 것입니다.
그러나 실험 결과는 그런 이미지가 생성되지 않음을 보여줍니다.
왜 그럴까요?
이 현상을 매니폴드(manifold) 관점에서 설명해 보겠습니다.


## manifold 직관적으로 이해하기

이미지를 구성하는 픽셀, 화소를 하나의 차원으로 간주하여 우리는 고차원 공간에 한 점으로 이미지를 매핑시킬 수 있습니다.
내가 가진 학습 데이터셋에 존재하는 수많은 이미지를 고차원 공간 속에 매핑시키면 유사한 이미지는 특정 공간에 함께 있을 것입니다.
이 내용과 관련한 좋은 시각화 자료는 [여기](http://vision-explorer.reactive.ai/#/galaxy?_k=n2cees)를 참고하시기 바랍니다.
그리고 그 점들의 집합을 잘 아우르는 **subspace**가 존재할 수 있을텐데 그것을 우리는 **매니폴드(manifold)**라고 합니다.

![](/public/img/manifold_learning_figure2.JPG "Figure2 of manifold_learning, Tangent Bundle Manifold Learning via Grassmann&Stiefel Eigenmaps")

고차원 공간에 놓인 점들이 특정한 공간 형태를 따라 분포되어 있음을 직관적으로 볼 수 있습니다.
이렇게 나타난 공간이 매니폴드입니다.
매니폴드는 다음과 같은 가정을 가지고 있습니다.

>
> Natural data in high dimensional spaces concentrates close to lower dimensional manifolds.<br>
> 고차원 데이터의 밀도는 낮지만, 이들의 집합을 포함하는 저차원의 매니폴드가 있다.
>
> Probability density decreases very rapidly when moving away from the supporting manifold.<br>
> 이 저차원의 매니폴드를 벗어나는 순간 급격히 밀도는 낮아진다.
>


## 데이터 포인트 간의 거리

매니폴드 공간을 정의하고 찾았다면 그것의 의미는 무엇일까요?
내가 가진 데이터들을 잘 아우르는 공간이라는 것은 알겠는데 그것으로 무엇을 할 수 있을까요?
지금부터 그것을 알아보겠습니다.

매니폴드 공간은 본래 고차원 공간의 subspace이기 때문에 차원수가 상대적으로 작아집니다.
이는 데이터 차원 축소(dimension reduction)를 가능하게 합니다. 
그리고 차원 축소가 잘 되었다는 것은 매니폴드 공간을 잘 찾았다는 것이기도 합니다.
본래 고차원 공간에서 각 차원들을 잘 설명하는 새로운 특징(feature)을 축으로 하는 공간을 찾았다는 뜻으로 해석할수도 있습니다.
아래 그림을 예시로 살펴보겠습니다.

![](/public/img/manifold_learning_figure3.JPG "Figure3 of manifold_learning, https://dmm613.wordpress.com/tag/machine-learning/")

유명한 MNIST 데이터셋은 784차원 이미지 데이터입니다.
이를 2차원으로 축소하였을 때 한 축은 두께를 조절하고 한 축은 회전을 담당함을 볼 수 있습니다.
매니폴드 공간에서 두 개의 축은 두 개의 특징(feature)를 의미하고 이를 변경하였을 때 변화되는 이미지 형태를 획득할 수 있습니다.
매니폴드 공간은 이렇게 의미론적 유사성을 나타내는 공간으로 해석할 수 있습니다.
이는 또 어떤 이점이 있을까요?

공간속에서 매핑된 데이터들이 얼마나 유사한지 측정하는 방법에는 거리를 재는 방법이 있습니다.
유클리디안 거리를 통해 가장 가까운 점들이 나와 가장 유사하고 생각하는 방법입니다.
그러나 고차원 공간상에서 나와 가까운 점이 실제로 나와 유사하지 않을 수 있다는 관점은 매니폴드로 설명할 수 있습니다.
아래 그림을 살펴보겠습니다.
 
![](/public/img/manifold_learning_figure4.JPG "Figure4 of manifold_learning, A Global Geometric Framework for Nonlinear Dimensionality Reduction")

고차원 공간에서 $B$와 $A1$ 거리가 $A2$ 거리보다 가깝습니다.
그러나 매니폴드 공간에서는 $A2$가 $B$에 더 가깝습니다.
이미지 데이터 픽셀 간 거리는 $\{A1, B\}$가 더 가까울 수 있으나 의미적인 유사성 관점에서는 $\{A2, B\}$가 더 가까울 수 있는 것입니다.
근처에 있는 점이 나랑 유사하다고 생각했지만 실제로는 아닐 수 있는 예시가됩니다.
이것을 실제 이미지로 확인한다면 어떻게 될까요?

![](/public/img/manifold_learning_figure5.JPG "Figure5 of manifold_learning, https://www.cs.cmu.edu/~efros/courses/AP06/presentations/ThompsonDimensionalityReduction.pdf")

자세히보면 고차원 공간에서 이미지는 팔이 2개 골프채가 2개로 좌우 이미지의 픽셀 중간모습을 보여줍니다.
이것은 우리가 원하는 사진이 아닙니다.
반대로 매니폴드 공간에서 중간값은 공을 치는 중간과정 모습, 의미적으로 중간에 있는 모습을 보여줍니다.
우리가 원하는 것도 사실 이것이라고 할 수 있겠지요.
매니폴드를 잘 찾으면 의미적인 유사성을 잘 보존할 수 있습니다.
또한 유사한 데이터를 획득하여 학습 데이터에 없는 데이터를 획득할 가능성도 열리게됩니다.
