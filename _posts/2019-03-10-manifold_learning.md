---
layout: post
title: 의미를 보존하는 공간, manifold
category: Non-Category
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


매니폴드를 잘 찾으면 유사한 데이터를 획득할 수 있을 것이고 이것은 곧 학습 데이터에 없는 데이터도 획득할 가능성이 열림을 의미합니다.

매니폴드는 다음과 같은 가정을 가지고 있습니다.

>
> natural data in high dimensional spaces concentrates close to lower dimensional manifolds.
> 고차원 데이터의 밀도는 낮지만, 이들의 집합을 포함하는 저차원의 매니폴드가 있다.
>
> probability density decreases very rapidly when moving away from the supporting manifold.
> 이 저차원의 매니폴드를 벗어나는 순간 급격히 밀도는 낮아진다.
>

## 데이터 포인트 간의 거리

고차원 공간상에서 각 점들 사이의 유사도는 거리로 측정합니다.
그러나 과연 이것이 올바른 접근법일까요?
근처에 있는 점이 나랑 유사하다고 생각했지만 실제로는 아닐 수 있습니다.
그 예시가 아래와 같습니다.

![](/public/img/manifold_learning_figure3.JPG "Figure3 of manifold_learning, A Global Geometric Framework for Nonlinear Dimensionality Reduction")
 
위 이미지와 같은 이유로 의미적 관계를 담고있는 subspace, 매니폴드를 찾아야 할 필요가 있습니다.
그리고 각 데이터 포인트를 매니폴드 공간에 projection 시키면 점들 사이의 관계도 재정립될 수 있습니다.
차원이 축소되는 효과도 함께 거둘 수 있습니다.  

## 알고리즘

오토인코더는 unsupervised learning 알고리즘입니다.
인코더가 데이터를 저차원으로 임베딩시키면서 매니폴드 공간을 찾아주게 되고 디코더는 이 저차원 latent vector를 이용해서 디코딩을 수행합니다. 
이 오토인코더의 입력에 noise를 추가하고 noise가 없는 output을 산출하는 알고리즘이 denoising autoencoder입니다.
노이즈를 추가하였지만 어느정도 의미는 보존된 입력값을 가지고 있기 때문에 결과적으로 의미적으로 같지만 입력형태가 다른 인풋을 같은 매니폴드 공간에 매핑함으로써 더 좋은 매니폴드 공간을 찾을 수 있다는 것이 denoising autoencoder의 contribution입니다.





