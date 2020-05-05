---
title: GAN, Generative Adversarial Nets
category: Incomplete writing
---

Generative Model은 원하는 특정 결과를 생성해내는 모델군이라고 할 수 있습니다.
생성할 수 있는 결과물은 다양할 수 있겠지만 지금 이 포스트에서는 이미지를 생성해내는 모델, 그 중에서도 GAN을 다루고자 합니다.
저 자신이 공부하고 있는 내용을 바탕으로 작성되었기에 잘못된 해석이나 이해가 포함될 수 있으니 첨언과 조언은 항상 부탁드립니다.

## 무엇을 배울 것인가?

Generative Model은 P(x), P(x|z), P(x|y)와 같이 확률분포를 학습하는 모델로 정의가 됩니다.
우리가 만들고 싶은 데이터들이 생성되는 메커니즘이 존재하고 그것을 확률분포라고 보는 것입니다.
이 분포를 알 수 있다면 학습에 사용된 데이터뿐만 아니라 그것과 유사한, 그렇지만 완전히 똑같지는 않은 여러 데이터를 생성해 낼 수 있을 것입니다.

## GAN의 접근 방식

GAN은 두 딥러닝 모델이 서로 경쟁하여 성능을 향상시키는 방식으로 알려져 있습니다.
초기에는 미흡한 성능을 보이더라도 점차 경쟁을 통해 나아져가는 것이죠.
Adversarial 이라는 단어가 GAN에 포함된 것도 이러한 프레임워크를 사용하기 때문입니다.

좀 더 자세하게 살펴보면 우선 GAN을 이루는 두 모델은 *generator*와 *discriminator*라고 부릅니다.
Generator가 이미지를 생성하고 discriminator는 그 이미지를 판단합니다.
특정한 확률 분포 P(z)로부터 샘플링 된 랜덤한 노이즈 벡터 z를 입력으로 받아 generator는 이미지를 생성합니다.
이 때 만들어지는 딥러닝 모델의 아키텍쳐는 어떤 형태든 크게 상관이 없습니다(물론 좋은 이미지를 만들어내는 아키텍쳐가 존재할테지만 임의의 함수라고 놓고 가겠습니다).
Discriminator는 이미지를 입력으로 받아 이것이 실제 이미지인지 아닌지를 판단합니다.
학습 데이터셋에 있는 이미지인 경우 확률값 1을 반환하고 generator가 생성한 이미지인 경우 확률값 0을 반환하도록 합니다.

Generator와 discriminator가 모두 딥러닝 아키텍쳐를 가지는 모델들이기 때문에 SGD 방식을 이용해서 학습할 수 있습니다.
그러려면 loss function이 잘 정의되어야겠죠.
GAN의 loss function은 아래와 같습니다.

$$ \min_{G}\max_{D}{V(D,G)} = E_{x \sim p_{data}(x)}[\log D(x)] + E_{z \sim p_{z}(x)}[\log (1-D(G(z)))]$$

이 수식은 generator의 입장과 discriminator 입장에서 해석해야 합니다.
먼저 discriminator 입장에서 보면 크게 두가지 입력 이미지를 받아 loss를 계산합니다.
위 수식의 첫번째 term은 $P_{data}$에서 얻어진 샘플 x를 입력으로 받아 이미지를 판단합니다.
이 때 $P_{data}$는 train DB를 뜻하고 얻어진 샘플 x는 실제 이미지를 뜻합니다.
Discriminator는 이 데이터의 확률값이 1에 가깝도록 학습합니다.
두번째 term은 $P_{z}$에서 얻어진 샘플 z를 이용해 generator가 생성한 이미지 $G(z)$를 판단하는 부분입니다.
이 때 discriminator는 $G(z)$에서 얻은 이미지의 확률값이 0에 가깝도록 학습합니다.

Generator는 두번째 term loss를 통해 학습하는데 이는 discriminator가 학습하는 방향과 adversarial합니다.
Discriminator는 G(z)에 대한 확률값을 0으로 판단하려고 학습하는데 반해 Generator는 이 확률값을 1이 되도록 학습하는 것입니다.
이 과정을 반복하면서 generator는 점차 trainDB에 있는 이미지 데이터들과 유사한 이미지를 생성해 낼 것입니다.

Ian Goodfellow는 optimal discriminator가 주어졌을 때 어떠한 generator에 대한 확률값을 다음과 같이 표현했습니다.

$$ D_G^* (x) = \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}$$

이 수식은 discriminator 입장에서 해당 수식을 최대화하도록 학습할 때 유도됩니다.

$$
\begin{align}
V(G,D) &= \int_{x} p_{data}(x)\log(D(x))\, dx + \int_{z} p_{z}(z)\log(1-D(g(z)))\, dz \\
&= \int_{x} p_{data}(x)\log(D(x)) + \int_{z} p_{g}(x)\log(1-D(x))\, dx \\
\end{align}
$$

학습셋에 있는 확률값은 최대화, generator에서 나온 이미지데이터에 대한 확률값을 최소화하면 위의 수식은 최대값을 가지게 됩니다.
즉, 해당 수식을 $D(x)$가 [0, 1]인 범위에서 미분하면 최대값을 가지는 optimal discriminator의 수식이 유도됩니다.

이 optimal discriminator가 존재한다고 가정하고, minimax game을 진행하고 있는 GAN의 목적함수는 다음과 같이 재정의 될 수 있습니다.

$$
\begin{align}
C(G) &= \max_{D}{V(G,D)} \\
&= E_{x \sim p_{data}}[\log D_G^* (x)] + E_{z \sim p_{z}}[\log (1-D_G^* (G(z)))] \\
&= E_{x \sim p_{data}}[\log D_G^* (x)] + E_{x \sim p_{g}}[\log (1-D_G^* (x))] \\
&= E_{x \sim p_{data}}[\log \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}] + E_{x \sim p_{g}}[\log \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}] \\
\end{align}
$$

이 목적함수를 이용해 global minimum을 찾으려면 $ p_g = p_{data} $가 되어야합니다.
그 말인즉슨 generator가 trainDB의 분포를 잘 학습했음을 의미합니다.
논문에서 Ian Goodfellow는 Theorem 1을 통해 global minimum에 도달했을 때 $C(G)$가 $-\log(4)$임을 증명했습니다.
이것을 결론부터 거슬러 올라가면 다음과 같이 유도할 수 있습니다.

$$
\begin{align}
C(G) &= -\log(4) + 2 * JSD(P_{data} || P_g) \\
&= -\log(4) + KL(p_{data} || \frac{p_{data}+P_g}{2}) + KL(p_g || \frac{p_{data}+P_g}{2}) \\
&= -\log(4) + \sum_{i} p_{data}(i)*\log(\frac{p_{data}(i)}{\frac{p_{data}+P_g}{2}}) + \sum_{i} p_{g}(i)*\log(\frac{p_{g}(i)}{\frac{p_{data}+P_g}{2}}) \\
&= -\log(4) + \log(2) + \sum_{i} p_{data}(i)*\log(\frac{p_{data}(i)}{p_{data}+P_g}) + \log(2) + \sum_{i} p_{g}(i)*\log(\frac{p_{g}(i)}{p_{data}+P_g}) \\
&= \sum_{i} p_{data}(i)*\log(\frac{p_{data}(i)}{p_{data}+P_g}) + \sum_{i} p_{g}(i)*\log(\frac{p_{g}(i)}{p_{data}+P_g}) \\
&= E_{x \sim p_{data}}[\log \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}] + E_{x \sim p_{g}}[\log \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}] \\
\end{align}
$$

수식을 따라가니 결국 위에서 재정의한 $C(G)$와 같은 결론을 얻었습니다.
이 수식을 따라갈 때 필요한 Kullback-Leibler divergence(KL-divergence)와 Jensen-Shannon divergence 수식은 아래와 같습니다.

- KL-divergence

$$ KL(P||Q) = \sum_{i} P(i)\log(\frac{P(i)}{Q(i)}) $$

- Jensen-Shannon divergence

$$ JSD(P||Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M) $$

