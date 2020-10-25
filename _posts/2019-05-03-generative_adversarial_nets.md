---
title: GAN, Generative Adversarial Nets
category: Incomplete writing
---

본 포스트는 딥러닝을 활용해 센세이션을 일으킨 Generative Model, [GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) 논문을 정리한 것입니다.
2014년 NIPS에 억셉된 후 어마어마하게 발전된 개념이지요.
공부한 내용을 정리하는 것이기에 잘못된 점이 있을 수 있습니다.
해당 내용에 대한 교정, 첨언, 조언은 언제나 환영합니다.

## GAN의 컨셉

GAN은 Generative Adversarial Nets의 앞글자를 따 만들어진 단어로 이름에서 그 특징이 잘 나타나 있습니다.
생성 모델이며(generative), 적대적인 방법으로 학습하는(adversarial), 네트워크(nets)란 뜻입니다.
우선 GAN은 크게 두 네트워크로 구성되어 있습니다.
입력값을 받아 어떤 결과물을 생성해내는 generative model과 입력값이 generative model로부터 온 것인지 학습 데이터셋에서 온 것인지 판단하는 discriminative model이 그것입니다.
앞으로 generative model은 G로 discriminative model은 D로 표현하겠습니다.
GAN에 이름에 있는 adversarial의 의미는 이 G와 D가 서로 적대하고 있다는 의미입니다.

![](/public/img/generative_adversarial_nets_figure1.JPG "Figure1 of generative adversarial nets")

G의 목적은 D를 실수하게 만드는 것입니다.
D의 목적이 입력으로 들어온 값이 G에서 온 것인지 학습 데이터셋에서 온 것인지를 판별하는 것인데 이를 실수하게 만든다는 것은 G의 결과물이 학습 데이터와 구분이 되지 않는다는 말이 됩니다.
즉, G가 학습 데이터와 유사한 데이터를 생성한다는 것입니다.
GAN은 이렇게 두 네트워크를 학습하여 학습 데이터와 유사한 데이터를 생성하는 생성 모델 G를 만드는 것이 최종 목적입니다.
이제 수식적으로 이를 살펴보겠습니다.

## 핵심 목적함수

좀 더 자세하게 살펴보면 우선 GAN을 이루는 두 모델은 *generator*와 *discriminator*라고 부릅니다.
Generator가 이미지를 생성하고 discriminator는 그 이미지를 판단합니다.
특정한 확률 분포 P(z)로부터 샘플링 된 랜덤한 노이즈 벡터 z를 입력으로 받아 generator는 이미지를 생성합니다.
이 때 만들어지는 딥러닝 모델의 아키텍쳐는 어떤 형태든 크게 상관이 없습니다(물론 좋은 이미지를 만들어내는 아키텍쳐가 존재할테지만 임의의 함수라고 놓고 가겠습니다).
Discriminator는 이미지를 입력으로 받아 이것이 실제 이미지인지 아닌지를 판단합니다.
학습 데이터셋에 있는 이미지인 경우 확률값 1을 반환하고 generator가 생성한 이미지인 경우 확률값 0을 반환하도록 합니다.

GAN의 목적 함수(objective function)는 아래와 같습니다.

$$ \min_{G}\max_{D}{V(D,G)} = E_{x \sim p_{data}(x)}[\log D(x)] + E_{z \sim p_{z}(x)}[\log (1-D(G(z)))]$$

Generator와 discriminator가 모두 딥러닝 아키텍쳐를 가지는 모델들이기 때문에 SGD 방식을 이용해서 학습할 수 있습니다.

#### G의 관점

Generative Model은 P(x), P(x|z), P(x|y)와 같이 확률분포를 학습하는 모델로 정의가 됩니다.
우리가 만들고 싶은 데이터들이 생성되는 메커니즘이 존재하고 그것을 확률분포라고 보는 것입니다.
이 분포를 알 수 있다면 학습에 사용된 데이터뿐만 아니라 그것과 유사한, 그렇지만 완전히 똑같지는 않은 여러 데이터를 생성해 낼 수 있을 것입니다.

Generator는 두번째 term loss를 통해 학습하는데 이는 discriminator가 학습하는 방향과 adversarial합니다.
Discriminator는 G(z)에 대한 확률값을 0으로 판단하려고 학습하는데 반해 Generator는 이 확률값을 1이 되도록 학습하는 것입니다.
이 과정을 반복하면서 generator는 점차 trainDB에 있는 이미지 데이터들과 유사한 이미지를 생성해 낼 것입니다.


#### D의 관점

먼저 discriminator 입장에서 보면 크게 두가지 입력 이미지를 받아 loss를 계산합니다.
위 수식의 첫번째 term은 $P_{data}$에서 얻어진 샘플 x를 입력으로 받아 이미지를 판단합니다.
이 때 $P_{data}$는 train DB를 뜻하고 얻어진 샘플 x는 실제 이미지를 뜻합니다.
Discriminator는 이 데이터의 확률값이 1에 가깝도록 학습합니다.
두번째 term은 $P_{z}$에서 얻어진 샘플 z를 이용해 generator가 생성한 이미지 $G(z)$를 판단하는 부분입니다.
이 때 discriminator는 $G(z)$에서 얻은 이미지의 확률값이 0에 가깝도록 학습합니다.


## 이론적 background

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

## 결과

