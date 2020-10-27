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
각 단어들이 뜻하는 것이 무엇인지 하나씩 살펴보겠습니다.

우선 GAN은 생성 모델을 만드는 것이 최종 목적인 알고리즘입니다.
학습하려는 데이터의 분포를 $P(x)$라고 가정하겠습니다.
이 분포를 잘 모방하는 모델 $P_g(x)$ 네트워크를 만든다면 이 모델을 생성 모델이라고 합니다(이 때 g는 네트워크의 파라미터를 의미합니다).
그러면 이 생성 모델은 어떻게 만들어 질 수 있을까요?

Adversarial이라는 단어가 설명하는 생성 모델을 만들기 위한 새로운 프레임워크가 GAN 논문의 가장 큰 contribution입니다.
GAN은 크게 generator와 discriminator라는 두 모델로 구성되어 있습니다.
Generator가 생성 모델(generative model)이며 discriminator(discriminative model)가 판별 모델입니다.
Generator가 하는 일은 학습 데이터 $P(x)$를 모방하여 특정 데이터를 생성해내는 것이고, discriminator가 하는 일은 generator가 생성한 데이터를 판단하는 것입니다.
앞으로 generator은 G로 discriminator은 D로 표현하겠습니다.

G와 D는 서로 대결하는 구도를 통해 성능을 높여갑니다.
먼저 G는 D의 판단이 잘못되는 것을 목적으로 합니다.
생성된 이미지가 실제 학습 데이터와 유사해질수록 D가 올바른 판단을 내리는 것이 어려워질텐데 그것이 G의 목적입니다.
반대로 D는 G에서 생성된 데이터와 DB에 있는 학습 데이터를 입력으로 받아 이것의 출처가 G인지 DB인지 판단하는 일을 수행합니다.
DB가 곧 $P(x)$이기 때문에 G가 D를 속일 수 있는 DB와 유사한 데이터를 생성한다는 것은 $P_g(x)$가 $P(x)$를 잘 모방했다는 의미가 됩니다.
이렇게 G와 D를 학습하여 $P(x)$를 모당하는 G를 만들어내는 프레임워크를 설명하는 단어가 adversarial 입니다.

![](/public/img/generative_adversarial_nets_figure1.JPG "Figure1 of generative adversarial nets")

마지막으로 nets의 의미는 위에서 설명한 G와 D를 딥러닝 아키텍쳐로 구성되었다는 뜻입니다.
딥러닝은 데이터 수가 많고 고차원일 때 내재된 특징을 잘 찾아내는 능력으로 많은 머신러닝 알고리즘 중 좋은 성능을 보이고 있습니다.
그리고 SGD 방식으로 쉽게 학습시킬 수 있습니다.
GAN 또한 딥러닝 네트워크 구조로 G와 D를 만들었기 때문에 고차원 데이터 분포 $P(x)$를 잘 모방할 수 있을 뿐더러 gradient descent 방식으로 학습시킬 수 있습니다.

## 핵심 목적함수

이제 수식적으로 GAN을 살펴보겠습니다.
먼저 GAN의 목적 함수(objective function)는 아래와 같습니다.

$$ \min_{G}\max_{D}{V(D,G)} = E_{x \sim p_{data}(x)}[\log D(x)] + E_{z \sim p_{z}(x)}[\log (1-D(G(z)))]$$

이 수식에서 $P_{data}$는 학습 데이터 셋이 있는 DB를 의미하고 $P_{z}$는 임의의 확률 분포를 의미합니다.
G는 $P_{z}$에서 샘플링 된 랜덤한 노이즈 벡터 z를 입력으로 받아 DB의 학습데이터와 유사한 데이터를 생성합니다.
이렇게 만들어진 데이터를 $G(z)$라고 합니다.
D는 $G(z)$와 $P_{data}$에서 샘플링 된 학습 데이터 x를 입력으로 받아 0~1 사이의 확률값을 결정합니다.
이런 일을 수행하는 딥러닝 모델 G와 D의 아키텍쳐는 어떤 형태든 크게 상관이 없습니다(물론 좋은 성능을 내는 아키텍쳐가 존재할테지만 임의의 함수라고 놓고 가겠습니다).
위 수식은 G의 관점과 D의 관점에서 다르게 해석할 필요가 있습니다.
따라서 두 관점을 구분해서 자세히 살펴보겠습니다.

#### G의 관점

G는 $P_{z}$에서 샘플링 된 랜덤한 노이즈 벡터 z를 입력으로 받아 $G(z)$를 생성합니다.
그리고 이 값이 $P_{data}$에서 샘플링 된 x와 유사하기를 희망합니다.
D는 입력값이 학습 데이터와 유사할수록 확률 1에 가까운 값을 반환하고 그렇지 않으면 0에 가까운 값을 반환합니다.
즉, $G(z)$는 D를 통해 1에 가까운 확률값을 반환받도록 학습되어야겠지요.
만약 G가 학습이 잘 되어 1에 가까운 확률값을 반환받는다면 GAN의 목적 함수 $V(D,G)$는 어떻게 될까요?

$V(D,G)$의 구성요소 중 G와 관련이 있는 것은 $E_{z \sim p_{z}(x)}[\log (1-D(G(z)))]$ 두번째 term 입니다.
$G(z)$가 D를 통해 1에 가까운 값을 반환받는 것을 목적으로 하기 때문에 괄호안의 log 수식은 $\log (1-1)$이 되어 마이너스 무한대에 가까운 값을 가지게 될 것입니다.
즉, 목적 함수 $V(D,G)$는 굉장히 작은 값을 가지게 되겠지요.
그렇기 때문에 목적 함수의 좌변 $\min_{G}\max_{D}{V(D,G)}$에서 G의 목적이 $V(D,G)$를 최소화한다고 작성되어 있는 것입니다.

#### D의 관점

D의 목적은 입력으로 받은 데이터가 G에서 온 것인지 DB에서 온 것인지 판단하는 것입니다.
G에서 온 데이터라면 0에 가까운 낮은 확률값을 반환해주고 DB에서 온 데이터라면 1에 가까운 높은 확률값을 반환하도록 네트워크를 학습합니다.
즉, $D(G(z))$는 0에 가까운 값, $D(x)$는 1에 가까운 값을 반환한다면 이 경우에 목적 함수 $V(D,G)$는 어떻게 될까요?

D는 $V(D,G)$의 구성요소 두 개와 모두 관련이 있습니다.
먼저 우변의 첫번째 term $E_{x \sim p_{data}(x)}[\log D(x)]$에서 $D(x)$가 1에 가까운 값을 가지기 때문에 수식은 0에 수렴합니다.
또 두번째 term인 $E_{z \sim p_{z}(x)}[\log (1-D(G(z)))]$는 $D(G(z))$가 0에 가깝기 때문에 마찬가지로 수식은 0에 수렴합니다.
결국 D는 목적 함수 $V(D,G)$를 0에 수렴시키는 것이 목적이게 됩니다.
그리고 이 값은 $V(D, G)$가 가질 수 있는 최대값이고요.
그렇기 때문에 목적 함수의 좌변 $\min_{G}\max_{D}{V(D,G)}$에서 D의 목적이 $V(D,G)$를 최대화한다고 작성되어 있는 것입니다.

#### Framework 동작 방식

Optimal D가 먼저 주어지면 학습이 이뤄질 수 없다.
G가 적절한 gradient를 받기 못하기 때문이다.
추가적으로 더 많은 gradient를 받는 trick으로 $1-D(G(z))$를 쓰는 것보다 다른 수식을 쓰는 방법도 있다.

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

