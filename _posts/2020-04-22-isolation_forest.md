---
title: Isolation Forest
category: Incomplete writing
---

본 포스트는 2008년 IEEE에 억셉된 [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest) 논문을 정리한 것입니다.
공부한 내용을 정리하는 것이기에 잘못된 점이 있을 수 있습니다.
해당 내용에 대한 교정, 첨언, 조언은 언제나 환영합니다.

## 기존의 방법론 세계

이상치 탐지(anomaly detection)를 수행하는 알고리즘은 크게 model-based, distance-based, density-based 방법론으로 나눌 수 있습니다.
그리고 model-based는 다시 statistical methods, classification-based, clustering-based로 나눌 수 있습니다.
각 범주에 속하는 알고리즘들은 나름의 이론과 우수성이 있지만 저자들은 기존의 방법론들이 다음과 같은 점에서 아쉬움이 있음을 주장합니다.

먼저 정상 데이터의 범주(profile of normal instances)를 명확히 하는 것에 초점을 맞추고 있는 model-based 방법론의 최적화 방식에 의문을 제기합니다.
왜냐하면 정상 데이터 범주를 최적화시키는 것이 비정상 데이터 탐지를 최적화시키는 것과 동치가 아니기 때문입니다.
이는 다르게 표현하면 정상 데이터 영역이 너무 specialization되어 정상 데이터를 비정상으로 판단하는 false alarm(False Negative, FN)이 발생할 수 있다는 이야기입니다.

![](/public/img/isolation_forest_figure1.JPG "Figure1 of isolation forest")

또한 거리나 밀도 측정 계산이 상당한 계산량을 요구하는 점도 언급합니다.
계산량은 알고리즘의 동작 속도나 메모리 효율과 직결된 요소이기 때문에 매우 중요합니다.
또 고차원 데이터(large & high dimenstional data)를 다룰 수 있는가와도 연관되어 있습니다.
그렇기에 가급적 계산량이 작은 편이 알고리즘 성능 면에서 좋지만 거리나 밀도 기반 방법론들은 이 부분에서 아쉬움이 있습니다.
기존 방법론들의 이러한 아쉬움과 비교하여 iForest는 어떤 장점을 가진 알고리즘이고 어떻게 이상치를 탐지해내는지 알아보겠습니다.

## Isolation Forest는 어떤 알고리즘인가?

iForest는 model-based 방법론 영역에 속한 tree 계열 알고리즘으로 비정상 데이터를 고립(isolation)시키는 방법입니다.
논문에서 고립은 **"한 데이터가 다른 데이터와 분리되는 것"**으로 정의됩니다.
그렇다면 비정상 데이터를 고립시킨다는 것은 어떤 의미인 것인지 살펴보겠습니다.

비정상 데이터는 정상 데이터와 다른 특징을 가지고 있습니다.
논문에서 이 특징을 'few & different'라고 언급하고 있는데 이는 비정상 데이터는 그 숫자가 적은 소수이면서 정상 데이터와는 다른 변수 값을 가지고 있다는 뜻입니다.
Tree 구조에서 데이터의 분기(partitioning)는 랜덤하게 선택된 변수의 최대값과 최소값 사이의 특정 값을 기준으로 하며 모든 데이터가 고립될 때 까지 이를 반복합니다.
비정상 데이터는 숫자가 적기 때문에(few) 분기에 필요한 수가 적고 분기되기 좋은 값으로 인해(different) 고립이 빠르게 될 가능성이 높습니다.
이는 논문의 그림에서도 확인이 가능합니다.

![](/public/img/isolation_forest_figure2.JPG "Figure2 of isolation forest")

결과적으로 비정상 데이터들이 담긴 노드는 트리의 루트 노드와 매우 가깝게 위치할 것입니다.
반대로 정상 데이터들은 트리와 매우 멀리 떨어진 노드에 위치하게 됩니다.
해당 알고리즘은 forest인 만큼, 한 개의 트리에서 나오는 정보만 아니라 여러 tree에서 나온 경로 정보를 사용합니다.
한 데이터가 각 tree에서 나온 경로 길이를 평균 낸 값을 평균 기대경로라고 부르며 이 값이 작을수록 비정상 데이터일 확률이 높다고 여겨 분류를 수행하는 알고리즘이 iForest입니다.

이 알고리즘의 효율성을 높이는 한 축은 full Tree를 구축할 필요가 없다는 점입니다.
비정상 데이터를 탐지하는 것이 목적이기 때문에 전체 full Tree의 평균 경로보다 비정상 데이터 노드들의 평균 경로가 훨씬 짧게 될 것입니다.
즉, 평균 경로 이상으로 트리를 확장시킬 필요가 없고 이는 부분 모델(partial model)을 구축하는 것이 가능함을 시사합니다.
논문에서는 정상 데이터를 분리하는 모든 부분을 구축할 필요가 없다고 언급하는데 같은 이야기입니다.
또 알고리즘의 효율성을 높이는 다른 한축은 작은 sub-sampling 사이즈를 가지는 점입니다.
작은 sampling 사이즈를 선택해도 알고리즘의 구성 방식에는 크게 영향을 주지 않는데 비해, 데이터 셋이 가질 수 있는 masking, swamping 문제를 피할 수 있으면서 동시에 계산량에서도 이득을 얻을 수 있습니다.

## 수식으로 살펴본 Isolation Forest

**(논문에서 n개 instance, n개 externel-node 등 n이 가리키는 것이 무엇인지 혼란스러울 수 있으니 주의!!)**

iForest를 구성하는 기본적인 요소는 iTree 입니다.
iTree는 먼저 [완전이진트리(proper binary tree, full binary tree)](https://www.quora.com/What-is-a-proper-binary-tree)를 기본으로 합니다.
iTree는 주어진 데이터 셋을 특정 기준에 따라 반복적으로 분기하며 다음 3가지 조건 중 하나를 충족시킬 때 멈추게 됩니다.

- 제한 높이(height limit)에 도달한 경우
- 노드의 데이터가 1개인 경우(\|X\|=1)
- 노드에 들어있는 데이터의 값이 모두 동일한 경우

n개 데이터 샘플이 주어진 데이터 셋 $X={x_1, x_2, ..., x_n}$는 iTree에서 분기되어 최종 노드에 담기게 됩니다.
BST(binary search tree)에는 search의 결과물이 내부 노드(internal node)에 있는지 혹은 외부 노드(external)에 있는지에 따라 successful search와 unsuccessful search로 나눠지는 개념이 있습니다.
iTree는 이 중 unsuccessful search의 구조와 유사하며 이 개념에서 두가지 정보를 얻을 수 있습니다.
첫번째는 tree가 얼마나 깊어질 수 있는지 입니다.
논문에서 path length라고 부르는 $h(x)$는 앞으로 비정상 점수(anomaly score)를 계산하는데 매우 중요한 요소인데 unsuccessful search tree에서 $h(x)$가 가질 수 있는 범위는 $0 < h(x) \leq n-1$입니다.
두번째는 tree의 평균깊이(average path length)입니다.
아래와 같은 수식으로 정의되며 유도 과정은 [이곳](https://book.huihoo.com/data-structures-and-algorithms-with-object-oriented-design-patterns-in-c++/html/page309.html)을 참고하시기 바랍니다.

$$ c(n) = 2H(n-1) - (2\frac{n-1}{n}) $$

비정상 점수는 $h(x)$와 $c(n)$을 활용해 다음과 같이 정의됩니다.

$$ s(x,n) = 2^{-\frac{E(h(x))}{c(n)}} $$

$h(x)$의 기대값을 구하는 이유는 알고리즘이 forest이기 때문에 여러 tree에서 나온 결과를 평균내야 하기 때문입니다.
위에서 $h(x)$의 범위를 알았고 이 값이 0에 가까워지면 s는 1에 근접합니다.
반대로 n-1에 가까워지면 s는 0에 근접하게 되지요.

$$ 0 < s \leq 1 $$

다시 처음으로 돌아가서 우리는 비정상 데이터가 'few & different' 성질로 인해 루트 노드에 가깝게 분기될 것이라고 했습니다.
이를 수식에 대입하면 $h(x)$값이 0에 가까워진다는 것을 의미하고 s는 1에 가까워 질 것입니다.
즉, 비정상 점수 s가 커질수록 데이터가 비정상이라고 할 수 있습니다.

## 코드 구현 내용

## 의견

