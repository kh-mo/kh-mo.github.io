---
title: Isolation Forest
category: Incomplete writing
---

본 포스트는 2008년 IEEE에 억셉된 [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest) 논문을 정리한 것입니다.
공부한 내용을 정리하는 것이기에 잘못된 점이 있을 수 있습니다.
해당 내용에 대한 교정, 첨언, 조언은 언제나 환영합니다.

## 기존의 방법론 세계

이상치 탐지(anomaly detection)을 수행하는 알고리즘은 크게 model-based, distance-based, density-based 방법론으로 나눌 수 있습니다.
그리고 model-based는 다시 statistical methods, classification-based, clustering-based로 나눌 수 있습니다.
Isolation forest(이하 iForest)는 이 중 model-based의 방법론 영역에 속합니다.
각 범주에 속하는 알고리즘들은 나름의 이론과 우수성이 있지만 저자들은 iForest가 다음과 같은 점에서 우수성이 있음을 주장합니다.

먼저 같은 model-based 방법론들이 정상 데이터의 범주(profile of normal instances)를 명확히하는 것에 초점을 맞추고 있는 것과 달리 iForest는 명시적으로 비정상 데이터를 고립(isolation)시키는 방법을 채택합니다.
이는 정상 데이터 범주를 최적화시키는 것이 비정상 데이터 탐지를 최적화시키는 것과 동일한 의미가 아니라는 것입니다.
정상 데이터 영역이 너무 specialization되어 정상 데이터를 비정상으로 판단하는 false alarm 발생가능성에 대한 이야기입니다.
또한 계산 복잡도가 높아 고차원 데이터(large & high dimenstional data)를 다루기 어렵다는 점도 있습니다.

다음으로 distance-based, density-based 방법론들과 비교해 다음과 같은 우위가 있음을 주장합니다.
iForest는 부분 모델(partial model)을 구축하는 것이 가능하며 굉장히 작은 sub-sampling을 쓸 수 있습니다.
트리의 큰 축을 담당하는 정상 데이터를 분리하는 부분이 필요없기 때문에 부분 모델을 구축하는 것이 가능하며 작은 sample 사이즈는 swamping, masking 문제를 줄이기 때문입니다.
거리나 밀도 측정 계산이 필요 없기 때문에 계산복잡도에 이점이 있고(iForest는 선형계산량) 따라서 메모리 요구량이 적습니다.
이런 특징은 고차원 데이터를 다룰 수 있음을 뜻하기도 합니다.

## Isolation Forest는 어떤 알고리즘인가?

iForest는 비정상 개체를 고립시키는 방법을 사용하는 Tree 기반 알고리즘으로 model-based 방법론 범주에 속합니다.
비정상 개체가 고립된다는 것은 트리에서 특정 노드에 비정상 개체들이 담긴다는 것을 의미하는 데 이 노드가 루트노트와 굉장히 가깝다는 것이 핵심입니다.
비정상 데이터들은 정상 데이터 대비 그 숫자가 적은 소수이면서 상당히 다른 속성값을 가지기 때문에 tree에서 빠르게 분기가 됩니다.
논문에서도 비정상 데이터들의 'few & different' 속성에 주목하여 비정상 데이터가 담긴 노드들의 평균 경로 길이가 정상 데이터들보다 짧음을 이용해 이상치 탐지 알고리즘을 개발했습니다.
iForest는 언급한 속성을 가진 isolation Tree(이하 iTree)의 앙상블이기 때문에 주요 하이퍼 파라미터로 트리 숫자와 sub-sampling 사이즈를 가집니다.


Sub-sampling을 사용할 수 있다는 것은 알고리즘의 동작 속도를 올리는 효과가 있음.
swamping, masking을 줄일 수 있음.



본 논문이 고려하는 기존의 이상치 탐지(anomaly detection) 모델들은 정상 개체로 판단되는 범주를 만드는 model-based 방법입니다.
새로운 테스트 데이터가 만약 모델이 설정한 정상 범주에 들어오지 않는다면 이 개체는 비정상으로 판단됩니다.
선행 연구들보다 sub-sampling을 통해 계산 복잡도, 메모리 사용량에서 얻는 이득이 큰 장점이 있습니다.




## 수식적으로 살펴본 Isolation Forest

## 코드 구현 내용

## 의견

