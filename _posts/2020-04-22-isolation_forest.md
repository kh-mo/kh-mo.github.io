---
title: Isolation Forest
category: Incomplete writing
---

본 포스트는 2008년 IEEE에 억셉된 [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest) 논문을 정리한 것입니다.
공부한 내용을 정리하는 것이기에 잘못된 점이 있을 수 있습니다.
해당 내용에 대한 교정, 첨언, 조언은 언제나 환영합니다.

## 기존의 방법론 세계

본 논문이 고려하는 기존의 이상치 탐지(anomaly detection) 모델들은 정상 개체로 판단되는 범주를 만드는 model-based 방법입니다.
새로운 테스트 데이터가 만약 모델이 설정한 정상 범주에 들어오지 않는다면, 이 개체는 비정상으로 판단됩니다.


## Isolation Forest는 어떤 알고리즘인가?

Isolation Forest는 비정상 개체를 고립시키는 방법을 사용하는 알고리즘으로 model-based 방법론에 속합니다.
선행 연구들보다 sub-sampling을 통해 계산 복잡도, 메모리 사용량에서 얻는 이득이 큰 장점이 있습니다.


## 수식적으로 살펴본 Isolation Forest

## 코드 구현 내용

## 의견

