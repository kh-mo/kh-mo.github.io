---
title: 강화학습 책 간단 정리
category: tmp
---

본 포스트는 **파이썬과 케라스로 배우는 강화학습**을 간략히 정리한 포스트입니다.
기초적인 내용이지만 주 업무, 학업으로 다루고 있지 않기 때문에 언제든지 다시 확인할 수 있도록 제 언어로 다시 정리한 포스트입니다.

## chapter1

강화학습이란 행동과 결과 사이의 관계를 학습하는 방법입니다.
스스로 어떤 행동을 하고 이 때 환경으로부터 얻는 좋은 보상을 보면서 이 보상을 받는 행동 빈도를 증가시키는 것이 강화학습의 기본 아이디어입니다.

## chapter2

강화학습은 주어진 어떤 문제를 푸는 방법입니다.
이 주어진 문제는 대부분 순차적인 문제이며 보통 MDP(markov decision process)로 정의할 수 있습니다.
MDP의 구성요소는 상태(state), 행동(action), 보상함수(reward), 상태변환확률(state transition probability), 감가율(discount factor), 정책(policy)입니다.
해당 요소들을 통해 에이전트는 환경과 상호작용하여 좋은 정책을 배워갑니다.
이 때 좋은 행동을 했기에 받는 보상을 함수로 표현한 것이 보상함수입니다.
보상함수는 크게 벨만 기대 방정식(bellman expectation equation), 벨만 최적 방정식(bellman optimality equation), 큐함수(q function)으로 이루어져 있습니다.

## chapter3

작은 문제들의 합으로 큰 문제를 푼다는 개념은 다이나믹 프로그래밍(dynamic programming)이라고 합니다.
강화학습으로 풀고자 하는 큰 문제를 step별로 쪼개서 풀 수 있다는 관점에서 강화학습은 일종의 다이나믹 프로그래밍이기도 합니다.
정책을 평가(evaluation)하고 정책을 발전(improvement)시키는 과정을 반복하는 방법론이 정책 이터레이션(policy iteration)이며 이것은 명시적인 정책이 있는 방법론입니다.
혹은 정책이 가치함수에 내제되어 있다고 가정하여 최대 가치함수를 반환하는 방향으로 가치함수를 업데이트 방법론이 가치 이터레이션(value iteration)입니다.
다이나믹 프로그래밍은 모든 상태에 대한 가치함수를 계산하는 방법론이기때문에 차원이 너무 클 경우 적용할 수 없습니다.
계산복잡도가 너무 클 뿐만 아니라 환경에서 얻을 수 있는 모든 정보도 정확해야 하기 때문입니다.
따라서 근사를 수행하는 '학습'의 개념이 들어간 강화학습을 사용하게 됩니다.

### 정책 이터레이션
에이전트 멤버 변수 : 환경정보, 가능한 행동, 가치 테이블, 정책 테이블
에이전트 메소드 : 정책 평가, 정책 발전
환경 멤버 변수 : 상태(state), 보상(reward), 감가율(discount factor)
환경 메소드 : 스탭진행(_step), 초기화(_reset), 화면으로 보여주기(_render)

### 가치 이터레이션
에이전트 멤버 변수 : 환경정보, 가능한 행동, 가치 테이블
에이전트 메소드 : 가치평가
환경 멤버 변수 : 상태(state), 보상(reward), 감가율(discount factor)
환경 메소드 : 스탭진행(_step), 초기화(_reset), 화면으로 보여주기(_render)


## chapter4
전체 환경을 다 계산하는 것이 아니라 경험과 상호작용을 통해 최적 정책을 찾아가는 방법에 대한 이야기를 할 것입니다.
온폴리시 상태에서 에이전트가 환경과의 상호작용을 통해 가치함수를 업데이트 시켜가는 것을 예측(prediction)이라고 합니다.
예측 방법론에는 에피소드(episode)를 끝까지(terminal state)까지 간 후 거쳐간 상태들의 기대값을 구하는 몬테카를로 방식과 매 타임스탭마다 가치함수를 업데이트 하는 시간차(temporal-difference) 방법론이 있습니다.
몬테카를로 방법은 최종상태까지 도착했기때문에 bias는 적으나 variance가 높고 시간차 방법은 실시간으로 적용하기에 좋습니다.
온폴리시 상태에서 정책을 발전시키는 방법을 제어(control)이라고 합니다.
큐함수를 업데이트 시키는 방법론이 살사(sarsa)입니다.
시간차 제어와 엡실론 탐욕 정책을 함께 사용합니다.
엡실론 탐욕 정책은 탐험(exploration)을 수행하기 위해 일정한 엡실론 확률로 최적 정책에 따른 행동이 아닌 다른 행동을 하는 것을 의미합니다.
살사의 경우 다음 큐함수의 행동으로 인해 의도치않은 오류가 발생할 수 있습니다.
그 대안으로써 오프폴리시 방법인 큐러닝(Q-learning)이 있습니다.
큐 러닝은 살사의 다음 큐함수 중 행동을 최적 가치함수 값을 얻는 행동으로 고정하는 방법론을 의미합니다.
이 방법을 쓸 경우 살사의 단점을 극복할 수 있다고 합니다.

## chapter5
환경에 대한 모델 없이 샘플링을 통해 학습하는 것을 모델 프리(model free)라고 합니다.
이제 매개변수로 함수를 근사하는 근사함수(function approximator)를 사용합니다.
근사하려는 함수는 가치함수입니다.
큐함수를 근사하고자 할 때(딥살사), 입력은 n차원의 벡터이고 출력은 가능한 행동에 대한 값들입니다.
또 정책을 근사하는 정책기반 강화학습이 있다.
상태가 주어지면 바로 행동을 선택하는 방법론이다.
입력은 n차원의 벡터이고 출력은 가능한 행동에 대한 확률값들입니다.
