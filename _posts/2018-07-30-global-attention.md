---
layout: post
title: 입력 시퀀스를 고려한 번역 모델, Global Attention
category: Translation
---

## Meta Information
* 제목 : Neural Machine Translation by Jointly Learning to Align and Translate
* 2014.09 : arXiv submit
* 2015.05 : ICLR

## 연구 동기
인코더 디코더 기반의 신경망 네트워크가 번역 모델에서 높은 성능을 낼 수 있는 가능성을 보이면서 더 나은 성능을 얻기 위한 여러 연구가 진행되었습니다.
오늘 소개할 논문은 디코더가 출력할 단어를 예측하는 과정에서 가장 관련성이 높은 입력 문장의 부분을 찾는 모델을 제안한 연구입니다.
기본적인 인코더 디코더 모델은 인코더가 입력 문장을 순차적으로 인코딩하여 고정된 길이의 벡터(fixed-length vector)를 출력합니다.
그리고 디코더는 이 벡터를 받아 번역 디코딩을 시작합니다.
그러나 고정된 길이의 벡터는 입력 문장의 모든 정보를 벡터 하나로 압축해야하기 때문에 효율적이지 못합니다.
이 문제는 번역해야 하는 문장이 길어질수록 더 심화됩니다.
이를 해결하기 위해 디코더 매 타임 스텝마다 디코더가 번역해야 하는 단어와 가장 관련성이 높은 입력 정보를 선택하는 방법이 제안되었고 이것이 전역 어텐션(global attention) 방법론입니다.
