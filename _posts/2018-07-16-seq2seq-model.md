---
layout: post
title: sequence to sequence 모델
category: Translation
---

## Meta Information
제목 : Sequence to Sequence Learning with Neural Networks
2014.09 : arXiv submit
2014.12  : NIPS

## 연구 동기
이번에 소개할 논문은 시퀀스(sequence)가 있는 데이터를 처리하는 대표적인 딥러닝 모델인 sequence to sequence 모델(이하 seq2seq)입니다.
Deep Neural Network(DNN)가 레이블 된 데이터를 충분히 가지고 있을 경우 여러 AI 어플리케이션에서 높은 성능을 보이지만, 가장 큰 한계점으로 여겨진 부분은 입력과 출력이 고정된 크기의 벡터로 표현되어야 한다는 것입니다.
그러나 언어처럼 매 순간 입력 길이가 달라져 변동성이 큰 데이터는 DNN이 처리하기 매우 까다롭습니다.
즉, DNN을 이용해서 machine translation, speech recognition, question answering과 같은 문제를 푸는 것은 한계가 있습니다.
그렇기 때문에 이 논문에서는 시퀀스가 데이터를 가진 어떤 문제가 주어졌을 때, 이를 학습할 수 있는 방법론을 제안하는 것을 목적으로 했고 그것이 seq2seq 모델입니다.
