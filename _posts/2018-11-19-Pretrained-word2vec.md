---
layout: post
title: Pretrained Word2vec
category: Non-Category
---

## Pretrained Word2vec이란?

Pretrained는 미리 학습이 되었다는 의미입니다.
딥러닝 모델은 가지고 있는 파라미터 수가 굉장히 많기 때문에 이를 처음부터 학습하는 것은 상당한 시간과 컴퓨팅 파워를 요구합니다.
그래서 현실적으로 이미 학습되어 있는 모델을 가져다가 fine-tuning을 하게 됩니다.

NLP분야에서 사용하는 대표적인 pretrained 모델은 단어 벡터입니다.
가장 잘 알려진 단어 벡터 생성 모델인 word2vec 알고리즘으로부터 학습된 단어 벡터를 이용해 많은 NLP문제를 풀 수 있습니다.
본 포스트에서는 pretrained word2vec을 사용하는 방법에 대해 살펴보겠습니다.  

## Google Code Archive, binary file

구글 코드 아카이브(https://code.google.com/archive/p/word2vec/)에서 pretrained word2vec 모델을 다운받을 수 있습니다.
구글 뉴스 데이터셋에서 얻은 300차원의 3백만개 단어 벡터는 바이너리 형식으로 구성된 1.5G 크기의 파일입니다.

## gensim package

해당 바이너리 파일을 읽어 단어벡터를 확인할 때 유용한 패키지로 gensim이 있습니다.
이 패키지를 이용해서 파일을 읽고, 단어 갯수를 확인하고, 단어 벡터를 보는 코드는 하단에 나타나 있습니다. 

<script src="https://gist.github.com/kh-mo/64499b18463a7ac2358254f8eb8ac5d0.js"></script>
