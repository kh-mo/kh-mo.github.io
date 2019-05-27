---
layout: post
title: An Overview on Data Representation Learning From Traditional Feature Learning to Recent Deep Learning 번역
category: Representation-Learning
---

## Abstract

약 100년 전부터, 본질적인 데이터 구조를 학습하기 위해서, 많은 representation learning 방법론이 제안되어왔고, 그 안에는 linear와 nonlinear, supervised와 unsupervised가 포함되어 있습니다.
특히, deep architecture는 최근 representation learning에 광범위하게 적용되었고, image classification, object detection, speech recognition과 같은 많은 task에 최고의 결과를 가져다주었습니다. 
이 논문에서, 우리는 data representation learning 방법론들의 개발과정을 리뷰합니다.
특히, 우리는 전통적인 feature learning 알고리즘과 sota 딥러닝 모델을 조사했습니다.
Data representation의 역사와 유용한 resource와 toolbox가 소개될 겁니다.
마지막으로, 우리는 data representation learning에 관한 몇가지 흥미로운 연구 방향과 주목할만한 점을 언급하며 이 논문을 마무리합니다.

## Introduction

Artificial intelligence, bioinformatics, finance와 같은 많은 도메인에서 data representation learning은 이어 설계되는 classification, retrieval, recommendation task를 가능하게 하는 중요한 단계입니다.
일반적으로, 큰 규모의 application에 대해, 어떻게 본질적인 데이터 구조를 학습하고 데이터로부터 가치있는 정보를 발견할 수 있는지가 점점 더 중요하고 도전적인 과제가 됩니다.

약 100년 전부터, 많은 데이터 representation learning 방법론이 제안되어 왔습니다.
그 중에서도, PCA(principal component analysis)는 1901년에 K.Pearson이 제안했고, LDA(linear discriminant analysis)는 1936년에 R.Fisher가 제안했습니다.
PCA와 LDA는 선형 방법론입니다.
그렇기는 하지만, PCA는 unsupervised 방법론이고 LDA는 supervised 방법론입니다.
PCA와 LDA에 기반하여 kernel PCA, GDA(generalized discriminant analysis)와 같은 다양한 확장된 방법론들이 제안되었습니다.
2000년에는 머신러닝 커뮤니티에서 고차원 데이터의 본질적인 구조를 발견하는 manifold learning에 대한 연구가 시작되었습니다.
PCA나 LDA같은 이전의 global 접근법과는 달리, Isomap(isometric feature mapping)과 LLE(locally linear embedding)같은 manifold learning 방법론은 일반적으로 locality 기반입니다.
2006년에는 G.Hinton과 그의 공저자들이 성공적으로 deep neural network를 차원 축소에 적용했고 Deep Learning의 개념을 제안했습니다.
오늘날 높은 효과를 보이기에 deep learning 알고리즘은 인공지능을 넘어 많은 분야에서 사용되고 있습니다.

반면에 artificial neural network에 대한 연구는 많은 성공과 어려움을 겪어왔습니다.
1943년에 W.McCulloch와 W.Pitts가 신경망을 위한, 후속 연구에서 M-P 모델이라고 불리는 첫 artificial neuron, linear threshold을 만들었습니다.
이후에 D.Hebb은 Hebbian 이론이라고 불리는 신경 가소성(neural plasticity-새로운 환경에 뇌가 적응해가는 능력) 메카니즘에 기반한 hypothesis of learning을 제안했습니다.
근본적으로, M-P 모델과 Hebbian 이론은 neural network 연구와 인공지능 분야의 development of connectionism을 위한 상황을 조성했습니다.
1958년 F.Rosenblatt은 binary classification을 위한 two-layer neural network인 perceptron을 만들었습니다.
그러나 M.Minsky와 S.Papert는 perceptron이 XOR(exclusive-or) 문제를 풀 수 없다는 점을 지적했습니다.
1974년 P.Werbos가 MLP(multi-layer perceptrons)을 학습하기 위한 back propagation 알고리즘을 제안할때까지 neural network 연구는 침체되었습니다.
특히, 1986년 D.Rumelhart와 G.Hinton, R.Williams은 back propagation 알고리즘이 neural network의 hidden layer안에서 유용한 내부적 representation을 생성할 수 있다는 것을 발견했습니다.
Back propagation 알고리즘을 이용하면 이론적으로 neural network의 많은 layer를 학습할 수 있지만 2가지 중요한 문제점이 있습니다.
모델의 과적합(overfitting)과 기울기 발산(gradient diffusion)이 그 문제점입니다.
2006년 G.Hinton은 deep neuural network의 finetuing과 greedy layer별 pre-training을 수행하는 아이디어를 결합해 representation learning 연구에 돌파구를 시도했습니다.
Neural network community를 혼란스럽게한 이슈들은 따라서 해결되었습니다.
이후에 많은 deep learning 알고리즘들이 제안되었고 성공적으로 다양한 도메인에 적용되었습니다.

이 논문에서 우리는 전통적인 feature learning과 최근 deep learning 둘 모두에 관한 data representation learning의 개발과정을 리뷰합니다.
이 논문의 나머지는 다음과 같이 구성되어 있습니다.
2장에서는 선형 알고리즘, 선형 알고리즘의 kernel 확장 알고리즘, manifold learning 방법론과 같은 전통적인 feature learning을 다룹니다.
3장에서는 최근 deep learning의 진보와 중요 모델, tookbox에 대해 다룹니다.
4장에서는 data representation learning에 대한 흥미로운 연구 방향에 대해서 결론을 내립니다.

## Traditional feature learning

이번 장에서 우리는 분류기나 다른 예측기를 만들 때 유용한 정보를 훨씬 쉽게 추출할 수 있는 데이터 변형을 학습하는 것을 목적으로 하는 shallow 모델이 속한 전통적인 feature learning 알고리즘에 집중합니다.
이런 이유로 우리는 SIFT(scale-invariant feature transform), LBP(local binary pattern), HOG(histogram of oriented gradient)와 같은 image descriptor나 TF-IDF(term frequency-inverse document frequency)와 같은 document statistic 계열의 manual feature engineering 방법론을 고려하지 않을 것입니다.

그것들을 공식화하는 관점에서, 알고리즘은 일반적으로 선형-비선형, supervised-unsupervised, generative-discriminative, global-local과 같은 유형으로 구분됩니다.
예를들어 PCA는 선형이자 unsupervised, generative, global feature learning 방법론이고 반면에 LDA는 선형이자 supervised, discriminative, global 방법론입니다.
이번 장에서 우리는 feature learning 알고리즘들을 global인 것과 local한 것으로 범주화한 분류체계를 적용할 것입니다.
일반적으로 global 방법론은 학습된 feature space에서 데이터의 global information을 보존하려하고 local 방법론은 새로운 representation을 학습하는 동안 데이터 사이의 local similarity를 보존하는 것에 집중합니다.
예를들어 PCA와 LDA와 달리, LLE는 locality 기반의 feature learning 알고리즘입니다.
게다가 우리는 manifold learning으로 locality 기반의 feature learning을 부르는데 고차원 데이터에서 manifold structure hidden을 발견하기 때문입니다.

문헌에서 Van der Maaten, Postma, Van den Herik는 34개 feature learning 알고리즘의 코드를 포함한 차원축소를 위한 MATLAB toolbox를 제공합니다.
Yan et al.의 페이퍼에서 많은 차원 축소 알고리즘군을 하나의 공식으로 통합한 graph embedding이라고 알려진 일반적인 프레임워크를 제안합니다.
Zhong, Chherawala, Cheriet의 페이퍼는 handwriting recognition에 대한 3종류 supervised 차원축소 방법론을 비교했다.
그동안에 Zhong, Cheriet의 페이퍼는 tensor로서 입력 데이터를 고려하는 tensor representation learning 관점에서 프레임워크를 제안했는데 이는 많은 선형, kernel, tensor 차원축소 방법론을 한 학습 기준으로 통합했다.

#### Global feature learning

위에서 언급한 것 처럼, PCA는 초기 선형 feature learning 알고리즘 중 하나입니다.
그것의 간결성덕분에, PCA는 차원 축소에 광범위하게 사용되었습니다.
그것은 가능한 correlated 변수값들의 집합을 선형 uncorrelated 변수값들의 집합으로 바꾸기 위해 직교 변환을 사용합니다.
어느정도, 전통적인 MDS(multidimensional scaling)가 PCA는 둘다 선형 방법론이고 고유값 분해를 사용하여 최적화한다는 점에서 유사합니다.
PCA와 MDS 사이의 차이점은 PCA는 입력이 data matrix이고 반면에 MDS는 데이터의 distance matrix입니다.
고유값 분해를 제외하고 SVD(singular value decomposition)은 최ㅏ적화에 잘 사용됩니다.
정보검색에서 LSA(Latent semantic analysis)는 열 사이의 유사한 구조는 보존하면서 행의 숫자를 줄이는 SVD를 사용해서 최적화를 합니다.(이 때 열은 단어 행은 문서를 표현합니다.)
PCA의 variants인 kernel PCA는 kernel trick을 사용해서 비선형 차원 축소를 위해 PCA를 확장한 것이고 probabilistic PCA는 PCA의 확률적 버전입니다.
게다가 PPCA에 기반하여, Lawrence는 GPLVM(Gaussian process latent variable model)을 제안했는데 이것은 완전한 확률적인 것으로 비선형 잠재 변수를 모델링하고 잠재 공간을 관찰 공간으로 비선형 매핑하는 법을 학습할 수 있습니다.
Supervisory information을 GPLVM 프레임워크에 통합시키기 위해서 Urtasun과 Darrell은 discriminative GPLVM을 제안했습니다.
그러나 DGPLVM이 LDA와 GDA의 학습 기준에 기반했기에 DGPLVM에서 학습된 잠재 공간의 차원은 C가 class의 갯수일 때, 최대 C-1개라고 제한되게 됩니다.
이 문제를 해결하기 위해서, Zhong은 잠재변수를 supervisory information에서 graph를 구조화시킨 GMRF(Gaussian Markov random field)로 강제한 GPLRF(Gaussian process latent random field)를 제안했습니다.
다른 사람들이 수행한 PCA의 더 많은 확장은 sparse PCA, robust PCA, probabilistic relational PCA 등이 있습니다.

LDA는 학습된 저차원 subspace에서 같은 class에 속한 데이터는 가깝고 다른 class에 속한 데이터들은 멀어지도록 강제하는 supervised, 선형 feature learning 방법론입니다.
LDA는 face recognition에서 성공적으로 사용되었고 Fisherfaces라고 불리는 새로운 feature를 얻었습니다.
GDA는 LDA의 kernel 버전입니다.
일반적으로 LDA와 GDA는 일반화된 고유값 분해로 학습됩니다.
그러나 Wang은 일반화된 고유값 분해의 solution이 LDA의 수식적 관점에서 original trace ratio 문제를 근사할 수 있다는 점을 지적했습니다.
그러므로 그들은 trace ratio 문제를 trace difference 시리즈 문제로 변형해 iterative 알고리즘을 사용하여 풀었습니다.
게다가 Jia는 trace ratio 문제를 풀기위해 수렴이 증명된 새로운 Newton-Raphson 방법론을 제안했습니다.
Zhong, Shi, Cheriet는 trace ratio formulation에 기반하고 데이터의 relational 정보를 충분히 탐험하는 relational Fisher analysis라고 불리는 새로운 방법론을 제안했습니다.
Zhong, Ling은 trace ratio 문제를 위해 iterative 알고리즘을 분석했고 필요성을 증명하고 trace ratio 문제의 최적 솔루션의 존재를 위한 충분한 조건을 증명했습니다.
게다가 LDA의 확장에는 incremental LDA, DGPLVM, MFA(marginal Fisher analysis)가 있습니다.

위에서 언급한 feature learning 알고리즘을 제외하고 ICA(Independent component analysis), CCA(canonical correlation analysis), ensemble learning based feature extraction, multitask feature learning과 같은 많은 다른 feature learning 방법론이 있습니다.
게다가 tensor 데이터를 직접 처리하기위해, 많은 tensor representation learning 알고리즘이 제안되었습니다.
예를들어 Yang는 2DPCA 알고리즘을 제안해 face recognition 문제에 PCA의 이점을 보였습니다.
Ye, Janardan, Li는 two-order tensor representation learning에 대한 2DLDA 알고리즘을 제안했습니다.
특히, [55] 논문에서 이론적인 수렴이 보장되는 large margin low rank tensor representation learning 알고리즘이 소개되었습니다.

#### Maninfold learning

이번장에서 우리는 manifold learning이라고 불리는 locality 기반의 feature learning 방법론에 집중할 것입니다.
비록 대부분의 manifold learning 알고리즘이 비선형 차원 축소 접근법이지만 locality preserving projections이나 MFA와 같은 몇몇 선형 차원 축소 방법론도 있습니다.
한편 몇 비선형 차원 축소 알고리즘은 마치 Sammon mapping이나 KPCA같은 고차원 데이터의 본질적인 구조를 발건하는 것이 목적이 아닌 manifold learning 방법론이 아닌것에 주목합니다.

2000년에 "Science"는 manifold learning에 대한 두가지 흥미로운 페이퍼를 출판했다.
첫 페이퍼는 고전적인 MDS에 Floyd-Warshall 알고리즘을 결합한 Isomap이다.
Isomap은 Floyd-Warshall 알고리즘을 사용해서 데이터 사이의 pairwise 거리를 계산하고 계산된 pairwise 거리에 대한 고전적인 MDS를 사용하여 데이터의 저차원 임베딩을 학습합니다.
두번째 페이퍼는 LLE에 대한 것으로 각 포인트에 대한 locality 정보를 이웃의 가중치를 재구성하도록 인코딩합니다. 
이후, 많은 manifold learning 알고리즘이 제안되었습니다[61,62,63,64,65,59,23,66].
특히 [67]의 연구는 LTSA(Local Tangent Space Alignment)의 아이디어를 LE(Laplacian Eigenmaps)과 결합한 것으로 local tangent space에서 유클리디안 거리를 사용한 데이터의 local 유사도를 구하고 데이터의 저차원 임베딩을 학습하기 위해 LE를 사용했습니다.
[68] 연구에서 Bheriet은 manifold learning 접근법을 모양 기반의 역사적인 Arabic 문서 인지에 사용했고 이전의 방법론들보다 주목할만한 성과를 얻었습니다.

위에 언급된 방법론들과 더불어 어느정도 데이터의 기반 구조를 고려하는 distance metric learning, semi-supervised learning, dictionary learning, non-negative matrix factorization과 같은 알고리즘들이 주목받았습니다.

## Deep Learning 


