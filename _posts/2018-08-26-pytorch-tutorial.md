---
layout: post
title: Pytorch Tutorial
category: Unclassified
---

# python2, 3 호환
from __future__ import print_function

# uniform distribution
torch.rand(5,3)

# normal distribution
torch.randn(5,3)

# type 추적
torch.zeros(5,3).type()

# numpy to torch
a = np.ones(5)
b = torch.from_numpy(a)

# cuda 사용가능여부
torch.cuda.is_available()

# cuda 메모리에 변수 올리기
y = torch.ones_like(x,device=torch.device("cuda"))
x = x.to(torch.device("cuda"))

# cuda에서 cpu로 메모리로 변환
a.to(torch.device("cpu"))

# 특정 값으로 변수 채우기
a.fill_(3.5)

# torch tensor와 numpy array는 같은 메모리 위치를 참조하기 때문에 하나가 변하면 다른것도 변함
a = torch.ones(5)
b = a.numpy()
a.add_(1)
print(a)
print(b)

#크기 10인 torch.tensor를 3으로 채우고 GPU할당
torch.full((10, ), 3, device=torch.device("cuda"))

# gradient 계산에 변수를 포함하려면 requires_grad=True로 지정해야 함
x = torch.ones(2,2, requires_grad=True)
y = x+2

# gradient 계산, 입력값이 scalar가 아니면 명시적으로 적어줘야 함
y.backward(x)
print(x.grad)
