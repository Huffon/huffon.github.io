---
layout: post
title: "PyTorch utils.prune 모듈로 BERT 다이어트 시키기"
subtitle: '다이어트는 사람한테만 필요한게 아니잖아?'
author: "devfon"
header-style: text
lang: kr
tags:
  - AI
  - NLP
  - Deep Learning
---

## 들어가며

![](/img/in-post/fatty.jpg)

최근 다이어트를 하고 있다. 컴퓨터 공부를 시작한 무렵부터 살이 급속도로 찌고 있었는데, 그동안 너무 안일하게 스스로를 방치한 것 같다. 지금에서야 스스로를 자책하며 **간헐적 단식**을 지키고 있는데 나만 다이어트를 하는 것은 억울하다. **BERT** 가족들이 그렇게 다이어트를 하고 싶다던데 얘네들도 같이 체중 감량을 할 필요가 있다. ML계의 [양치승](https://namu.wiki/w/%EC%96%91%EC%B9%98%EC%8A%B9) 관장이 되어 **BERT**를 다이어트 시켜보자.

이 글을 읽고 있는 독자라면 2018년 [BERT](https://arxiv.org/abs/1810.04805)의 등장 이후 자연어 처리 분야에서 사용되는 모델의 사이즈가 점점 더 커지고 있다는 사실쯤은 알고 있을 것이다. 그리고 이처럼 큰 크기의 모델의 파라미터 수를 줄이는 방법에는 **Quantization**, **Knowledge Distlilation**, **Pruning** 등 다양한 기법이 존재한다. 이번 포스트에서는 **PyTorch** 공식 튜토리얼 중 [**Pruning Tutoiral**]()을 참고해 Hugging face의 [**transformers**]() 라이브러리 내 BERT 모델을 다이어트 시키는 방법에 대해 다루고자 한다. 

![](/img/in-post/model-size.png)

## Pruning 이란?

Pruning은 쉽게 이야기해 **가지치기**와 같은 기법이다.

Networks generally look like the one on the left: every neuron in the layer below has a connection to the layer above, but this means that we have to multiply a lot of floats together. Ideally, we’d only connect each neuron to a few others and save on doing some of the multiplications; this is called a “sparse” network.

Sparse models are easier to compress, and we can skip the zeroes during inference for latency improvements.

![](/img/in-post/pruning.png)

If you could rank the neurons in the network according to how much they contribute, you could then remove the low ranking neurons from the network, resulting in a smaller and faster network.

Getting faster/smaller networks is important for running these deep learning networks on mobile devices.

The ranking, for example, can be done according to the L1/L2 norm of neuron weights. After the pruning, the accuracy will drop (hopefully not too much if the ranking is clever), and the network is usually trained-pruned-trained-pruned iteratively to recover. 

If we prune too much at once, the network might be damaged so much it won’t be able to recover. So in practice, this is an iterative process — often called ‘Iterative Pruning’: Prune / Train / Repeat.

## BERT 다이어트 시키기

우리는 바퀴의 동작 원리를 이해할 필요는 있지만, 재발명할 필요는 없다. **Pruning** 역시 이미 발명된 바퀴를 활용하면 쉽게 모델에 적용해볼 수 있다. **PyTorch**에는 `utils.prune` 이라는 이미 잘 짜여진 모듈이 있다. 해당 모듈은 사용자가 기작성한 모델에 자신이 원하는 가지치기 기법을 적용할 수 있도록 도와주는 모듈이다. 본 모듈을 활용하면 손쉽게 **Pruning**을 적용해볼 수 있다.

우리는 해당 모듈을 활용해 이제는 너무나도 친숙해진 허깅페이스의 **transformers** 라이브러리에서 잠자고 있는 **BERT**를 깨워 다이어트를 시켜볼 것이다. 먼저 운동에 필요한 기구들을 가져오도록 하자.

```python
import torch
import torch.nn.utils.prune as prune
from transformers import BertModel

model = BertModel.from_pretrained('monologg/kobert')
```
