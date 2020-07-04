

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Simple Chit-Chat based on KoGPT2](#simple-chit-chat-based-on-kogpt2)
  - [Purpose](#purpose)
  - [Architecture](#architecture)
  - [Install](#install)
  - [How to Train](#how-to-train)
  - [How to Chat!](#how-to-chat)
  - [Colab tutorials](#colab-tutorials)
  - [TODO](#todo)

<!-- /code_chunk_output -->


# Simple Chit-Chat based on KoGPT2 

## Purpose

- [공개된 한글 챗봇 데이터](https://github.com/songys/Chatbot_data)와 pre-trained [KoGPT2](https://github.com/SKT-AI/KoGPT2)를 이용한 간단한 챗봇 실험
- `KoGPT2`의 다양한 활용 가능성을 타진하고 성능을 정성적으로 평가한다.

## Architecture

학습 데이터에 적합하게 [Hello! GPT-2](https://arxiv.org/abs/1907.05774)를 응용, 아래와 같은 아키텍처를 설계함

<img src="imgs/simsimi.png" alt="drawing" style="width:600px;"/>


- 데이터의 Q 필드를 `<usr>`발화, A 필드를 `<sys>`발화 그리고 감정 레이블을 `<sent>`로 매핑해 `P(<sys>|<usr>, <sent>)`를 최대화 할 수 있는 모델을 학습한다. 감정 레이블은 [이곳](https://github.com/songys/Chatbot_data)의 정의를 따른다(일상다반사 0, 이별(부정) 1, 사랑(긍정) 2).

## Install


```python
# MXNet은 [링크](https://mxnet.apache.org/get_started?)를 참고하여 시스템에 맞는 버전으로 설치한다.
pip3 install mxnet-cuXX
pip3 install gluonnlp sentencepiece pandas torch transformers pytorch_lightning 
pip3 install git+https://github.com/SKT-AI/KoGPT2#egg=kogpt2
git clone --recurse-submodules https://github.com/haven-jeon/KoGPT2-chatbot.git
```

## How to Train

- PyTorch

```bash
CUDA_VISIBLE_DEVICES=0 python train_torch.py --gpus 1 --train --max_epochs 2
```


- MXNet

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --num-epoch 2 --train
```

## How to Chat!

```bash
# CUDA_VISIBLE_DEVICES=0 python train.py --chat
CUDA_VISIBLE_DEVICES=0 python train_torch.py --gpus 1 --chat
user > 방가워용.
Simsimi > 저도요.

user > 어버이날엔 뭘 하죠?
Simsimi > 부모님께 효도하는 것도 좋을 것 같아요.

user > 용돈이 너무 적어요.
Simsimi > 아껴 쓰세요.

user > 미친듯이 고민해도 방법이 없네요.
Simsimi > 좋은 방법이 있을 거예요.

user > 큼지막한 방에서
Simsimi > 혼자만의 시간을 가지세요.

user > 큰돈을 만드는 방법은?
Simsimi > 돈을 많이 버는 것뿐이에요.

user > 집은 너무 비싼거 같아요.
Simsimi > 비싸다고 해서 다 좋은 건 아니에요.
```

## Colab tutorials

- PyTorch
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haven-jeon/KoGPT2-chatbot/blob/master/KoGPT2_chatbot_pytorch.ipynb)
- MXNet
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haven-jeon/KoGPT2-chatbot/blob/master/KoGPT2_chatbot.ipynb)



## TODO

- ~~감정 정보를 조건으로 준 응답 생성~~
- ~~PyTorch 버전 작업~~
- 다양한 generation 방식 테스트

