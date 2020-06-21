
# Simple Chit-Chat based on KoGPT2 

## Purpose

- 공개된 한글 체팅 데이터와 pre-trained `KoGPT2`를 이용한 간단한 쳇봇 구축
- `KoGPT2`의 다양한 활용 가능성 확인

## Architecture


<img src="imgs/simsimi.png" alt="drawing" style="width:600px;"/>


- 데이터의 Q를 `<usr>`발화 A를 `<sys>`발화로 매핑하고 $P(<sys>|<usr>)$를 최대화 할 수 있는 모델을 학습한다.


## Install

- `MXNet`은 [링크](https://mxnet.apache.org/get_started?)를 참고하여 시스템에 맞는 버전으로 설치한다.

```python
pip3 install mxnet-cuXX
pip3 install gluonnlp, sentencepiece, pandas
pip3 install git+https://github.com/SKT-AI/KoGPT2#egg=kogpt2
git clone --recurse-submodules https://github.com/haven-jeon/KoGPT2-chatbot.git

```


## How to Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --num-epoch 2 --train
```

## How to Chat!

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --chat
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
