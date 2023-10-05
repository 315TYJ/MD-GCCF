# MD-GCCF（pytorch）
## Introduction

In this work, we proposed a multi-view deep graph contrastive learning for collaborative filtering (MD-GCCF) from two perspectives to resolve some issues in the field of collaborative filtering.

## Enviroment Requirement

`pip install -r requirements.txt`



## Dataset

We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book.

see more in `dataloader.py`

## An example to run a 3-layer MD-GCCF

run MD-GCCF on **Gowalla** dataset:

* command

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="gowalla" --topks="[20]" --recdim=64`
