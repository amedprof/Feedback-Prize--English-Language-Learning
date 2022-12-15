# Feedback-Prize--English-Language-Learning
3rd-Place-Solution for Feedback-Prize---English-Language-Learning.

Competition [Link](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview)

The solution write-up [Link](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)

# Models
| order selected by Hill Climbing   | backbone      | cv score      |
| -------------   | ------------- |-------------  |
| 1	              | deberta-v3-large |	0.447     |
| 2	              | deberta-v3-large-squad2	| 0.4524 |
| 3	              | deberta-v3-large	| 0.4498 |
| 4	              | deberta-large-mnli	| 0.4548 |
| 5	              | deberta-v3-large	| 0.4492 |
| 6	              | xlm-roberta-large	| 0.4575 |
| 7	              | deberta-v3-large-squad2	| 0.457 |
| 8	              | deberta-v3-large-squad2	| 0.4525 |
| 9	              | deberta-v3-large	| 0.4489 |
| 10	             | deberta-v3-large-squad2	| 0.4565 |
| 11	             | RAPIDS-SVR	| 0.4526 |
| 12	             | deberta-v3-large	| 0.4495 |
| 13	             | deberta-v3-large	| 0.4502 |
| 14	             | TF-deberta-v3-base	| 0.4554 |
| 15	             | deberta-v3-large	| 0.4527 |
| 16	             | deberta-v3-large	| 0.4522 |
| 17	             | deberta-v3-large	| 0.45 |
| 18	             | deberta-v3-large	| 0.4516 |
| 19	             | deberta-v3-large	| 0.4509 |
| 20	             | deberta-v3-base	| 0.4575 |
| 21	             | deberta-v3-large-squad2	| 0.4501 |
| 22	             | deberta-v3-large	| 0.4512 |
| 23	             | roberta-large	| 0.4571 |
|24	              | deberta-v3-large	| 0.45 |

# Install
Build docker image from [Dockerfile](https://github.com/Amed1710/Feedback-Prize--English-Language-Learning/blob/main/Dockerfile)

# Data
In order to run the code, you will need to download the competition [data](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data) and extract it in the **data** folder.

# Training a model
To run experiments : 

For non deberta models :
 1. you should change `model_name` in `configs/non_deberta_config.yaml` 
 2. run `python train.py --config configs/non_deberta_config.yaml`

For deberta models :

 1. you should change `model_name` in `configs/deberta_v3_large.yaml` 
 2. run `python train.py --config configs/deberta_v3_large.yaml`

For 2xPooling models:
 1. you should use one of `configs/deberta-v3-large-2xpooling-paragraph.yaml` , `configs/deberta-v3-large-2xpooling-sentences.yaml` , `configs/deberta-v3-large-2xpooling-words.yaml`
 2. run `python train.py --config configs/deberta-v3-large-2xpooling-paragraph.yaml`
 
For non PyTorch models :
 1. run rapids-svr-cv-0-450.ipynb in `src/model_zoo folder`
 2. run tf-deberta-v3-base-cv-0-455.ipynb in `src/model_zoo folder`
 
If you wish to only train PyTorch models, skip these two and set `NUM_MODELS = 22` in inference code.

# Inference
To reproduce our final score, run this [code](https://www.kaggle.com/code/cdeotte/3rd-place-solution-lb-0-4337-cv-0-4420) from kaggle kernels.

# Hardware
Models were trained using ZbyHP Z8 workstation with Ubuntu 20.04.1 LTS.
