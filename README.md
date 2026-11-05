# Norwegian News Sentiment

Predicting how positive or negative a Norwegian text is, as a score from **-1 (very negative)** to **1 (very positive)**.

The models are trained on ~43,000 Norwegian reviews from [NoReC](https://github.com/ltgoslo/norec). Each review's 1–6 rating is turned into a sentiment score. The best model is a fine-tuned [NB-BERT](https://huggingface.co/NbAiLab/nb-bert-base), and it is then used to score newspaper articles from Norsk Aviskorpus, to see how the tone of Norwegian news changes over time and between papers.

**Built with:** Python, PyTorch, Hugging Face Transformers, Optuna, scikit-learn

## Approach

- **Data:** review texts are lowercased and whitespace is cleaned up. Ratings 1–6 are mapped linearly to [-1, 1].
- **Models** (all are regression models trained with MSE loss):
  1. **Mean baseline:** always predicts the average training score.
  2. **BiLSTM:** a bidirectional LSTM that starts from NB-BERT's word embeddings.
  3. **Fine-tuned NB-BERT:** NB-BERT with a linear regression head on the `[CLS]` token. This is the final model.
- **Tuning:** learning rate, dropout, weight decay and batch size were searched with Optuna, with pruning to stop bad trials early.

## Results

Validation set, from the best epoch of the logged training run:

| Model | RMSE (-1 to 1 scale) | R² | Avg. error in rating points (1–6) |
|---|---|---|---|
| Mean baseline | ≈ 0.45 | – | ≈ 1.1 |
| Fine-tuned NB-BERT | **0.257** | **0.62** | **≈ 0.64** |

NB-BERT cuts the baseline error by more than 40%. The training curves are in [`notebooks/history_visualization.ipynb`](notebooks/history_visualization.ipynb) and the raw log is in [`logs/`](logs/).

## Project structure

```
├── scripts/
│   ├── baseline_model.py             # mean baseline
│   ├── run_training.py               # train the BiLSTM
│   ├── run_training_transformer.py   # fine-tune NB-BERT
│   ├── optuna_tuning*.py             # hyperparameter search
│   ├── run_evaluation*.py            # test-set evaluation
│   └── get_sentiment.py              # score news articles with the trained model
├── src/
│   ├── data/                         # loading, preprocessing, PyTorch dataset
│   ├── models/                       # BiLSTM and transformer regression models
│   ├── training/                     # training loop and early stopping
│   └── utils/                        # config loading, seeding, class weights
├── notebooks/                        # data exploration and training curves
├── logs/                             # training history of the final model
├── BiLSTM_params.yaml
└── transformer_params.yaml
```

## Getting started

```bash
pip install -r requirements.txt
```

Download [NoReC](https://github.com/ltgoslo/norec) and put it in `data/` like this:

```
data/
├── metadata.json   # document id -> rating
├── train/          # one review per file, named by document id
├── dev/
└── test/
```

Train a model (a GPU is strongly recommended for NB-BERT):

```bash
python scripts/baseline_model.py
python scripts/run_training_transformer.py   # or run_training.py for the BiLSTM
```

Each run is saved to `logs/run_<timestamp>/` with its config, training history and weights. To evaluate a model or use it, copy its weights into `src/models/` (`transformer_model.pth` or `BiLSTM_model.pth`) and run:

```bash
python scripts/run_evaluation_transformer.py
python scripts/get_sentiment.py   # reads data/all_articles_from_norsk_aviskorpus.csv (id, date, paper, text)
```
