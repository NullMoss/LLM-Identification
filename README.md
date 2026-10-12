# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project. 

**Team leader: Joshua Wong (individual project)**

Different LLMs have different linguistic, stylistic, and structural “fingerprints” in their responses. Can we differentiate the AI behind the text using machine learning?

## Results at a glance

Test set: 7,315 answers from 17 LLMs to 440 questions never seen in training
(chance = 0.059).

| Research question | Finding |
|---|---|
| **RQ1:** Can we identify which LLM generated a response? | Yes. BiLSTM 0.636 macro-F1, TextCNN 0.632, TF-IDF + Logistic Regression baseline 0.723. |
| **RQ2:** Does the user's prompt help? | No. Prompt-only models are at chance, and adding the prompt to the response slightly lowers macro-F1. 94% of questions were sent to all 17 LLMs. |
| **RQ4 (extra credit):** What distinguishes the LLMs? | Mostly response length, word choice and punctuation. Formatting carries almost no signal in this dataset, and broken-encoding characters are a dataset artifact. |

## Dataset

[TXD-22: A Large-Scale Benchmark Dataset](https://www.kaggle.com/datasets/mdsadiqiqbal/txd-22-a-large-scale-benchmark-dataset)
(Kaggle). One CSV with columns `Question`, `Answer` and `Source`: 66,000 answers
from 22 sources. Five sources are human-written or mixed human/AI, so they are
removed, leaving **17 LLMs and 50,980 answers**. The notebooks download the data
automatically with `kagglehub`; the data itself is not stored in this repo.

## Models

| Model | Description |
|---|---|
| Majority class | Always predicts the most common LLM (chance floor) |
| TF-IDF + Logistic Regression | Word 1–2-gram baseline |
| **TextCNN** | Embedding → Conv1d (kernels 3/4/5) → max-pool → linear |
| **BiLSTM** | Embedding → bidirectional LSTM (packed) → final states → linear |

Training: cross-entropy, AdamW (lr 1e-3), batch size 64, up to 10 epochs,
early stopping on validation macro-F1.

## How to run

**Google Colab (recommended)**

1. Open a notebook in Colab (*File → Open notebook → GitHub*, then paste this
   repo's URL).
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*.

Each notebook is self-contained. Every setting is in the `Config` class near
the top, and all figures and tables are written to `results/`.
