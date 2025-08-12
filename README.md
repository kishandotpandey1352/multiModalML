
# multiModal — Byte-level fine-tuning (AG News)

This mirrors your **byte pre-training**:
- Inputs are **0..255** bytes
- PAD/MASK token = **255**
- Encoder = your `ByteEncoder(config)` (mean-pooled)
- Avoids namespace collision by importing Hugging Face as `import datasets as hf_datasets` inside our local `datasets/*`

## Layout
```
multiModal/
├── encoder/byte_encoder.py
├── heads/byte_classifier.py
├── datasets/agnews_hf.py
├── utils/config.py
├── train_task.py           # universal trainer
└── tasks/fine_tune_agnews.py
```

## Quick start
```bash
pip install datasets
python tasks/fine_tune_agnews.py
# or
python train_task.py
```

It will try to load your pre-trained encoder from `checkpoints/V1/model.pth` and fine-tune on Hugging Face AG News.
Checkpoints land in `checkpoints/`:
- `shared_encoder_finetuned.pth`
- `agnews_head.pth`
and best_* variants based on validation accuracy.
