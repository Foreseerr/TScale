# Training 125M model on OWT for 50B tokens

Setup for this train run tries to reproduce the one used in [H3 paper (table 3)](https://arxiv.org/pdf/2212.14052) - train 125M model on Open Web Text for 50B tokens.

Open Web Text dataset tokenized with gpt2 tokenizer was taken from [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master/data/openwebtext). Model dimensions are e512 h3 d50 (state width 512, 3 heads per layer, 50 layers). Model was trained for 510k batches, each batch 96x1024 token fragments.

Train/val loss graphs:
![Nice train loss graph](img/tl125M.png)

Graphs are smooth since train and test losses here are computed over fixed fragment set. No moving average or the like is applied.

Val set perplexity, results with (*) taken from H3 paper:
|Model|Perplexity|Hellaswag
|------|---------|--|
|H3 (*)|21.0 |-
|Transformer (*)|20.6|-
|H3 Hybrid (2 Attn) (*)|19.6|-
|TScale|17.4|32.2
