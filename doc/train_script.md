# Train Script

Training model with [train_gpt](code/gpt/train) or [fed](doc/fed.md) is controlled by the script written on DSL. Script consists of variable assignments and operations. Script looks like this:

```
VARIABLE = 100 # comment
OPERATION(12, 'asdf')
```

## Variables

* **MODEL_DIMS = 'e512tt128d39'**
String, containing model dimensions, see below

* **TRAIN_CONFIG = 'b16ww1024'**
String, containing batch parameters, see below

* **DROP_CONFIG = 'drop1ch1'**
String containing dropout and learning rate parameters, see below

## Model operations

* **create_model(flag1, flag2..)**
Create new model. Takes list of MPF_* flags on input. See MPF_* flags description below

* **load_model('filename.bin')**
Load model from binary file. Loads model saved to binary files by gpt_train during training if SAVE_MODEL is set.

* **load_fed_model('filename.avg')**
Load model from binary file in fed format. Loads model saved by fed_center.

* **reset_model_grad()**
Set accumulated gradients to zero. Needed if training continues with different parameters, f.e. batch size is changed.

## gpt_train specific variables and operations

* **DEVICE_COUNT = 4**
Set number of GPUs to use, default is 1

* **MAX_ITERS = 50000**
Set train iterations count

* **EVAL_INTERVAL = 1000**
Train&test log loss is computed every EVAL_INTERVAL iteration. 

* **EVAL_ITERS = 20**
Number of iterations that are used to compute train&test log loss. Loss is computed on fixed set of batches, this introduces some bias but greatly reduces variance, which is better for comparing different model and training method changes.

* **SAVE_MODEL = false**
If SAVE_MODEL is set model is saved on disk every EVAL_INTERVAL iterations, otherwise only achieved loss is displayed. Default is yes.

* **PRINT_ITER_TRAIN_ERR = true**
If set gpt_train prints train logloss on each iteration. Default is no.

* **load_checkpoint(N)**
N - number of iteration to load model from. Can be used to continue aborted for some reason training run.

* **train()**
Train model on local host. MAX_ITERS iterations are performed. Train and test log loss scores are reported along the training process.

* **net_train('workers.txt')**
Perform sync distributed model train. List of worker IP address is loaded from text file provided. One IP address on each line is expected. 

* **compute_exact_test(Ncheckpoint, Navrg)**
Load model from Ncheckpoint iteration. If Navrg is non zero then load all models in [Ncheckpoint - Navrg; Ncheckpoint] range and average them. Sample random test batches, report average score over all sampled batches.

* **compute_choice_score('hellaswag_val.bin')**
Compute mmlu-like score, requires precomputed binary file with test queries. Expects query file in llm.c format, [import script](pysrc/hellaswag).

* **check_cpu_gpu_match()**
Compute discrepancy between reference fp32 cpu implementation and gpu implementation.

## TRAIN_CONFIG

**TRAIN_CONFIG** – string combining batch parameters, it has form “bXXfYY”.

* bXX – use XX fragments per batch

* fYY – use YY tokens per fragment

## MODEL_DIMS

**MODEL_DIMS** – string combining model configuration parameters, it has form “eNNhHHdZZwXXreluRR”. Omitted parameters have default value.

* eNN – set size of embedding and state vectors to NN, must be multiple of 128. Default is 256

* hHH - use HH heads per layer, head Q/K/V dimensions is fixed at 128. Default is 1.

* dZZ – set model depth, must be even

* wWW - set attention window size to WW

* reluRR - set relu layer state size multiple, default is 1. FFN state size on each layer has dimension RR x head_count x 128.

## DROP_CONFIG

**DROP_CONFIG** – string combining several training parameters, it has form “dropXXchYYlrZZslowSSregRRtailNN”. Omitted parameters have default value.

* dropXX – keep XX fraction of tokens, replace others with special <?> token

* chYY – dropout, keep YY parameters intact, zero out the rest (same set for each layer to improve regularization effect)

* lrZZ – use learning rate ZZ, default is 0.01

* slowSS - slow start, linearly increase learning rate for the first SS iterations. Default is 1000.

* regRR - add L2 regularization, the higher RR the more regularization. Default is no L2 regularization.

* tailNN – linearly reduce learning rate at the training finish, learning rate is reduced from lr at MAX_ITERS * (1 – 1/NN) to 0 at MAX_ITERS

## create_model() flags

Some flags:

* MPF_TAIL_LOSS – compute logloss on second half of the window (to get more realistic values for small windows)

* MPF_GROK_BINARY_OP – use special code in few place to experiment with modulo 97 arithmetic dataset

* MPF_SIM_QUANT_2BIT – experimental, simulate 2-bit model parameters quantization

## Script examples

There are few example train scripts at [cfg](cfg) folder. To load train script from file run gpt_train with '-s train_script.txt' argument. Shortest valid train script:

```
create_model(MPF_TAIL_LOSS)
train()
```
