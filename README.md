﻿
# TScale

This repo contains transformer train and inference code written in C++ and CUDA.

TScale is designed to run on consumer hardware. To achive best results it features
- Optimized transformer architecture with faster convergence and ~2x reduced attention costs
- Support for fp8 and int8 model weights and activations precision
- Optimized for consumer nVidia GPUs including fast reduced precision training without sacrificing model quality
- CPU offload reduces GPU memory requirements for training
- Sync distributed training on several same config hosts
- 1-bit gradient compression allowing using regular ethernet links for interconnect
- Async distributed training on arbitrary hosts with negligible network traffic. In this mode training can be run on geographically separated hosts

# Distributed training of 1.5B model on consumer GPU

By using inexpensive GPUs and async distributed mode TScale trains LLMs fast and affordable. Log loss for the 1.5B model trained on fineweb-edu for 2 days and $500 on several spot instances with 4090:
![Nice train loss graph](/img/fed_hellaswag.png)

# Training your own 1T model at home

1T model size sounds beyond reach for most people and even organisations. However if we consider creative ways to count model size then there is nothing impossible. In this case we build a model with 1T index which we lookup for every token to make prediction with much smaller model. In terms of logloss/perplexity this construction easily achieves stellar results. Index for fineweb-edu occupies about 1T of disk space. Training run of 125M model with this ~1T index achieves **x8** perplexity reduction:

|Model|Perplexity|
|-----|-|
|125M |19.02|
|125M + 1T index|2.28|

# Read more

[Training 125M model](doc/125M_model.md)

[Training 1.5B model](doc/1.5B_model.md)

[Training 1T (!) model in your kitchen](doc/1T_model.md)

[Async distributed train](doc/fed.md)

[Notes on model and compute precision](doc/precision.md)

[TScale transformer model](doc/model.md)

[Data indexing](doc/lm_search.md)

[Tokenizer](doc/tokenizer.md)

# Build

To build the the code CUDA v12.3 and C++ compiler are required, msvc for windows,  cmake+clang for Linux. To support cross platform build files generation this repo uses [fo](doc/fo.md), lightweight solution/build files generator. To generate build files you need to compile fo/fo.cpp and run it with two arguments. First argument is root of source tree, second argument is directory to store build files to.

## Windows

```bash
D:\TScale>fo.exe code sln
```

Then open code.sln from d:\TScale\sln\code.sln.

## Linux

To compile TScale for linux you need to compile fo.cpp, generate CMakeLists.txt file, run cmake, run make.

```bash
~/TScale/fo$ clang++-18 fo.cpp -o fo
~/TScale/fo$ cd ..
~/TScale$ ./fo/fo code make.dir
~/TScale$ cd make.dir
~/TScale/make.dir$ cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo .
~/TScale/make.dir$ make
```

# Get train data

Examples in the code use [enwik9](https://mattmahoney.net/dc/textdata.html) dataset and its truncacted version enwik8. Also Hugging Face hosted datasets openwebtext, ontocord/CulturaY, danasone/librusec are used in examples. To import them use [hf_import](/pysrc/hf_import/import.py).

# Train model

[gpt_train](/code/gpt/train) is used to train a model. It is controlled by the [train script](/doc/train_script.md) and [data script](/doc/data_script.md). Default scripts are stored in [main_gpt.cpp](/code/gpt/train/main_gpt.cpp). To load train script from file run gpt_train with '-d data_script.txt -s train_script.txt' arguments.

## quick run

Compile gpt-train. Run it in the root directory:

```bash
~/TScale$ ./make.dir/gpt-train
```
 
## sync distributed run

Currently training can be distributed only among pow2 number of worker hosts. 

To start a worker process run gpt_train with '-w 10000' argument. 10000 specifies port number to use.

To run master process call net_train('worker.txt') function in train script. List worker IP addresses in the file provided to net_train().

## multiple GPU

To use multiple GPU devices set DEVICE_COUNT variable in train script to number of GPUs to use. For distributed runs DEVICE_COUNT is applied on each worker, heterogeneous configurations are not supported.

## scripts

Description of scripts used in training: [data script](doc/data_script.md), [train script](doc/train_script.md)


# Inference test

To try inferencing from the trained model you can use [gpt_infer](/code/gpt/infer). It runs basic http server on 11311 port and allows sampling continuations from the model. Current implementation is slow and designed for demonstration purposes only.

# License

MIT
