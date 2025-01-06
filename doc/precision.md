# Supported compute precision

TScale supports training and inference with fp16, fp8 and int8 precision for matmul operations and fp16 and fp8 precision for attention compute.

## Matmul performance on 4090

| Type | Tops/sec|
|------|---------|
|int8  |526      |
|fp8   |310      |
|fp16 with fp16 accum|261|
|fp16 with fp32 accum|159|

Note that fp8 matmuls are 2x slower then advertised in nVidia whitepapers. Slowdown of fp8 and fp16+fp32 accum matmuls look artificial. nSight and other nVidia tools correctly indicate 50% tensor unit load for them. Other AD102 SKUs like L40 work full speed.

From the table we can see that using int8 as much as possible is the key to good performance on 40xx series gpus. Using int8 in forward pass is straightforward for TScale architecture since all vectors are normalized before applying matmuls. Backward pass is more involved since gradients posess much larger dynamic range. However if we consider gradients for each token separately then we can approximate them with int8 much better by using per token gradient scale. Per token gradient scaling and dithering achieve stable and precise enough training with int8 gradients.

Computing attention and especially attention gradients with int8 is surprisingly hard. Unclear if its possible to get stable training with int8 attention.

## Precision loss for int8/fp8 precision

Computing forward/backward passes with reduced precision leads to imprecise results. To measure how much precision is lost TScale code contains reference fp32 cpu implementation of forward/backward passes, [gpt_cpu](../code/gpt/gpt_cpu.cpp). Table below summarizes precision loss in basepoints for different matmul/attention precision.

|Model params|State|Matmul|Attention|Forward precision|Backward+Forward precision|
|-----------------|------|---------|-----------|----------------------|-------|
|fp16|fp16|fp16|fp16|0.2 bp| 1.7 bp|
|fp16|int8|fp16|fp16|3.9 bp| 21.5 bp|
|fp16|e4m3|fp16|fp16|7.4 bp| 21.9 bp|
|int8|int8|int8|fp16|3.6 bp| 31.2 bp|
|int8|int8|int8|fp8|4.4 bp| 23.9 bp|
|e4m3|e4m3|fp8|fp16|7.1 bp|24.1 bp|
|e4m3|e4m3|fp8|fp8|7.9 bp|26 bp|

A bit counterintuitive result is that int8 path precision is better then fp8 path. This happens due to values distribution which is closer to normal rather then to exponential. For example, model parameters approximation with int8 is 2.5x more precise then with e4m3. 

## fp8 attention kernels perf

TScale uses optimized fp8 attention kernels.
|Direction|Effective TFlops|
|--|--|
|forward|220|
|backward (gradients)|166|

Gradient kernels work slower because they compute result in two passes. In first pass they compute maximum gradient per token to be able to use e4m3 precision for result computation. By using careful dynamic range analysis it might be possible to estimate dynamic range without explicit computation, by now it remains work TBD.

## Model training performance

Training performance measurements performed on 4090. Table below contains 10 iteration time for 24k tokens batches for 1.5B model (e1280h20d42). Attention width is 1024 tokens.

| Matmul|Attention|Time|
|-------|---------|-----------------------------|
|int8   |fp8      | 12.8 sec                    |
|fp8    |fp8      | 14.8 sec                    |
|fp16   |fp16     | 25.6 sec                    |

Comparison with llm.c in terms of tera parameters per second (ModelSize x BatchSize x BatchCount / Time) on 1.5B model:

|Config|TParams/sec|
|--|--|
|llm.c, H100|72|
|llm.c, A100|28|
|TScale, 4090|28|
