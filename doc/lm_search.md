# Longest Match from the history

Regular text predicting transformer takes embedding of the current character/token as input. For absolute position encoding embedding of position is added to the input. This information about position helps transformer to predict next token better. With this mechanism we can provide any additional useful information to the transformer - just add embeddings of additional features which might be useful for the transformer task - predicting text.

Text compression can be cast as text prediction task. So we can try to leverage some text compression techniques to create useful features for transformers. For example, we can take one of the PPM family algorithms and feed features from it to transformer along with character & position embeddings.

One of the strongest features for text compression is the longest match continuation. [LZ77](https://en.wikipedia.org/wiki/LZ77_and_LZ78) algorithm is based on this idea. To compute it we search for the exact prefix match in history - same fragment of text as the last N characters. Our prediction is that the next character will be the same as the one followed found exact prefix match. Among multiple matches the longest match performs best. So to compute this feature for each train and test token in our dataset we have to scan preceding tokens, find the longest exact match and feed embedding of it's continuation to the transformer.

TScale support two ways to perform this. With limited window - it's called PPM and unlimited - called LMS. PPM and LMS can be enabled with USE_PPM and USE_LMATCH options in [data script](doc/data_script.md).

Unlimited longest match search requires plenty of memory. Just storing precomputed longest match continuation doubles dataset size. Keeping index for quick lookup during inference requires much more memory. Provided basic implementation uses sorted prefix arrays and consumes 8 bytes per token for index.

Online longest match search over fineweb-EDU requires about 1T ram, which is beyond reach for mere mortals on single host. To compute test scores for LMS models provided code precomputes longest match for test dataset and loads index chunks from disk to compute hellaswag score.
