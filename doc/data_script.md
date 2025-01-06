# Data Script

Data used in training is described with training DSL. Data script consists of variable assignments and operations. Script looks like this:

```
VARIABLE = 100 # comment
OPERATION(12, 'asdf')
```

## Variables

* **TEST_FRACTION = 0.01**
Fraction of dataset used for test, default is 0.05

* **USE_PPM = true**
If USE_PPM is set [PPM](ppm.md) is enabled. Default is no.

* **USE_LMATCH = true**
If USE_LMATCH is set [longest match search](lm_search.md) is enabled. Default is no.

## Model operations

* **create_model(flag1, flag2..)**
Create new model. Takes list of MPF_* flags on input. See MPF_* flags description below

* **load_model('filename.bin')**
Load model from binary file. Model is saved to binary files during training if SAVE_MODEL is set.

* **load_checkpoint(N)**
N - number of iteration to load model from. Can be used to continue aborted for some reason training run.

## Tokenizer operations

* **set_vocab_size(N, K)**
N - token count, used with load_tokenized_* functions. N = 50257 for gpt2 tokenizer. Adds K additional tokens (current implementation adds only one additional token which specifies fragment start with any K specified).

* **set_doc_start_token(N)**
N - token id of doc delimiting token. Can be used with load_tokenized_* functions.

* **load_tokenizer('filename.bin')**
Load tokenizer from binary file. [Tokenizer](tokenizer) binary file can be created with [gpt_tokenizer](../code/gpt/tokenizer).

* **make_byte_tokenizer()**
Create byte tokenizer. Generates one token per bytes, uses 256 different tokens, one for each byte value. 

## Dataset operations

* **make_char_dataset('shakespear.txt')**
Loads single text file. Creates tokenizer out of used bytes in this text file. Splits file into two parts, first part is used  for train, second for test. 

* **connect_data_server('11.22.33.97')**
Connect to data server located on host with specified ip address. To run data server compile and launch [data_server](../code/gpt/data_server) with appropriate config.

* **connect_http_data_server('11.22.33.97')**
Connect to data server located on specified ip address over http. Example of such [server](../code/gpt/http_data_server) written in python.

* **load_tokenized_train('train.bin')**
Load tokenized dataset and add it to train set. Argument specifies binary file with sequence of tokens. Each token is stored as ui16.
 
* **load_tokenized_test('test.bin')**
Load tokenized dataset and it it to test set.

* **load_text('doc1.txt')**
Load text file, tokenize it with selected tokenizer and add to dataset. Code assumes utf8 encoding of the text.

* **load_folder('cultura')**
Load all files in the specified folder and add them to dataset. Each file is considered a text document.

* **load_docset('cultura/2.bin')**
Load document pack and add each document to dataset. Document packs can be created with [hf import](../pysrc/hf_import_text). Document pack is  a binary file consisting of serialized documents. Each document has 4 byte header with document length followed by utf8 encoded text of the document.

* **index_docset_folder('cultura')**
Tokenize and create PPM and LM features if needed for all document packs in the specified folder. Stores result to index.bin and index_hdr.bin files. Can be used to preprocess large datasets once and then use them to train models.

* **index_tokenized_folder('cultura', token_width, header_size)**
Same as index_docset_folder() but files are treated as binary files with tokens. Token width can be 2, 3, 4. First header_size bytes are omitted

* **load_indexed_docset_folder('cultura')**
Load tokenized with inde_docset_folder() documents and add them to dataset. The only way to work with document collections which do not fit into RAM is to index them with index_docset_folder() and then load them with load_indexed_docset_folder().

* **set_lmatch_index_folder('data/lmatch/cultura')**
Specify location of longest match index files.

## Script examples

There are few example data scripts in [cfg folder](../cfg). To load data script from file run gpt_train with '-d data_script.cfg' argument. Shortest valid data script:

```
make_char_dataset('shakespear.txt')
```
