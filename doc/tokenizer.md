# Tokenizer

[Tokenizers](doc/tokenizer.md) are created by [gpt_tokenizer](/code/gpt/tokenizer).

Code uses custom text tokenizers. Purpose of these experimental tokenizers is to optimize logloss on test, not to use as few tokens as possible. There are several tokenizers:
- **TK_CHAR**
Tokenizes each byte separately, assign tokens only to used in train bytes
- **TK_WORD**
Looks like to improve model quality it is beneficial to tokenize each word separately. This is the simplest per word tokenizer. It assigns separate token to the most frequent words, the rest text is encoded byte-by-byte
- **TK_GREEDY**
More complex variant of word tokenizer. It assigns tokens to frequent word parts. This tokenizer splits text into words and spaces. Spaces are encoded byte-by-byte. For each word it finds longest token which is prefix of the word, outputs it and repeats process with what is left of the word. 
- **TK_GREEDY_CAPITAL**
This tokenizer adds special CAPITAL token. For words that start with capital letter it outputs CAPITAL token and then encodes lowercased word. This encodes removes same word tokenization difference for word encoding at the start of the sentence and in the middle. 

