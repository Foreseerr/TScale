# saves hf dataset to a binary file for training

import os
import struct
from datasets import load_dataset # huggingface datasets, pip install datasets

file_id = 0
dst_folder = 'D:/text/open_web_text/'
file = None
file_len = 0

def process(example):
    global file_id
    global file
    global file_len

    if file is None or file_len > 150000000:
        if file:
            file.close()
        file = open(f'{dst_folder}{file_id}.bin', 'wb')
        file_id += 1
        file_len = 0
        
    # UTF-8 encoding by default
    text = example['text'].encode()
    
    # write len as 32-bit unsigned integer
    data_len = len(text)
    file.write(struct.pack('I', data_len))
    
    # write text
    file.write(text)
    
    #increment len
    file_len += data_len


if __name__ == '__main__':
    # some datasets require token
    token = os.environ.get("HF_TOKEN", "")
    dataset = load_dataset("openwebtext")
    #dataset = load_dataset("ontocord/CulturaY", "ru", token=token)
    #dataset = load_dataset("danasone/librusec", token=token)
    dataset.map(process)
    if file:
        file.close()
