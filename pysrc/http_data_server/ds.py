from flask import Flask
from flask import request
import json
import random
import struct

# constants
train_file_name = 'D:/111enwiki9/gpt2_train.bin'
server_port = '32323'

# open train data file
train_file = open(train_file_name, 'rb')
train_file.seek(0, 2)
train_size = train_file.tell() // 2

# run flask server
app = Flask(__name__)

@app.route("/stats")
def rq_stats():
    stats = {}
    stats['UsePPM'] = False
    stats['Compression'] = 1
    stats['VocabSize'] = 50257
    stats['DocStartToken'] = 50256
    stats['HasTest'] = False
    bias = [0] * stats['VocabSize']
    stats['Bias'] = bias
    stats['UsePPM'] = False
    return json.dumps(stats)


@app.route("/fragments")
def rq_fragments():
    trt = request.args.get('trt', default = 0, type = int)
    seed = request.args.get('seed', default = 0, type = int)
    fcount = request.args.get('count', default = 1, type = int)
    flen = request.args.get('len', default = 64, type = int)
    # ignore seed, trt
    res = []
    for _ in range(fcount):
        rsize = flen + 1
        max_pos = train_size - rsize
        if max_pos < 0:
            raise ValueError("File is too small to read the requested number of shorts.")
        start_pos = random.randint(0, max_pos)

        train_file.seek(start_pos * 2)
        data = train_file.read(rsize * 2)
        shorts = struct.unpack(f'{rsize}H', data)
        
        frag = {}
        frag['Text'] = shorts[:-1]
        frag['Target'] = shorts[1:]
        res.append(frag)
    return json.dumps(res)


if __name__ == "__main__":
    #app.run(host='localhost', port='32323')
    app.run(host='0.0.0.0', port=server_port)
