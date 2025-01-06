FROM nvidia/cuda:12.4.0-base-ubuntu22.04

WORKDIR /usr/src/app
COPY gpt-fed_worker ./

#RUN apt-get update && apt-get install -y gdb

CMD ["sh", "-c", "./gpt-fed_worker -t 10 -n 32768 -c ${FED_CENTER}"]