#!/usr/bin/env bash

VOCAB_SIZE=10000
MODELNAME="seq2seq"
CUDA_VISIBLE_DEVICES=2 python -u ../nmt.py \
    --cuda \
    --sup_train_src_path train.ja.tok.low \
    --sup_train_tgt_path train.ja.tok.low \
    --valid_src_path dev.ja.tok.low \
    --valid_tgt_path dev.en.tok.low \
    --test_src_path test.ja.tok.low \
    --test_tgt_path test.en.tok.low \
    --src_vocab_size ${VOCAB_SIZE} \
    --tgt_vocab_size ${VOCAB_SIZE} \
    --batch_size 64 \
    --beam_size 5 \
    --valid_freq 5 \
    --hidden_dim 256 \
    --input_feed ctx \
    --normal_dropout 0.2 \
    --att_type D \
    --init_dec cell \
    --drop_h 1 \
    --drop_emb 0.2 \
    --uniform_init 0.1 \
    --pad_src_sent \
    --sort_src \
    --display_freq 100 \
    --valid_freq 1000 \
    --valid_metric ppl \
    --model_name ${MODELNAME} 2>&1 | tee ${MODELNAME}.log