#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用法示例：
    python make_data.py demo.jsonl 3 4096 [--vocab path/to/vocab.txt]
优化功能：
  - 检查并使用已有的临时分词文件，避免重复分词
  - 保留临时分词文件以供后续使用
  - 对源 jsonl 文件进行一次性分词，保存为二进制格式
  - 使用二进制文件进行多轮随机混洗
  - 生成最终的 .bin 和 .idx 文件
  - 计算 magic_prime
  - 支持自定义词表文件
优势：
  - 可重用已有的分词结果，显著提升处理速度
  - 分词只在必要时进行一次
  - 内存占用保持低水平
  - 混洗效果与原版一致
  - 灵活支持不同的词表
"""

import json
import math
import random
import sys
import os
import struct
import numpy as np
from pathlib import Path
import argparse

# -------------------------------
# 导入自定义模块
# -------------------------------
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from src.binidx import MMapIndexedDataset

tokenizer = None  # 全局变量，在main中初始化


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


# -------------------------------
# 检查临时分词文件是否存在且有效
# -------------------------------
def check_temp_files(temp_prefix):
    token_file = f"{temp_prefix}.tokens"
    offset_file = f"{temp_prefix}.offsets"

    if not (os.path.exists(token_file) and os.path.exists(offset_file)):
        return False

    try:
        # 尝试打开并读取文件头部来验证文件完整性
        with open(token_file, 'rb') as tf, open(offset_file, 'rb') as of:
            # 读取第一个offset记录
            offset_bytes = of.read(12)
            if not offset_bytes:
                return False
            offset, length = struct.unpack("<QI", offset_bytes)

            # 验证token文件中是否包含相应的数据
            tf.seek(0)
            token_bytes = tf.read(length * 2)
            if len(token_bytes) != length * 2:
                return False

        return True
    except Exception:
        return False


# -------------------------------
# 二进制token数据集的写入器
# -------------------------------
class TokenizedDatasetBuilder:
    def __init__(self, output_prefix):
        self.token_file = open(f"{output_prefix}.tokens", "wb")
        self.offset_file = open(f"{output_prefix}.offsets", "wb")
        self.current_offset = 0
        self.doc_count = 0

    def add_tokens(self, tokens):
        # 写入token序列
        token_bytes = struct.pack(f"<{len(tokens)}H", *tokens)
        self.token_file.write(token_bytes)

        # 写入该文档的offset和长度
        offset_record = struct.pack("<QI", self.current_offset, len(tokens))
        self.offset_file.write(offset_record)

        self.current_offset += len(tokens)
        self.doc_count += 1

        if self.doc_count % 500 == 0:
            print(f"Processed {self.doc_count} documents", end="\r")

    def finalize(self):
        self.token_file.close()
        self.offset_file.close()
        return self.doc_count, self.current_offset


# -------------------------------
# 二进制token数据集的读取器
# -------------------------------
class TokenizedDatasetReader:
    def __init__(self, prefix):
        self.token_file = open(f"{prefix}.tokens", "rb")
        self.offset_file = open(f"{prefix}.offsets", "rb")

        # 读取所有文档的offset信息到内存
        self.doc_offsets = []
        while True:
            offset_bytes = self.offset_file.read(12)  # 8(offset) + 4(length)
            if not offset_bytes:
                break
            offset, length = struct.unpack("<QI", offset_bytes)
            self.doc_offsets.append((offset, length))

    def get_doc_tokens(self, doc_idx):
        offset, length = self.doc_offsets[doc_idx]
        self.token_file.seek(offset * 2)  # uint16 = 2 bytes
        token_bytes = self.token_file.read(length * 2)
        return np.frombuffer(token_bytes, dtype=np.uint16)

    def __len__(self):
        return len(self.doc_offsets)

    def close(self):
        self.token_file.close()
        self.offset_file.close()


# -------------------------------
# MMapIndexedDatasetBuilder（原版不变）
# -------------------------------
class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


# -------------------------------
# 质数检测函数（优化版本）
# -------------------------------
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# -------------------------------
# 主处理函数
# -------------------------------
def process_data(in_file, n_epoch, vocab_file=None):
    global tokenizer
    prefix = Path(in_file).stem
    # 如果使用了自定义词表，在临时文件名中加入词表的标识
    if vocab_file:
        vocab_identifier = Path(vocab_file).stem
        temp_prefix = f"{prefix}_temp_{vocab_identifier}"
    else:
        temp_prefix = f"{prefix}_temp"
    final_prefix = prefix

    print(f"### Processing {in_file}")
    if vocab_file:
        print(f"### Using custom vocabulary: {vocab_file}")

    # 检查是否存在可用的临时分词文件
    if check_temp_files(temp_prefix):
        print("### Found existing tokenization files, loading...")
        reader = TokenizedDatasetReader(temp_prefix)
        total_docs = len(reader)
        total_tokens = reader.doc_offsets[-1][0] + reader.doc_offsets[-1][1]
        reader.close()
        print(f"Loaded {total_docs} documents with {total_tokens} tokens")
    else:
        print("### Phase 1: One-time tokenization...")
        # 第一阶段：一次性分词
        builder = TokenizedDatasetBuilder(temp_prefix)

        with open(in_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    text = json.loads(line)["text"]
                    tokens = tokenizer.encode(text)
                    # 验证分词结果
                    if tokenizer.decode(tokens) != text:
                        print("ERROR: Tokenization verification failed!")
                        continue
                    tokens.append(0)  # 文档结束标记
                    builder.add_tokens(tokens)
                except Exception as e:
                    print(f"\nError processing document: {e}")
                    continue

        total_docs, total_tokens = builder.finalize()
        print(f"\nTokenized {total_docs} documents, total tokens: {total_tokens}")

    print("\n### Phase 2: Shuffling and building final dataset...")
    # 第二阶段：随机混洗
    reader = TokenizedDatasetReader(temp_prefix)
    final_builder = MMapIndexedDatasetBuilder(f"{final_prefix}.bin")

    for epoch in range(n_epoch):
        print(f"\nEpoch {epoch + 1}/{n_epoch}: shuffling documents")
        # 只混洗文档索引
        doc_indices = list(range(len(reader)))
        random.shuffle(doc_indices)

        for doc_idx in doc_indices:
            tokens = reader.get_doc_tokens(doc_idx)
            final_builder.add_item(tokens)
            final_builder.end_document()

    final_builder.finalize(f"{final_prefix}.idx")
    reader.close()

    print("\n### Temporary tokenization files are preserved for future use")
    print(f"    - {temp_prefix}.tokens")
    print(f"    - {temp_prefix}.offsets")

    return total_tokens, total_docs


# -------------------------------
# 参数解析函数
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Process and tokenize text data')
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('n_epoch', type=int, help='Number of epochs')
    parser.add_argument('ctx_len', type=int, help='Context length')
    parser.add_argument('--vocab', help='Path to custom vocabulary file', default=None)
    return parser.parse_args()


# -------------------------------
# 主函数
# -------------------------------
if __name__ == "__main__":
    args = parse_args()

    # 初始化tokenizer，支持自定义词表
    vocab_path = args.vocab if args.vocab else "tokenizer/rwkv_vocab_v20230424.txt"
    tokenizer = TRIE_TOKENIZER(vocab_path)

    # 处理数据
    total_tokens, total_docs = process_data(args.input_file, args.n_epoch, args.vocab)

    # 验证输出
    print("\n### Verifying result...")
    data = MMapIndexedDataset(Path(args.input_file).stem)
    data_len = len(data)
    data_size = len(data._bin_buffer) // data._index._dtype_size

    # 验证首尾两个文档
    TODO = [0, data_len - 1]
    PREVIEW_LIMIT = 100
    for idx in TODO:
        ptr, size = data._index[idx]
        dix = data.get(idx=idx, offset=0, length=size).astype(int)
        print("-" * 70 + f"[{Path(args.input_file).stem} idx {idx} sz {size}]")
        assert dix[-1] == 0
        dix = dix[:-1]
        if len(dix) > PREVIEW_LIMIT:
            try:
                print(tokenizer.decode(dix[:PREVIEW_LIMIT]))
            except:
                try:
                    print(tokenizer.decode(dix[:PREVIEW_LIMIT + 1]))
                except:
                    print(tokenizer.decode(dix[:PREVIEW_LIMIT + 2]))
            print("· " * 30)
            try:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT:]))
            except:
                try:
                    print(tokenizer.decode(dix[-PREVIEW_LIMIT - 1:]))
                except:
                    print(tokenizer.decode(dix[-PREVIEW_LIMIT - 2:]))
        else:
            print(tokenizer.decode(dix))

    print(f"{'-' * 80}\n### Final output has {data_size} tokens, {data_len} documents. Dtype {data._index.dtype}")

    # 计算magic_prime
    if data_size >= args.ctx_len * 3:
        n_chunk = int(data_size // args.ctx_len) - 1
        # 调整起始点，使得 start % 3 == 2
        start = n_chunk - ((n_chunk - 2) % 3)
        magic_prime = None
        for i in range(start, 0, -3):  # 每次减3，只检查形如3k+2的数
            if is_prime(i):
                magic_prime = i
                break
        if magic_prime:
            print(f"\n### magic_prime = {magic_prime} (for ctxlen {args.ctx_len})")
            print(f'\n--my_exit_tokens {data_size} --magic_prime {magic_prime} --ctx_len {args.ctx_len}\n')
            exit(0)