import json
from pathlib import Path
from collections import Counter
from rwkv_tokenizer import TRIE_TOKENIZER


def analyze_tokens(vocab_path: str, jsonl_path: str):
    # 确保文件存在
    vocab_path = Path(vocab_path)
    jsonl_path = Path(jsonl_path)

    if not vocab_path.exists():
        raise FileNotFoundError(f"找不到词表文件: {vocab_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到JSONL文件: {jsonl_path}")

    # 初始化tokenizer（使用完整的词表路径）
    tokenizer = TRIE_TOKENIZER(str(vocab_path))

    # 读取原始词表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    # 创建token使用计数器
    token_counter = Counter()

    # 处理jsonl文件
    print("正在处理jsonl文件...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    text = json.loads(line)["text"]
                    tokens = tokenizer.encode(text)
                    token_counter.update(tokens)

                    if line_num % 1000 == 0:
                        print(f"已处理 {line_num} 行")
                except Exception as e:
                    print(f"处理第 {line_num} 行时出错: {str(e)}")
                    continue

    # 获取使用过的token集合
    used_tokens = set(token_counter.keys())
    print(f"\n共发现 {len(used_tokens)} 个不同的token")

    # 创建新词表
    new_vocab_lines = []
    for line in original_lines:
        # 解析原始行
        idx = int(line[:line.index(' ')])
        if idx in used_tokens:
            new_vocab_lines.append(line)

    # 生成输出文件名（在原词表同目录下）
    output_path = vocab_path.parent / f"{vocab_path.stem}_{len(used_tokens)}.txt"

    # 保存新词表
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_vocab_lines)

    print(f"\n新词表已保存至: {output_path}")
    print(f"原词表大小: {len(original_lines)}")
    print(f"新词表大小: {len(new_vocab_lines)}")

    # 输出使用频率最高的前10个token
    print("\n使用频率最高的10个token:")
    for token, count in token_counter.most_common(10):
        try:
            token_bytes = tokenizer.idx2token[token]
            if isinstance(token_bytes, bytes):
                token_str = token_bytes.decode('utf-8', errors='replace')
            else:
                token_str = str(token_bytes)
        except:
            token_str = f"<token_{token}>"
        print(f"Token {token}: {token_str} (使用次数: {count})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("用法: python script.py <词表文件路径> <jsonl文件路径>")
        print("示例: python script.py ./rwkv_vocab_v20230424.txt ./test.jsonl")
        sys.exit(1)

    vocab_path = sys.argv[1]
    jsonl_path = sys.argv[2]
    analyze_tokens(vocab_path, jsonl_path)