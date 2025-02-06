import argparse
import ast
import math
from pathlib import Path

import torch


def safe_eval(s):
    """安全地评估字符串表示的字面量"""
    try:
        return ast.literal_eval(s)
    except:
        return s


def load_vocab(vocab_file):
    """根据RWKV的tokenizer/rwkv_vocab_v20230424.txt的格式加载词表"""
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                parts = line.strip().split(' ')
                if len(parts) < 2:
                    print(f"警告：第{line_no}行格式不正确: {line.strip()}")
                    continue

                # 第一个部分是索引，最后一个部分是长度
                idx = int(parts[0])
                token = ' '.join(parts[1:-1])
                token = safe_eval(token)  # 使用更安全的eval
                vocab[idx] = token
            except Exception as e:
                print(f"警告：处理第{line_no}行时出错: {line.strip()}")
                print(f"错误: {str(e)}")
                continue
    return vocab


def create_token_mapping(old_vocab, new_vocab):
    """创建token的映射关系"""
    mapping = {}
    unmapped_tokens = []

    # 创建反向映射以加速查找
    old_token_to_idx = {token: idx for idx, token in old_vocab.items()}

    for new_idx, token in new_vocab.items():
        if token in old_token_to_idx:
            mapping[new_idx] = old_token_to_idx[token]
        else:
            unmapped_tokens.append((new_idx, token))

    if unmapped_tokens:
        print("警告：以下token在原词表中不存在：")
        for new_idx, token in unmapped_tokens[:10]:  # 只显示前10个
            try:
                print(f"  索引 {new_idx}: {repr(token)}")
            except:
                print(f"  索引 {new_idx}: <无法显示的token>")
        if len(unmapped_tokens) > 10:
            print(f"  ... 还有 {len(unmapped_tokens) - 10} 个token")

    return mapping


def convert_model_vocab(old_model_path, new_model_path, old_vocab_file, new_vocab_file):
    print(f"加载词表...")
    old_vocab = load_vocab(old_vocab_file)
    new_vocab = load_vocab(new_vocab_file)

    print(f"旧词表大小: {len(old_vocab)}")
    print(f"新词表大小: {len(new_vocab)}")

    # 创建token映射,保留0索引作为文档分隔符
    token_mapping = {0: 0}  # 确保0索引映射到0
    token_mapping.update({
        new_idx: old_idx
        for new_idx, token in new_vocab.items()
        for old_idx, old_token in old_vocab.items()
        if token == old_token and new_idx != 0 and old_idx != 0  # 跳过0索引
    })

    if len(token_mapping) != len(new_vocab) + 1:  # +1是因为要包含0索引
        print(f"警告：新词表中有{len(new_vocab) + 1 - len(token_mapping)}个token在原词表中不存在！")
        proceed = input("是否继续？[y/N] ")
        if proceed.lower() != 'y':
            print("操作已取消")
            return

    print(f"加载模型...")
    model_state = torch.load(old_model_path, map_location='cpu')

    # 获取模型配置
    n_embd = model_state['emb.weight'].shape[1]
    head_qk = 0
    if 'head_q.weight' in model_state:
        head_qk = model_state['head_q.weight'].shape[0]

    # 转换embedding层
    print(f"转换embedding层...")
    old_emb = model_state['emb.weight']
    new_emb = torch.empty((len(new_vocab) + 1, n_embd),  # +1为文档分隔符预留空间
                          dtype=old_emb.dtype, device=old_emb.device)

    # 使用原始的初始化方法
    scale = 1e-4
    with torch.no_grad():
        new_emb.uniform_(-scale, scale)

    # 复制已有的embedding,包括0索引
    for new_idx, old_idx in token_mapping.items():
        try:
            new_emb[new_idx] = old_emb[old_idx]
        except Exception as e:
            print(f"警告：复制embedding时出错 new_idx={new_idx}, old_idx={old_idx}")
            print(f"错误: {str(e)}")
            continue

    model_state['emb.weight'] = new_emb

    # 转换输出层
    print(f"转换输出层...")
    old_head = model_state['head.weight']
    new_head = torch.empty((len(new_vocab) + 1, n_embd),  # +1为文档分隔符预留空间
                           dtype=old_head.dtype, device=old_head.device)

    # 计算scale
    if len(new_vocab) + 1 > n_embd:  # +1考虑文档分隔符
        scale = 0.5 * math.sqrt((len(new_vocab) + 1) / n_embd)
    else:
        scale = 0.5
    if head_qk > 0:
        scale = 0.1

    with torch.no_grad():
        std = scale / math.sqrt(n_embd)
        new_head.normal_(0, std)

    # 复制已有的输出层权重,包括0索引
    for new_idx, old_idx in token_mapping.items():
        try:
            new_head[new_idx] = old_head[old_idx]
        except Exception as e:
            print(f"警告：复制输出层权重时出错 new_idx={new_idx}, old_idx={old_idx}")
            print(f"错误: {str(e)}")
            continue

    model_state['head.weight'] = new_head

    print(f"保存模型到 {new_model_path}...")
    torch.save(model_state, new_model_path)
    print("转换完成！")

def main():
    parser = argparse.ArgumentParser(description='RWKV模型词表转换工具')
    parser.add_argument('--old-model', type=str, required=True, help='原模型路径')
    parser.add_argument('--new-model', type=str, required=True, help='新模型保存路径')
    parser.add_argument('--old-vocab', type=str, required=True, help='原词表文件路径')
    parser.add_argument('--new-vocab', type=str, required=True, help='新词表文件路径')

    args = parser.parse_args()

    # 确保所有必要的文件都存在
    for file_path in [args.old_model, args.old_vocab, args.new_vocab]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

    # 确保输出目录存在
    Path(args.new_model).parent.mkdir(parents=True, exist_ok=True)

    convert_model_vocab(args.old_model, args.new_model, args.old_vocab, args.new_vocab)


if __name__ == '__main__':
    main()