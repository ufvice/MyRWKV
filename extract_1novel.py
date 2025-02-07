import json
import os
import re
import zipfile
import rarfile
import argparse


def clean_text(text):
    """清理文本中的无效字符"""
    return text.replace('�', '')


def detect_and_decode(content_bytes):
    """检测并解码二进制内容"""
    encodings = ['gb18030', 'utf-8', 'utf-16']

    for encoding in encodings:
        try:
            text = content_bytes.decode(encoding)
            invalid_chars = text.count('�')
            if invalid_chars == 0:
                print(f"[SUCCESS] 使用 {encoding} 解码成功")
                return text
            elif invalid_chars <= 100 or encoding == encodings[-1]:
                print(f"[WARNING] 使用 {encoding} 解码，包含 {invalid_chars} 个无效字符")
                return clean_text(text)
            else:
                print(f"[INFO] {encoding} 解码发现 {invalid_chars} 个无效字符，尝试其他编码")
        except UnicodeDecodeError:
            print(f"[INFO] {encoding} 解码失败")

    return None


def process_content(content):
    """处理小说文本内容"""
    print("\n[INFO] 开始处理文本内容...")
    lines = content.splitlines()
    original_lines = len(lines)
    print(f"[INFO] 原始行数: {original_lines}")

    # 处理文件头部
    header_index = None
    limit = min(20, len(lines))
    for keyword in ["正文", "简介"]:
        indices = [i for i in range(limit) if keyword in lines[i]]
        if indices:
            header_index = max(indices)
            print(f"[INFO] 在第 {header_index} 行找到关键词 '{keyword}'")
            break

    if header_index is not None:
        lines = lines[header_index + 1:]
        print(f"[INFO] 移除了文件头部 {header_index + 1} 行")
    else:
        lines = lines[10:] if len(lines) > 10 else []
        print("[INFO] 未找到明确的开始标记，默认移除前10行")

    # 清理广告和注释
    original_len = len(lines)
    lines = [line for line in lines if "www." not in line.lower() and "ps" not in line.lower()]
    removed_lines = original_len - len(lines)
    print(f"[INFO] 清理广告和注释后移除了 {removed_lines} 行")

    # 处理章节
    processed_lines = []
    chapter_count = 0
    number_pattern = r'(?:(?:[零一二三四五六七八九十百千万\d]+)|(?:\d+))'
    volume_pattern = re.compile(f"^(?:第{number_pattern}卷|卷{number_pattern})")
    chapter_pattern = re.compile(f"^(?:第{number_pattern}[章节]|[章节]{number_pattern})")

    for line in lines:
        sline = line.rstrip()
        if volume_pattern.match(sline):
            continue
        if chapter_pattern.match(sline):
            processed_lines.append(("", True))
            chapter_count += 1
            continue
        if sline.strip() == "":
            continue
        new_line = re.sub(r'^\s+', '', sline) if re.match(r'^\s+', sline) else sline
        processed_lines.append((new_line, False))

    print(f"[INFO] 检测到 {chapter_count} 个章节")

    if chapter_count < 10:
        print(f"[ERROR] 章节数不足10章，可能不是完整小说")
        return None

    final_lines = [text for text, is_marker in processed_lines]
    final_text = "\n".join(final_lines)
    print(f"[INFO] 最终文本长度: {len(final_text)} 字符")

    return clean_text(final_text)


def process_file(file_path):
    """处理单个文件并返回处理后的文本"""
    print(f"\n[INFO] 开始处理文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"[ERROR] 文件不存在: {file_path}")
        return None

    ext = os.path.splitext(file_path)[1].lower()
    content = None

    try:
        if ext == ".zip":
            with zipfile.ZipFile(file_path, 'r') as z:
                txt_files = [name for name in z.namelist() if name.lower().endswith(".txt")]
                if not txt_files:
                    print(f"[ERROR] ZIP文件中没有找到TXT文件")
                    return None
                print(f"[INFO] 找到TXT文件: {txt_files[0]}")
                content = detect_and_decode(z.read(txt_files[0]))

        elif ext == ".rar":
            with rarfile.RarFile(file_path, 'r') as r:
                txt_files = [name for name in r.namelist() if name.lower().endswith(".txt")]
                if not txt_files:
                    print(f"[ERROR] RAR文件中没有找到TXT文件")
                    return None
                print(f"[INFO] 找到TXT文件: {txt_files[0]}")
                content = detect_and_decode(r.read(txt_files[0]))

        elif ext == ".txt":
            with open(file_path, 'rb') as f:
                content = detect_and_decode(f.read())

        else:
            print(f"[ERROR] 不支持的文件格式: {ext}")
            return None

    except Exception as e:
        print(f"[ERROR] 处理文件时出错: {str(e)}")
        return None

    if content is None:
        print("[ERROR] 无法解码文件内容")
        return None

    return process_content(content)


def main():
    parser = argparse.ArgumentParser(description='单文件小说转换工具')
    parser.add_argument('input', help='输入文件路径 (支持 .txt, .zip, .rar)')
    parser.add_argument('output', help='输出的JSONL文件路径')
    args = parser.parse_args()

    text = process_file(args.input)
    if text is None:
        print("\n[ERROR] 文件处理失败")
        return

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({"text": text}, f, ensure_ascii=False)
            print(f"\n[SUCCESS] 已保存到: {args.output}")
    except Exception as e:
        print(f"\n[ERROR] 保存文件时出错: {str(e)}")


if __name__ == "__main__":
    main()