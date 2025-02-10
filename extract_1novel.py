import json
import os
import re
import zipfile
import rarfile
import argparse

MIN_CHAPTERS = 10  # 固定最小章节数要求


def clean_text(text):
    """清理文本中的无效字符"""
    return text.replace('�', '')


def detect_and_decode(content_bytes):
    """检测并解码二进制内容"""
    try:
        text = content_bytes.decode('gb18030', errors='replace')
        if '�' in text:
            if text.count('�') > 100:
                print(f"[WARN] gb18030 解码发现大量无效字符 ({text.count('�')}个)，尝试其他编码")
                try:
                    return content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    print("[INFO] UTF-8 解码失败，继续尝试其他编码")
                try:
                    return content_bytes.decode('utf-16')
                except UnicodeDecodeError:
                    print("[INFO] UTF-16 解码失败，尝试自动检测")
            else:
                print(f"[INFO] 使用 gb18030 解码，包含少量替换字符 ({text.count('�')}个)")
        return clean_text(text)
    except Exception as e:
        print(f"[ERROR] gb18030 解码失败: {str(e)}")
        return None


def process_compressed_file(file_path):
    """处理压缩文件中的小说内容"""
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".zip":
            with zipfile.ZipFile(file_path, 'r') as z:
                txt_files = [name for name in z.namelist() if name.lower().endswith(".txt")]
                if not txt_files:
                    print(f"[WARN] {file_path} 内没有 txt 文件")
                    return None
                content = detect_and_decode(z.read(txt_files[0]))
        elif ext == ".rar":
            with rarfile.RarFile(file_path, 'r') as r:
                txt_files = [name for name in r.namelist() if name.lower().endswith(".txt")]
                if not txt_files:
                    print(f"[WARN] {file_path} 内没有 txt 文件")
                    return None
                content = detect_and_decode(r.read(txt_files[0]))
        else:
            print(f"[ERROR] 不支持的压缩格式: {ext}")
            return None
    except Exception as e:
        print(f"[ERROR] 处理 {file_path} 时出错：{e}")
        return None

    if content is None:
        return None

    return process_content(content)


def process_txt_file(file_path):
    """处理txt文件中的小说内容"""
    try:
        with open(file_path, 'rb') as f:
            content = detect_and_decode(f.read())
            if content is None:
                return None
            return process_content(content)
    except Exception as e:
        print(f"[ERROR] 处理文件 {file_path} 时出错：{e}")
        return None


def check_line_type(line):
    """检查行的类型"""
    number_pattern = r'(?:(?:[零一二三四五六七八九十百千万\d]+)|(?:\d+))'
    volume_pattern = f"^[【\[]*(?:第{number_pattern}卷|卷{number_pattern})(?:[：:\s]\s*\S+)?[】\]]*$"
    chapter_pattern = f"第{number_pattern}[章节]|[章节]{number_pattern}"

    if re.match(volume_pattern, line.strip()):
        return 1
    if re.search(chapter_pattern, line.strip()):
        return 2
    return 0


def process_content(content):
    """处理小说文本内容"""
    lines = content.splitlines()

    # 去除开头
    header_index = None
    limit = min(20, len(lines))
    indices = [i for i in range(limit) if "正文" in lines[i]]
    if indices:
        header_index = max(indices)
    else:
        indices = [i for i in range(limit) if "简介" in lines[i]]
        if indices:
            header_index = max(indices)
    if header_index is not None:
        lines = lines[header_index + 1:]
    else:
        lines = lines[10:] if len(lines) > 10 else []

    # 去除广告和作者注
    lines = [line for line in lines if "www." not in line.lower() and "ps" not in line.lower()]

    # 去除末尾
    candidate = None
    start_index = max(len(lines) - 5, 0)
    for i in range(len(lines) - 1, start_index - 1, -1):
        if lines[i].strip() != "":
            candidate = i
            break
    if candidate is not None:
        if re.match(r"^=+", lines[candidate].strip()):
            remove_indices = {candidate, candidate - 1, candidate - 2}
            lines = [line for idx, line in enumerate(lines) if idx not in remove_indices]

    # 处理每行
    processed_lines = []
    chapter_count = 0

    for line in lines:
        sline = line.strip()
        if sline == "":
            continue

        line_type = check_line_type(sline)
        if line_type == 1:  # 单独的卷名行，直接忽略
            continue
        elif line_type == 2:  # 包含章节名的行，替换为回车
            processed_lines.append(("", True))
            chapter_count += 1
        else:  # 普通行
            new_line = re.sub(r'^\s+', '', sline) if re.match(r'^\s+', sline) else sline
            processed_lines.append((new_line, False))

    if chapter_count <= MIN_CHAPTERS:
        print(f"[WARN] 章节数({chapter_count})不足最小要求({MIN_CHAPTERS})")
        return None

    final_lines = [text for text, is_marker in processed_lines]
    final_text = "\n".join(final_lines)
    return clean_text(final_text)


def convert_to_jsonl(input_file, output_file):
    """将单个文件转换为JSONL格式"""
    print(f"[INFO] 正在处理：{input_file}")

    ext = os.path.splitext(input_file)[1].lower()
    text = None

    if ext in ('.zip', '.rar'):
        text = process_compressed_file(input_file)
    elif ext == '.txt':
        text = process_txt_file(input_file)
    else:
        print(f"[ERROR] 不支持的文件格式: {ext}")
        return False

    if text is None:
        print("[ERROR] 文本处理失败")
        return False

    try:
        with open(output_file, 'w', encoding="utf-8") as out_f:
            obj = {"text": text}
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[INFO] 已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"[ERROR] 保存文件时出错：{e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='小说文本转换工具')
    parser.add_argument('-i', '--input', required=True, help='输入文件（支持.zip、.rar、.txt）')
    parser.add_argument('-o', '--output', required=True, help='输出的JSONL文件')

    args = parser.parse_args()

    if convert_to_jsonl(args.input, args.output):
        print("[INFO] 转换完成！")
    else:
        print("[ERROR] 转换失败！")


if __name__ == "__main__":
    main()