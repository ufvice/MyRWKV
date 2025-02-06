import csv
import re


def convert_token_file_to_csv(input_file, output_file):
    # 用于存储转换后的数据
    rows = []

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过空行
            if not line.strip():
                continue

            # 使用正则表达式解析每行
            # 匹配格式: 数字 '内容' 数字
            match = re.match(r"(\d+)\s+'(.*)'\s+(\d+)", line)

            if match:
                index = match.group(1)  # 第一个数字
                token = match.group(2)  # 引号中的内容
                length = match.group(3)  # 第二个数字

                # 将解析的数据添加到rows列表
                rows.append([index, token, length])

    # 将数据写入CSV文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['Index', 'Token', 'Length'])
        # 写入数据
        writer.writerows(rows)


# 使用示例
if __name__ == "__main__":
    input_file = "rwkv_vocab_v20230424.txt"  # 输入的txt文件名
    output_file = "rwkv_vocab_v20230424.csv"  # 输出的csv文件名

    try:
        convert_token_file_to_csv(input_file, output_file)
        print(f"转换成功！结果已保存到 {output_file}")
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")