import os
import shutil
import xlrd
import re


def read_novel_data(xls_path):
    """Read and process novel ratings from XLS file."""
    workbook = xlrd.open_workbook(xls_path)
    worksheet = workbook.sheet_by_index(0)

    # Get header row to find column indices
    headers = [worksheet.cell_value(0, i) for i in range(worksheet.ncols)]
    title_idx = headers.index('Title')
    xiancao_idx = headers.index('XianCao')
    liangcao_idx = headers.index('LiangCao')
    gancao_idx = headers.index('GanCao')
    kucao_idx = headers.index('KuCao')
    ducao_idx = headers.index('DuCao')

    qualified_novels = set()

    # Process each row starting from row 1 (skipping header)
    for row_idx in range(1, worksheet.nrows):
        title = worksheet.cell_value(row_idx, title_idx)
        xiancao = float(worksheet.cell_value(row_idx, xiancao_idx) or 0)
        liangcao = float(worksheet.cell_value(row_idx, liangcao_idx) or 0)
        gancao = float(worksheet.cell_value(row_idx, gancao_idx) or 0)
        kucao = float(worksheet.cell_value(row_idx, kucao_idx) or 0)
        ducao = float(worksheet.cell_value(row_idx, ducao_idx) or 0)

        total_ratings = xiancao + liangcao + gancao + kucao + ducao
        xiancao_ratio = xiancao / total_ratings if total_ratings > 0 else 0

        # Check if novel meets either criteria
        if (total_ratings > 400) or (total_ratings <= 400 and xiancao_ratio > 0.6 and xiancao > 10):
            qualified_novels.add(title)

    return qualified_novels


def normalize_title(title):
    """Convert title to lowercase and remove common decorators for matching."""
    # Remove common decorators and whitespace
    title = re.sub(r'[《》（）()：:\s]', '', title.lower())
    return title


def copy_matching_files(source_dir, dest_dir, qualified_novels):
    """Copy files that match qualified novels to destination directory."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Create a normalized set of qualified novel titles for matching
    normalized_qualified = {normalize_title(title) for title in qualified_novels}

    matched_count = 0
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.rar', '.zip')):
            # Extract the title part before the author
            title_match = re.match(r'[《]?(.+?)[》]?(?:（|[\(])', filename)
            if title_match:
                file_title = normalize_title(title_match.group(1))
                if file_title in normalized_qualified:
                    source_path = os.path.join(source_dir, filename)
                    dest_path = os.path.join(dest_dir, filename)
                    shutil.copy2(source_path, dest_path)
                    matched_count += 1
                    print(f"Copied: {filename}")

    return matched_count


def main():
    # Configuration
    xls_path = '/mnt/c/Users/ufvice/Documents/ranking.xls'  # Path to your XLS file
    source_dir = '/mnt/c/Users/ufvice/Documents/zxcs（7681）'  # Directory containing the novel files
    dest_dir = 'selected_novels'  # Directory to copy selected novels to

    try:
        # Read and process the XLS data
        print("Reading novel ratings data...")
        qualified_novels = read_novel_data(xls_path)
        print(f"Found {len(qualified_novels)} qualified novels")

        # Copy matching files
        print("\nCopying matching files...")
        matched_count = copy_matching_files(source_dir, dest_dir, qualified_novels)
        print(f"\nOperation complete. Copied {matched_count} files to {dest_dir}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()