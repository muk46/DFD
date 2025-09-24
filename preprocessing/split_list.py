# split_list.py
def split_file(input_file, n_parts):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    part_size = len(lines) // n_parts
    for i in range(n_parts):
        start = i * part_size
        end = (i + 1) * part_size if i < n_parts - 1 else len(lines)
        part_lines = lines[start:end]

        output_file = f"video_list_part{i+1}.txt"
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.writelines(part_lines)
        print(f"✅ {output_file} 생성 (총 {len(part_lines)}개)")

if __name__ == "__main__":
    split_file("video_list.txt", 16)  # 4등분