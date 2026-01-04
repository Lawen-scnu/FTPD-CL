import os
import re
import shutil

def extract_figures_with_titles_and_descriptions(markdown_file,output_dir):
    # 正则表达式：匹配图片和标题
    image_pattern = r'!\[(.*?)\]\((.*?)\)'  # 匹配图片
    title_pattern1 = r'^\*?(?:图\s*\d+|Figure\s*\d+|Fig\.\s*\d+|FIGURE\s*\d+|FIG\.\s*\d+)(.*)$'
    title_pattern2 = r'^\*+(?:图\s*\d+|Figure\s*\d+|Fig\.\s*\d+|FIGURE\s*\d+|FIG\.\s*\d+)(.*)$'
    figure_number_pattern = r'(Figure\s*\d+|Fig\.\s*\d+|图\s*\d+FIGURE\s*\d+|FIG\.\s*\d+)'  # 提取图表编号

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取Markdown文件的目录和文件名
    base_dir = os.path.dirname(markdown_file)
    markdown_filename = os.path.splitext(os.path.basename(markdown_file))[0]

    # 创建以Markdown文件名为基础的文件夹
    output_dir = os.path.join(output_dir, markdown_filename)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    with open(markdown_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # 分割Markdown内容成段落
    paragraphs = re.split(r'\n+', content)
    extracted_data = []

    # 遍历所有段落，找到图片和标题
    for i, paragraph in enumerate(paragraphs):
        # 查找图片
        image_match = re.match(image_pattern, paragraph)
        if image_match:
            image_name = image_match.group(1)  # 图片的名称
            image_path = image_match.group(2)  # 图片的路径



            # 将图片路径调整为Markdown文件所在目录的绝对路径
            image_path = os.path.join(base_dir, image_path)

            # 查找紧接着图片的标题，跳过空行，并确保不是标题时退出查找
            title = None
            for j in range(i + 1, len(paragraphs)):
                next_paragraph = paragraphs[j].strip()

                # 如果是空行，跳过
                if not next_paragraph:
                    continue

                # 如果找到了标题，则保存并退出查找
                title_match = re.match(title_pattern1, next_paragraph)
                if not title_match:
                    title_match = re.match(title_pattern2, next_paragraph)
                if title_match:
                    title = title_match.group().strip()
                    # print(next_paragraph)
                    break  # 找到标题后停止查找

                # 如果该非空行不是标题，则停止查找
                else:
                    break

            if title:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # 提取图表编号（如 "Figure 6", "Fig. 6"）
                figure_number = re.search(figure_number_pattern, title)
                figure_number = figure_number.group() if figure_number else "Unknown"


                # 构造所有可能的图表编号格式，包含 "Figure 6" 和 "Fig. 6"
                modal = re.findall(r'[A-Za-z]+', figure_number)
                modal = modal[0]
                # possible_figures = [
                #     figure_number,  # 原始编号
                #     figure_number.replace("Figure", "Fig."),  # 变更为 "Fig. X"
                #     figure_number.replace("Fig.", "Figure")  # 变更为 "Figure X"
                # ]
                possible_figures = [
                    figure_number.replace(modal, modal),   # 原始编号
                    figure_number.replace(modal, "Fig"),  # 变更为 "Fig. X"
                    figure_number.replace(modal, "Figure"),  # 变更为 "Figure X"
                    figure_number.replace(modal, "FIG"),  # 变更为 "Fig. X"
                    figure_number.replace(modal, "FIGURE")  # 变更为 "Figure X"
                ]
                possible_figures = list(dict.fromkeys(possible_figures))
                # print(possible_figures)


                # 找到与当前图片标题相关的描述性语句
                description_paragraphs = []
                for j, other_paragraph in enumerate(paragraphs):
                    # 查找包含图片编号的段落（包括所有可能的图表编号格式）
                    if any(fig_number in other_paragraph for fig_number in possible_figures):
                        description_paragraphs.append(other_paragraph.strip())

                # 创建基于图表编号的文件夹
                figure_dir = os.path.join(output_dir, figure_number)
                if not os.path.exists(figure_dir):
                    os.makedirs(figure_dir)

                # 提取后缀并重命名
                suffix = os.path.splitext(image_path)[1]
                new_imgname = f"{figure_number}{suffix}"

                # 保存图片到该文件夹
                if os.path.exists(image_path):
                    shutil.copy(image_path, os.path.join(figure_dir, new_imgname))

                # 保存描述性语句到文件
                description_file = os.path.join(figure_dir, f"mention.txt")
                with open(description_file, 'w', encoding='utf-8') as desc_file:
                    for paragraph in description_paragraphs:
                        desc_file.write(paragraph + '\n')

                caption_file = os.path.join(figure_dir, f"caption.txt")
                with open(caption_file, 'w', encoding='utf-8') as cap_file:
                    cap_file.write(next_paragraph + '\n')

                # 将提取的数据保存
                extracted_data.append({
                    '图片名称': image_name,
                    '图片路径': image_path,
                    '图表编号': figure_number,
                    '图表标题': title,
                    '描述性语句': description_paragraphs
                })
    





# 主程序
folder_path = os.getcwd()
main_folder = os.path.join(folder_path, 'path1')
output_dir = os.path.join(folder_path, 'path2')# 替换为你的Markdown文件路径

for root, dirs, files in os.walk(main_folder):
    for file in files:
        # 只处理.md格式的文件
        if file.endswith('.md'):
            markdown_file = os.path.join(root, file)
            # print(f"正在处理文件: {markdown_file}")
            extract_figures_with_titles_and_descriptions(markdown_file,output_dir)
print("done!")
     
            
            
                


