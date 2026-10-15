"""
语义切分 + RecursiveCharacterTextSplitter 二次切割演示
按「段落 → 换行 → 句子 → 空格」优先级切割，不硬断句子
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# ========== 配置 ==========
MD_FILE = "output/20261014/ced0060b-4dee-44b8-9d03-f0de55b2b3a7/华为平板 C5 平板用户指南-(BZT3-W59&W69,EMUI10.1_01,ZH-CN)/华为平板 C5 平板用户指南-(BZT3-W59&W69,EMUI10.1_01,ZH-CN).md"
CHUNK_SIZE = 500       # 超过这个长度才二次切割
CHUNK_OVERLAP = 50     # 块之间重叠，保持上下文连贯

# 分隔符优先级：从大到小，优先在大边界切
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

# ========== 第一步：按 Markdown 标题做语义切分 ==========
def split_by_markdown_headings(md_text: str) -> list[dict]:
    """按 # / ## / ### 标题切分，每个块带层级信息"""
    blocks = []
    current_heading = "文档开头"
    current_lines = []

    for line in md_text.split("\n"):
        if line.startswith("#"):
            # 遇到新标题，保存上一个块
            if current_lines:
                blocks.append({
                    "heading": current_heading,
                    "content": "\n".join(current_lines).strip()
                })
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # 最后一个块
    if current_lines:
        blocks.append({
            "heading": current_heading,
            "content": "\n".join(current_lines).strip()
        })

    # 过滤空块
    return [b for b in blocks if b["content"]]


# ========== 第二步：对超长块用 RecursiveCharacterTextSplitter 二次切割 ==========
def is_mostly_html_table(text: str) -> bool:
    """判断内容是否主要是 HTML 表格（这类内容整块保留，不切割避免碎片）"""
    table_chars = text.count("<table>") + text.count("<tr>") + text.count("<td>")
    return table_chars > 10  # 表格标签超过 10 个就认为是表格块


def secondary_split(block: dict, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """对超过 chunk_size 的语义块做二次切割，返回多个子块"""
    content = block["content"]

    # 表格块整块保留，不切，避免产生碎片
    if is_mostly_html_table(content):
        return [{**block, "note": "表格块，整块保留"}]

    if len(content) <= CHUNK_SIZE:
        return [block]  # 不需要切

    sub_chunks = splitter.split_text(content)
    return [
        {
            "heading": block["heading"],
            "content": chunk,
            "is_split": True  # 标记：这是二次切割出来的
        }
        for chunk in sub_chunks
    ]


# ========== 主流程 ==========
def main():
    # 读取原始 MD
    with open(MD_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()

    print(f"原文档总长度：{len(md_text)} 字符")
    print(f"原文档前 500 字预览：\n{md_text[:500]}")
    print("\n" + "="*60 + "\n")

    # 第一步：语义切分
    semantic_blocks = split_by_markdown_headings(md_text)
    print(f"第一步：按 Markdown 标题切分出 {len(semantic_blocks)} 个语义块")
    print("\n各语义块长度：")
    for i, b in enumerate(semantic_blocks[:10]):
        marker = "⚠️ 超长" if len(b["content"]) > CHUNK_SIZE else "✅ 正常"
        print(f"  [{i+1}] {marker} {len(b['content']):4d} 字 | {b['heading'][:40]}")

    print("\n" + "="*60 + "\n")

    # 第二步：二次切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )

    final_chunks = []
    for block in semantic_blocks:
        final_chunks.extend(secondary_split(block, splitter))

    print(f"第二步：RecursiveCharacterTextSplitter 二次切割后共 {len(final_chunks)} 个块")
    print(f"切割规则：分隔符优先级 = {SEPARATORS}")
    print(f"chunk_size = {CHUNK_SIZE}, chunk_overlap = {CHUNK_OVERLAP}")

    print("\n" + "="*60 + "\n")
    print("二次切割示例（找一个被切分的块展示）：")
    split_examples = [c for c in final_chunks if c.get("is_split")]
    if split_examples:
        # 找同一个 heading 的连续子块
        same_heading = [c for c in final_chunks if c["heading"] == split_examples[0]["heading"]]
        print(f"\n原标题：{same_heading[0]['heading']}")
        print(f"这个语义块被切成了 {len(same_heading)} 个子块：\n")
        for i, chunk in enumerate(same_heading):
            print(f"  子块 {i+1}（{len(chunk['content'])} 字）：")
            print(f"  {chunk['content'][:200]}...")
            print()

    # 保存结果到文件
    output_file = "demos/chunk_demo_result.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# RAG 切分演示结果\n\n")
        f.write(f"## 原文档\n")
        f.write(f"- 文件：{os.path.basename(MD_FILE)}\n")
        f.write(f"- 总长度：{len(md_text)} 字符\n\n")
        f.write(f"### 原文档前 1000 字\n\n```\n{md_text[:1000]}\n```\n\n")

        f.write(f"## 第一步：按 Markdown 标题语义切分\n")
        f.write(f"共切出 {len(semantic_blocks)} 个语义块\n\n")
        for i, b in enumerate(semantic_blocks[:20]):
            marker = "⚠️ 超长" if len(b["content"]) > CHUNK_SIZE else "✅ 正常"
            f.write(f"- [{i+1}] {marker} {len(b['content'])} 字 | {b['heading'][:50]}\n")

        f.write(f"\n## 第二步：RecursiveCharacterTextSplitter 二次切割\n")
        f.write(f"- 分隔符优先级：段落(\\n\\n) → 换行(\\n) → 中文句号(。) → 逗号(，) → 空格 → 字符\n")
        f.write(f"- chunk_size = {CHUNK_SIZE}, chunk_overlap = {CHUNK_OVERLAP}\n")
        f.write(f"- 最终块数：{len(final_chunks)}\n\n")

        f.write(f"### 切割示例（同一个超长语义块被切成多个子块）\n\n")
        if split_examples:
            same_heading = [c for c in final_chunks if c["heading"] == split_examples[0]["heading"]]
            f.write(f"**原标题：** {same_heading[0]['heading']}\n\n")
            f.write(f"这个语义块被切成了 {len(same_heading)} 个子块：\n\n")
            for i, chunk in enumerate(same_heading):
                f.write(f"#### 子块 {i+1}（{len(chunk['content'])} 字）\n\n")
                f.write(f"{chunk['content']}\n\n---\n\n")

    print(f"\n结果已保存到：{output_file}")


if __name__ == "__main__":
    main()
