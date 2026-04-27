"""小说导入 — 章节切分"""

import re
from dataclasses import dataclass


@dataclass
class Chapter:
    index: int        # 从1开始
    title: str        # 章节标题（如果有）
    content: str      # 正文
    char_count: int   # 字数


# 常见中文章节标题模式
# 注意：结尾用 ([\s　]+.+)? 而非 .+ ，允许"第一回"单独一行（无标题），
# 同时避免把"第一回那天他遇到了xxx"这类正文句子误识别为章节标题。
CHAPTER_PATTERNS = [
    # 第二部：黑暗森林 第40章 面壁者三 / 第一部  第1章 xxx
    r"^第[一二三四五六七八九十百千\d]+部[^第]*第[一二三四五六七八九十百千\d]+章([\s　]+.+)?$",
    # 第一回 风雪惊变 / 第三章 xxx / 第一卷 xxx（标题可选）
    r"^第[一二三四五六七八九十百千]+[章节回卷]([\s　]+.+)?$",
    # 第40章 xxx / 第3回（数字版，标题可选）
    r"^第\d+[章节回卷]([\s　]+.+)?$",
    r"^Chapter\s*\d+([\s　]+.+)?$",
]


def split_chapters(
    text: str,
    max_chunk_chars: int = 6000,
) -> list[Chapter]:
    """
    将小说文本切分为章节。

    策略：
    1. 先尝试用正则匹配章节标题
    2. 如果找不到章节标记，按固定长度切分（在段落边界切）
    """
    lines = text.split("\n")

    # === 策略1：正则匹配章节标题 ===
    chapter_breaks: list[tuple[int, str]] = []  # (行号, 标题)

    compiled_patterns = [re.compile(p) for p in CHAPTER_PATTERNS]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pat in compiled_patterns:
            if pat.match(stripped):
                chapter_breaks.append((i, stripped))
                break

    if len(chapter_breaks) >= 3:
        # 找到了足够的章节标记，按标记切分
        return _split_by_markers(lines, chapter_breaks)

    # === 策略2：找不到章节标记，按段落边界固定长度切分 ===
    return _split_by_length(text, max_chunk_chars)


def _split_by_markers(
    lines: list[str], breaks: list[tuple[int, str]]
) -> list[Chapter]:
    # 去重：同一标题出现多次时（目录 + 正文），只保留最后一次（正文位置）。
    # 用 normalize 后的标题作 key，去除空格差异。
    seen: dict[str, int] = {}  # normalized_title -> index in breaks
    for i, (_, title) in enumerate(breaks):
        key = re.sub(r"\s+", "", title)
        seen[key] = i  # 覆盖写入，最终保留最后一次出现的索引

    deduped_indices = sorted(seen.values())
    deduped_breaks = [breaks[i] for i in deduped_indices]

    chapters = []
    chapter_no = 0   # 只在实际写入时递增，避免空内容章节造成编号跳空
    for idx, (start, title) in enumerate(deduped_breaks):
        end = deduped_breaks[idx + 1][0] if idx + 1 < len(deduped_breaks) else len(lines)
        content = "\n".join(lines[start + 1 : end]).strip()
        if content:
            chapter_no += 1
            chapters.append(Chapter(
                index=chapter_no,
                title=title,
                content=content,
                char_count=len(content),
            ))
    return chapters


def _split_by_length(text: str, max_chars: int) -> list[Chapter]:
    paragraphs = re.split(r"\n\s*\n", text)
    chapters = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > max_chars and current_chunk:
            chapters.append(Chapter(
                index=len(chapters) + 1,
                title=f"段落 {len(chapters) + 1}",
                content="\n\n".join(current_chunk),
                char_count=current_len,
            ))
            current_chunk = []
            current_len = 0
        current_chunk.append(para)
        current_len += len(para)

    if current_chunk:
        chapters.append(Chapter(
            index=len(chapters) + 1,
            title=f"段落 {len(chapters) + 1}",
            content="\n\n".join(current_chunk),
            char_count=current_len,
        ))

    return chapters
