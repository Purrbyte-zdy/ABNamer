import logging
import os
import re
from pathlib import Path

import olefile
from docx import Document

# 配置日志记录器
logger = logging.getLogger(__name__)


class FileReader:
    def txt(self, file_path: str) -> str:
        """读取txt文件，自动检测编码"""
        logger.info(f"开始读取文本文件: {file_path}")
        encodings = ["utf-8", "gbk", "latin-1", "utf-16"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                    logger.info(f"成功读取文件，使用编码: {encoding}")
                    return content
            except UnicodeDecodeError as ude:
                logger.warning(f"编码 {encoding} 解码失败: {str(ude)}")
                continue

        raise ValueError(f"无法解码文件: {file_path}")

    def docx(self, file_path: str) -> str:
        """读取docx格式的Word文档"""
        logger.info(f"开始读取DOCX文件: {file_path}")
        try:
            doc = Document(file_path)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs]
            return "\n".join(paragraphs)
        except Exception as error:
            logger.exception(f"读取DOCX文件失败: {str(error)}")
            raise

    def doc(self, file_path: str) -> str:
        """读取doc格式的Word文档"""
        logger.info(f"开始读取DOC文件: {file_path}")
        try:
            with olefile.OleFileIO(file_path) as ole:
                if not ole.exists('WordDocument'):
                    raise ValueError("文件不是有效的Word文档")

                doc_data = ole.openstream('WordDocument').read()
                try:
                    text = doc_data.decode('utf-16', errors='replace')
                except UnicodeDecodeError:
                    text = doc_data.decode('latin-1', errors='replace')

                text = re.sub(r'\x00', '', text)  # 移除NUL字符
                text = re.sub(r'\s+', ' ', text)  # 合并连续空格
                return text.strip()
        except Exception as error:
            logger.exception(f"读取DOC文件失败: {str(error)}")
            raise


def detect_language(text: str) -> str:
    if not text:
        return 'en'

    # 初始化字符计数器
    char_counts = {
        'cjk': 0,  # 中日韩统一表意文字 (包含汉字)
        'hiragana': 0,  # 平假名 (日语)
        'katakana': 0,  # 片假名 (日语)
        'hangul': 0,  # 韩文字母
        'cyrillic': 0,  # 西里尔字母 (俄语等)
        'arabic': 0,  # 阿拉伯字母
    }

    # 分析字符分布
    for char in text:
        # 中日韩统一表意文字 (包含汉字)
        if '\u4e00' <= char <= '\u9fff':
            char_counts['cjk'] += 1

        # 平假名 (日语)
        elif '\u3040' <= char <= '\u309f':
            char_counts['hiragana'] += 1

        # 片假名 (日语)
        elif '\u30a0' <= char <= '\u30ff':
            char_counts['katakana'] += 1

        # 韩文字母
        elif '\uac00' <= char <= '\ud7af':
            char_counts['hangul'] += 1

        # 西里尔字母 (俄语等)
        elif '\u0400' <= char <= '\u04ff':
            char_counts['cyrillic'] += 1

        # 阿拉伯字母
        elif '\u0600' <= char <= '\u06ff':
            char_counts['arabic'] += 1

    # 确定主要语言
    total_chars = len(text)

    # 日语检测 (平假名或片假名占显著比例)
    if (char_counts['hiragana'] + char_counts['katakana']) / total_chars > 0.1:
        return 'ja'

    # 韩语检测 (韩文字母占显著比例)
    if char_counts['hangul'] / total_chars > 0.1:
        return 'ko'

    # 俄语检测 (西里尔字母占显著比例)
    if char_counts['cyrillic'] / total_chars > 0.1:
        return 'ru'

    # 阿拉伯语检测 (阿拉伯字母占显著比例)
    if char_counts['arabic'] / total_chars > 0.1:
        return 'ar'

    # 中文检测 (CJK字符占显著比例)
    if char_counts['cjk'] / total_chars > 0.1:
        return 'zh'

    # 默认英语
    return 'en'


def read_file(file_path: str) -> str:
    """通用文件读取入口"""
    logger.info(f"开始读取文件: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = Path(file_path).suffix.lower()
    logger.debug(f"文件后缀: {suffix}")

    fr = FileReader()
    # 使用Python 3.12的match语句替代多分支if-else
    match suffix:
        case '.txt':
            logger.info("识别为文本文件，调用文本读取方法")
            return fr.txt(file_path)
        case '.docx':
            logger.info("识别为Word文档(.docx)，调用DOCX读取方法")
            return fr.docx(file_path)
        case '.doc':
            logger.info("识别为Word文档(.doc)，调用DOC读取方法")
            return fr.doc(file_path)
        case _:
            supported_types = ['.txt', '.docx', '.doc']
            logger.error(f"不支持的文件类型: {suffix}，支持的类型: {supported_types}")
            raise ValueError(f"不支持的文件类型: {suffix}")
