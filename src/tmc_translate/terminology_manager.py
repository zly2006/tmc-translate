import pandas as pd
from typing import List, Optional
from .models import Term
import logging

logger = logging.getLogger(__name__)


class TerminologyManager:
    """术语库管理器，负责从Excel文件读取和管理术语数据"""

    def __init__(self, excel_path: Optional[str] = None):
        self.terms: List[Term] = []
        self.excel_path = excel_path
        if excel_path:
            self.load_from_excel(excel_path)

    def load_from_excel(self, excel_path: str, sheet_name: str = 'Sheet1') -> None:
        """
        从Excel文件加载术语库
        Expected columns: english_name, chinese_name, english_description, chinese_description
        """
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            # 验证必需的列是否存在
            required_columns = ['english_name', 'chinese_name', 'english_description', 'chinese_description']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Excel文件缺少必需的列: {missing_columns}")

            # 清空现有术语
            self.terms.clear()

            # 加载术语数据
            for _, row in df.iterrows():
                # 跳过空行
                if pd.isna(row['english_name']) or pd.isna(row['chinese_name']):
                    continue

                term = Term(
                    english_name=str(row['english_name']).strip(),
                    chinese_name=str(row['chinese_name']).strip(),
                    english_description=str(row['english_description']).strip() if pd.notna(row['english_description']) else "",
                    chinese_description=str(row['chinese_description']).strip() if pd.notna(row['chinese_description']) else ""
                )
                self.terms.append(term)

            logger.info(f"成功加载 {len(self.terms)} 个术语")

        except Exception as e:
            logger.error(f"加载Excel文件失败: {e}")
            raise

    def search_terms(self, text: str, language: str = "both") -> List[Term]:
        """
        在文本中搜索相关术语

        Args:
            text: 要搜索的文本
            language: 搜索语言 ("english", "chinese", "both")

        Returns:
            相关术语列表
        """
        relevant_terms = []
        text_lower = text.lower()

        for term in self.terms:
            if language in ["english", "both"]:
                if (term.english_name.lower() in text_lower or
                    any(word.lower() in text_lower for word in term.english_name.split())):
                    if term not in relevant_terms:
                        relevant_terms.append(term)

            if language in ["chinese", "both"]:
                if (term.chinese_name in text or
                    any(char in text for char in term.chinese_name if len(char) > 1)):
                    if term not in relevant_terms:
                        relevant_terms.append(term)

        return relevant_terms

    def get_all_terms(self) -> List[Term]:
        """获取所有术语"""
        return self.terms.copy()

    def add_term(self, term: Term) -> None:
        """添加新术语"""
        self.terms.append(term)

    def create_sample_excel(self, output_path: str) -> None:
        """创建示例Excel文件"""
        sample_data = [
            {
                'english_name': 'Machine Learning',
                'chinese_name': '机器学习',
                'english_description': 'A method of data analysis that automates analytical model building',
                'chinese_description': '一种自动化分析模型构建的数据分析方法'
            },
            {
                'english_name': 'Artificial Intelligence',
                'chinese_name': '人工智能',
                'english_description': 'Intelligence demonstrated by machines, in contrast to natural intelligence',
                'chinese_description': '机器所展示的智能，与自然智能相对'
            },
            {
                'english_name': 'Neural Network',
                'chinese_name': '神经网络',
                'english_description': 'Computing systems inspired by biological neural networks',
                'chinese_description': '受生物神经网络启发的计算系统'
            }
        ]

        df = pd.DataFrame(sample_data)
        df.to_excel(output_path, index=False)
        logger.info(f"示例术语库已创建: {output_path}")
