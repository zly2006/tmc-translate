import json
import logging
import re
from difflib import SequenceMatcher
from typing import List, Optional, Set, Tuple

from .models import Term, SimpleTerm

logger = logging.getLogger(__name__)


class MinecraftLanguageManager:
    """Minecraft语言管理器，处理大规模术语数据，支持高阈值搜索和去重"""

    def __init__(self, json_path: Optional[str] = "assets/zh_cn.json", similarity_threshold: float = 0.8):
        self.terms: List[SimpleTerm] = []
        self.json_path = json_path
        self.similarity_threshold = similarity_threshold  # 相似度阈值
        self.max_results = 10  # 最大返回结果数

        # 用于快速查找的索引
        self.english_index: dict = {}
        self.chinese_index: dict = {}

        if json_path:
            self.load_from_json(json_path)

    def load_from_json(self, json_path: str) -> None:
        """
        从JSON文件加载Minecraft语言文件
        Expected format: { "english": "chinese", ... }
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if not isinstance(data, dict):
                raise ValueError("JSON文件必须是一个字典格式")

            # 清空现有术语
            self.terms.clear()
            self.english_index.clear()
            self.chinese_index.clear()

            # 用于去重的集合
            seen_pairs: Set[Tuple[str, str]] = set()

            logger.info("开始加载Minecraft语言文件，正在去重...")

            # 加载术语数据并去重
            for english_key, chinese_value in data.items():
                if not english_key or not chinese_value:
                    continue

                english_name = str(english_key).strip()
                chinese_name = str(chinese_value).strip()

                # 去重检查
                term_pair = (english_name.lower(), chinese_name)
                if term_pair in seen_pairs:
                    continue

                seen_pairs.add(term_pair)

                term = SimpleTerm(
                    english_name=english_name,
                    chinese_name=chinese_name
                )
                self.terms.append(term)

            # 构建索引以提高搜索性能
            self._build_indexes()

            logger.info(f"成功加载Minecraft语言文件: {len(self.terms)} 个术语（已去重）")

        except Exception as e:
            logger.error(f"加载JSON文件失败: {e}")
            raise

    def _build_indexes(self) -> None:
        """构建搜索索引以提高性能"""
        for i, term in enumerate(self.terms):
            # 英文索引 - 按单词分割
            english_words = re.findall(r'\b\w+\b', term.english_name.lower())
            for word in english_words:
                if word not in self.english_index:
                    self.english_index[word] = []
                self.english_index[word].append(i)

            # 中文索引 - 按字符和词语
            chinese_chars = list(term.chinese_name)
            for char in chinese_chars:
                if char not in self.chinese_index:
                    self.chinese_index[char] = []
                self.chinese_index[char].append(i)

            # 中文词语索引（2-4字词语）
            for length in range(2, min(5, len(term.chinese_name) + 1)):
                for start in range(len(term.chinese_name) - length + 1):
                    word = term.chinese_name[start:start + length]
                    if word not in self.chinese_index:
                        self.chinese_index[word] = []
                    self.chinese_index[word].append(i)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个字符串的相似度"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _exact_match_search(self, text: str, language: str) -> List[int]:
        """精确匹配搜索，返回术语索引列表"""
        candidates = set()
        text_lower = text.lower()

        if language in ["english", "both"]:
            # 英文精确匹配
            words = re.findall(r'\b\w+\b', text_lower)
            for word in words:
                if word in self.english_index:
                    candidates.update(self.english_index[word])

        if language in ["chinese", "both"]:
            # 中文精确匹配
            for char in text:
                if char in self.chinese_index:
                    candidates.update(self.chinese_index[char])

            # 中文词语匹配
            for length in range(2, min(5, len(text) + 1)):
                for start in range(len(text) - length + 1):
                    word = text[start:start + length]
                    if word in self.chinese_index:
                        candidates.update(self.chinese_index[word])

        return list(candidates)

    def _fuzzy_match_search(self, text: str, language: str, candidate_indices: List[int]) -> List[Tuple[int, float]]:
        """模糊匹配搜索，返回(索引, 相似度)列表"""
        matches = []
        text_lower = text.lower()

        for idx in candidate_indices:
            term = self.terms[idx]
            max_similarity = 0.0

            if language in ["english", "both"]:
                # 与英文名称比较
                similarity = self._calculate_similarity(text_lower, term.english_name.lower())
                max_similarity = max(max_similarity, similarity)

                # 与英文名称中的单词比较
                english_words = re.findall(r'\b\w+\b', term.english_name.lower())
                for word in english_words:
                    similarity = self._calculate_similarity(text_lower, word)
                    max_similarity = max(max_similarity, similarity)

            if language in ["chinese", "both"]:
                # 与中文名称比较
                similarity = self._calculate_similarity(text, term.chinese_name)
                max_similarity = max(max_similarity, similarity)

                # 子字符串匹配加分
                if text in term.chinese_name or term.chinese_name in text:
                    max_similarity = max(max_similarity, 0.9)

            if max_similarity >= self.similarity_threshold:
                matches.append((idx, max_similarity))

        return matches

    def search_terms(self, text: str, language: str = "both", max_results: Optional[int] = None) -> List[Term]:
        """
        高阈值搜索相关术语

        Args:
            text: 要搜索的文本
            language: 搜索语言 ("english", "chinese", "both")
            max_results: 最大返回结果数，默认使用self.max_results

        Returns:
            相关术语列表（转换为Term对象）
        """
        if max_results is None:
            max_results = self.max_results

        # 1. 精确匹配搜索
        candidate_indices = self._exact_match_search(text, language)

        if not candidate_indices:
            logger.debug(f"未找到精确匹配的术语候选: {text}")
            return []

        # 2. 模糊匹配和相似度计算
        matches = self._fuzzy_match_search(text, language, candidate_indices)

        if not matches:
            logger.debug(f"未找到相似度超过阈值({self.similarity_threshold})的术语: {text}")
            return []

        # 3. 按相似度排序并限制结果数量
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:max_results]

        # 4. 转换为Term对象
        result_terms = []
        for idx, similarity in top_matches:
            simple_term = self.terms[idx]
            term = simple_term.to_term()
            term.chinese_description = "专有名词参考，不一定符合语境，请只在确定时使用，并调整书写格式"
            result_terms.append(term)
            logger.debug(f"找到术语: {simple_term} (相似度: {similarity:.3f})")

        logger.info(f"搜索 '{text}' 找到 {len(result_terms)} 个高相关度术语")
        return result_terms

    def get_all_terms(self) -> List[Term]:
        """获取所有术语（转换为Term对象）"""
        return [term.to_term() for term in self.terms]

    def get_stats(self) -> dict:
        """获取术语库统计信息"""
        return {
            "total_terms": len(self.terms),
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "english_index_size": len(self.english_index),
            "chinese_index_size": len(self.chinese_index)
        }

    def set_similarity_threshold(self, threshold: float) -> None:
        """设置相似度阈值"""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            logger.info(f"相似度阈值已设置为: {threshold}")
        else:
            raise ValueError("相似度阈值必须在0.0到1.0之间")

    def set_max_results(self, max_results: int) -> None:
        """设置最大返回结果数"""
        if max_results > 0:
            self.max_results = max_results
            logger.info(f"最大结果数已设置为: {max_results}")
        else:
            raise ValueError("最大结果数必须大于0")
