from typing import List, Optional, Union
from .models import Term
from .terminology_manager import TerminologyManager
from .minecraft_language_manager import MinecraftLanguageManager
import logging

logger = logging.getLogger(__name__)


class HybridTerminologyManager:
    """混合术语管理器，同时支持标准术语库和Minecraft语言文件"""

    def __init__(self):
        self.standard_manager: Optional[TerminologyManager] = None
        self.minecraft_manager: Optional[MinecraftLanguageManager] = None
        self.active_managers: List[Union[TerminologyManager, MinecraftLanguageManager]] = []

    def add_standard_terminology(self, excel_path: str) -> bool:
        """添加标准术语库"""
        try:
            self.standard_manager = TerminologyManager(excel_path)
            if self.standard_manager not in self.active_managers:
                self.active_managers.append(self.standard_manager)
            logger.info(f"标准术语库已添加: {len(self.standard_manager.get_all_terms())} 个术语")
            return True
        except Exception as e:
            logger.error(f"添加标准术语库失败: {e}")
            return False

    def add_minecraft_language(self, json_path: str, similarity_threshold: float = 0.8, max_results: int = 10) -> bool:
        """添加Minecraft语言文件"""
        try:
            self.minecraft_manager = MinecraftLanguageManager(json_path, similarity_threshold)
            self.minecraft_manager.set_max_results(max_results)
            if self.minecraft_manager not in self.active_managers:
                self.active_managers.append(self.minecraft_manager)
            logger.info(f"Minecraft语言文件已添加: {len(self.minecraft_manager.get_all_terms())} 个术语")
            return True
        except Exception as e:
            logger.error(f"添加Minecraft语言文件失败: {e}")
            return False

    def search_terms(self, text: str, language: str = "both") -> List[Term]:
        """
        在所有活跃的术语管理器中搜索相关术语

        Args:
            text: 要搜索的文本
            language: 搜索语言 ("english", "chinese", "both")

        Returns:
            相关术语列表（合并所有管理器的结果）
        """
        all_terms = []
        seen_terms = set()  # 用于去重

        for manager in self.active_managers:
            try:
                # 对每个管理器单独处理搜索结果
                terms = manager.search_terms(text, language)

                for term in terms:
                    # 使用英文名和中文名的组合作为唯一标识符
                    term_key = (term.english_name.lower(), term.chinese_name)
                    if term_key not in seen_terms:
                        seen_terms.add(term_key)
                        all_terms.append(term)
            except Exception as e:
                logger.warning(f"搜索术语时出错 ({type(manager).__name__}): {e}")

        # 不再对总结果数进行额外限制，让各管理器自行控制
        logger.info(f"混合搜索找到 {len(all_terms)} 个相关术语")
        return all_terms

    def get_all_terms(self) -> List[Term]:
        """获取所有术语（合并所有管理器的结果）"""
        all_terms = []
        seen_terms = set()

        for manager in self.active_managers:
            try:
                terms = manager.get_all_terms()
                for term in terms:
                    term_key = (term.english_name.lower(), term.chinese_name)
                    if term_key not in seen_terms:
                        seen_terms.add(term_key)
                        all_terms.append(term)
            except Exception as e:
                logger.warning(f"获取术语时出错 ({type(manager).__name__}): {e}")

        return all_terms

    def get_stats(self) -> dict:
        """获取混合术语库的统计信息"""
        stats = {
            "total_managers": len(self.active_managers),
            "total_terms": len(self.get_all_terms()),
            "standard_manager": None,
            "minecraft_manager": None
        }

        if self.standard_manager and self.standard_manager in self.active_managers:
            stats["standard_manager"] = {
                "terms_count": len(self.standard_manager.get_all_terms()),
                "type": "标准术语库"
            }

        if self.minecraft_manager and self.minecraft_manager in self.active_managers:
            minecraft_stats = self.minecraft_manager.get_stats()
            stats["minecraft_manager"] = {
                "terms_count": minecraft_stats["total_terms"],
                "similarity_threshold": minecraft_stats["similarity_threshold"],
                "max_results": minecraft_stats["max_results"],
                "type": "Minecraft语言文件"
            }

        return stats

    def remove_standard_terminology(self) -> bool:
        """移除标准术语库"""
        if self.standard_manager and self.standard_manager in self.active_managers:
            self.active_managers.remove(self.standard_manager)
            self.standard_manager = None
            logger.info("标准术语库已移除")
            return True
        return False

    def remove_minecraft_language(self) -> bool:
        """移除Minecraft语言文件"""
        if self.minecraft_manager and self.minecraft_manager in self.active_managers:
            self.active_managers.remove(self.minecraft_manager)
            self.minecraft_manager = None
            logger.info("Minecraft语言文件已移除")
            return True
        return False

    def set_minecraft_similarity_threshold(self, threshold: float) -> bool:
        """设置Minecraft管理器的相似度阈值"""
        if self.minecraft_manager:
            self.minecraft_manager.set_similarity_threshold(threshold)
            return True
        return False

    def set_minecraft_max_results(self, max_results: int) -> bool:
        """设置Minecraft管理器的最大结果数"""
        if self.minecraft_manager:
            self.minecraft_manager.set_max_results(max_results)
            return True
        return False

    def has_any_manager(self) -> bool:
        """检查是否有任何活跃的管理器"""
        return len(self.active_managers) > 0

    def has_standard_manager(self) -> bool:
        """检查是否有标准术语管理器"""
        return self.standard_manager is not None and self.standard_manager in self.active_managers

    def has_minecraft_manager(self) -> bool:
        """检查是否有Minecraft语言管理器"""
        return self.minecraft_manager is not None and self.minecraft_manager in self.active_managers
