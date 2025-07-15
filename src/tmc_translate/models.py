from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Term:
    """术语数据类"""
    english_name: str
    chinese_name: str
    english_description: str
    chinese_description: str

    def __str__(self) -> str:
        return f"Term(EN: {self.english_name}, CN: {self.chinese_name})"

    def get_full_context(self, language: str = "both") -> str:
        """获取术语的完整上下文信息"""
        if language == "english":
            return f"{self.english_name}: {self.english_description}"
        elif language == "chinese":
            return f"{self.chinese_name}: {self.chinese_description}"
        else:
            return f"English: {self.english_name} - {self.english_description}\nChinese: {self.chinese_name} - {self.chinese_description}"


@dataclass
class SimpleTerm:
    """简化术语数据类 - 只包含名称对应，无描述"""
    english_name: str
    chinese_name: str

    def __str__(self) -> str:
        return f"SimpleTerm(EN: {self.english_name}, CN: {self.chinese_name})"

    def to_term(self) -> Term:
        """转换为完整的Term对象"""
        return Term(
            english_name=self.english_name,
            chinese_name=self.chinese_name,
            english_description="",
            chinese_description=""
        )


@dataclass
class TranslationContext:
    """翻译上下文数据类"""
    source_text: str
    target_language: str
    relevant_terms: List[Term]
    translation_result: Optional[str] = None

    def get_terms_context(self) -> str:
        """获取相关术语的上下文字符串"""
        if not self.relevant_terms:
            return ""

        context_parts = []
        for term in self.relevant_terms:
            if self.target_language == "chinese":
                if term.english_description:
                    context_parts.append(f"- {term.english_name} → {term.chinese_name}: {term.chinese_description}")
                else:
                    context_parts.append(f"- {term.english_name} → {term.chinese_name}")
            else:
                if term.chinese_description:
                    context_parts.append(f"- {term.chinese_name} → {term.english_name}: {term.english_description}")
                else:
                    context_parts.append(f"- {term.chinese_name} → {term.english_name}")

        return "\n".join(context_parts)
