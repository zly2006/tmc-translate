from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import logging
from langchain_core.language_models import BaseLanguageModel
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from .models import Term, TranslationContext
from .terminology_manager import TerminologyManager

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """模型提供者抽象基类"""

    @abstractmethod
    def get_llm(self) -> BaseLanguageModel:
        """获取语言模型"""
        pass

    @abstractmethod
    def get_embeddings(self):
        """获取嵌入模型"""
        pass


class OllamaProvider(ModelProvider):
    """Ollama模型提供者"""

    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def get_llm(self) -> BaseLanguageModel:
        return Ollama(model=self.model_name, base_url=self.base_url)

    def get_embeddings(self):
        return OllamaEmbeddings(model=self.model_name, base_url=self.base_url)


class GeminiProvider(ModelProvider):
    """Gemini模型提供者"""

    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.api_key = api_key
        self.model_name = model_name

    def get_llm(self) -> BaseLanguageModel:
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1
        )

    def get_embeddings(self):
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )


class RAGTranslator:
    """RAG翻译器主类"""

    def __init__(self, terminology_manager, model_provider: ModelProvider):
        self.terminology_manager = terminology_manager  # 现在可以是任意类型的术语管理器
        self.model_provider = model_provider
        self.llm = model_provider.get_llm()
        self.embeddings = model_provider.get_embeddings()
        self.vector_store: Optional[Chroma] = None

        # 初始化向量存储
        print('正在初始化向量存储...')
        self._setup_vector_store()

        # 翻译提示模板
        self.translation_prompt = PromptTemplate(
            template="""你是一个专业的中英文翻译助手，特别擅长处理技术术语。

任务：将以下{source_language}文本翻译成{target_language}，请特别注意专业术语的准确翻译。

相关术语参考：
{terminology_context}

原文：
{source_text}

翻译要求：
1. 保持原文的语义和语调
2. 确保专业术语翻译的准确性和一致性
3. 译文应该自然流畅
4. 如果遇到术语库中的术语，请严格按照术语库中的对应翻译

翻译结果：""",
            input_variables=["source_language", "target_language", "terminology_context", "source_text"]
        )

        # 创建翻译链
        self.translation_chain = self.translation_prompt | self.llm | StrOutputParser()

    def _setup_vector_store(self) -> None:
        """设置向量存储"""
        try:
            # 将术语转换为文档
            documents = []
            for term in self.terminology_manager.get_all_terms():
                # 创建英文文档
                en_doc = Document(
                    page_content=f"{term.english_name}: {term.english_description}",
                    metadata={
                        "type": "english_term",
                        "term_id": f"en_{term.english_name.replace(' ', '_')}",
                        "english_name": term.english_name,
                        "chinese_name": term.chinese_name
                    }
                )
                documents.append(en_doc)

                # 创建中文文档
                cn_doc = Document(
                    page_content=f"{term.chinese_name}: {term.chinese_description}",
                    metadata={
                        "type": "chinese_term",
                        "term_id": f"cn_{term.chinese_name}",
                        "english_name": term.english_name,
                        "chinese_name": term.chinese_name
                    }
                )
                documents.append(cn_doc)

            if documents:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory="./chroma_db"
                )
                logger.info(f"向量存储初始化完成，包含 {len(documents)} 个文档")
            else:
                logger.warning("术语库为空，无法初始化向量存储")

        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            self.vector_store = None

    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测：如果包含中文字符，认为是中文
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return "chinese"
        return "english"

    def _find_relevant_terms(self, text: str, language: str) -> List[Term]:
        """查找相关术语"""
        # 首先使用术语管理器的搜索功能
        relevant_terms = self.terminology_manager.search_terms(text, language)

        # 如果有向量存储，使用语义搜索增强结果
        if self.vector_store:
            try:
                # 进行语义搜索
                search_results = self.vector_store.similarity_search(
                    text,
                    k=5,
                    filter={"type": f"{language}_term"} if language != "both" else None
                )

                # 从搜索结果中提取术语
                for doc in search_results:
                    english_name = doc.metadata.get("english_name")
                    chinese_name = doc.metadata.get("chinese_name")

                    # 查找对应的完整术语对象
                    for term in self.terminology_manager.get_all_terms():
                        if (term.english_name == english_name and
                            term.chinese_name == chinese_name and
                            term not in relevant_terms):
                            relevant_terms.append(term)
                            break

            except Exception as e:
                logger.warning(f"向量搜索失败，使用关键词搜索: {e}")

        return relevant_terms

    def translate(self, text: str, target_language: Optional[str] = None) -> TranslationContext:
        """
        翻译文本

        Args:
            text: 要翻译的文本
            target_language: 目标语言 ("chinese" 或 "english")，如果为None则自动检测

        Returns:
            翻译上下文对象
        """
        # 检测源语言
        source_language = self._detect_language(text)

        # 确定目标语言
        if target_language is None:
            target_language = "english" if source_language == "chinese" else "chinese"

        # 查找相关术语
        relevant_terms = self._find_relevant_terms(text, source_language)

        # 创建翻译上下文
        context = TranslationContext(
            source_text=text,
            target_language=target_language,
            relevant_terms=relevant_terms
        )

        # 准备术语上下文
        terminology_context = context.get_terms_context()
        if not terminology_context:
            terminology_context = "无特定术语参考"

        # 执行翻译
        try:
            source_lang_name = "中文" if source_language == "chinese" else "英文"
            target_lang_name = "英文" if target_language == "english" else "中文"

            translation_result = self.translation_chain.invoke({
                "source_language": source_lang_name,
                "target_language": target_lang_name,
                "terminology_context": terminology_context,
                "source_text": text
            })

            context.translation_result = translation_result.strip()
            logger.info(f"翻译完成: {source_language} -> {target_language}")

        except Exception as e:
            logger.error(f"翻译失败: {e}")
            context.translation_result = f"翻译失败: {str(e)}"

        return context

    def refresh_vector_store(self) -> None:
        """刷新向量存储（当术语库更新时调用）"""
        self._setup_vector_store()
