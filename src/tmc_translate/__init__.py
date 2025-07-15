"""TMC Translate - A RAG-based Chinese-English terminology translation system using LangChain"""

from .models import Term, SimpleTerm, TranslationContext
from .terminology_manager import TerminologyManager
from .minecraft_language_manager import MinecraftLanguageManager
from .hybrid_terminology_manager import HybridTerminologyManager
from .rag_translator import RAGTranslator, OllamaProvider, GeminiProvider
from .main import main

__version__ = "0.1.0"
__all__ = [
    "Term",
    "SimpleTerm",
    "TranslationContext",
    "TerminologyManager",
    "MinecraftLanguageManager",
    "HybridTerminologyManager",
    "RAGTranslator",
    "OllamaProvider",
    "GeminiProvider",
    "main"
]
