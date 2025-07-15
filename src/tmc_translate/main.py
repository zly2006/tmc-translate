import os
import logging
from typing import Optional, Union

import dotenv

from .terminology_manager import TerminologyManager
from .minecraft_language_manager import MinecraftLanguageManager
from .rag_translator import RAGTranslator, OllamaProvider, GeminiProvider
from .models import TranslationContext

# 加载环境变量
dotenv.load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranslationApp:
    """翻译应用主类"""

    def __init__(self):
        self.terminology_manager: Optional[Union[TerminologyManager, MinecraftLanguageManager]] = None
        self.translator: Optional[RAGTranslator] = None
        self.current_model_type: str = ""
        self.terminology_type: str = ""  # 术语库类型

    def setup_terminology(self) -> bool:
        """设置术语库"""
        print("\n=== 术语库设置 ===")

        # 选择术语库类型
        print("选择术语库类型:")
        print("1. 标准术语库 (包含名称和描述)")
        print("2. Minecraft语言文件 (JSON格式，仅包含名称对应)")

        type_choice = input("请选择术语库类型 (1-2): ").strip()

        if type_choice == "1":
            return self._setup_standard_terminology()
        elif type_choice == "2":
            return self._setup_minecraft_language()
        else:
            print("❌ 无效选择")
            return False

    def _setup_standard_terminology(self) -> bool:
        """设置标准术语库"""
        print("\n--- 标准术语库设置 ---")
        self.terminology_type = "标准术语库"

        # 检查是否有现有的术语库文件
        excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx') or f.endswith('.xls')]

        if excel_files:
            print(f"发现Excel文件: {', '.join(excel_files)}")
            choice = input("选择术语库文件 (输入文件名) 或按回车创建示例文件: ").strip()

            if choice and choice in excel_files:
                try:
                    self.terminology_manager = TerminologyManager(choice)
                    print(f"✅ 成功加载标准术语库: {len(self.terminology_manager.get_all_terms())} 个术语")
                    return True
                except Exception as e:
                    print(f"❌ 加载术语库失败: {e}")
                    return False

        # 创建示例术语库
        sample_file = "sample_terminology.xlsx"
        try:
            temp_manager = TerminologyManager()
            temp_manager.create_sample_excel(sample_file)
            print(f"✅ 已创建示例标准术语库: {sample_file}")

            use_sample = input("是否使用示例术语库？(y/n): ").lower().strip()
            if use_sample == 'y':
                self.terminology_manager = TerminologyManager(sample_file)
                print(f"✅ 使用示例标准术语库: {len(self.terminology_manager.get_all_terms())} 个术语")
                return True
            else:
                print("请准备Excel术语库文件后重新运行程序")
                return False

        except Exception as e:
            print(f"❌ 创建示例术语库失败: {e}")
            return False

    def _setup_minecraft_language(self) -> bool:
        """设置Minecraft语言文件"""
        print("\n--- Minecraft语言文件设置 ---")
        self.terminology_type = "Minecraft语言文件"

        # 检查是否有现有的JSON文件
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]

        if json_files:
            print(f"发现JSON文件: {', '.join(json_files)}")
            choice = input("选择Minecraft语言文件 (输入文件名) 或按回车创建示例文件: ").strip()

            if choice and choice in json_files:
                # 配置搜索参数
                threshold = self._get_similarity_threshold()
                max_results = self._get_max_results()

                try:
                    self.terminology_manager = MinecraftLanguageManager(
                        choice,
                        similarity_threshold=threshold
                    )
                    self.terminology_manager.set_max_results(max_results)

                    stats = self.terminology_manager.get_stats()
                    print(f"✅ 成功加载Minecraft语言文件:")
                    print(f"   - 术语数量: {stats['total_terms']} 个")
                    print(f"   - 相似度阈值: {stats['similarity_threshold']}")
                    print(f"   - 最大结果数: {stats['max_results']}")
                    return True
                except Exception as e:
                    print(f"❌ 加载Minecraft语言文件失败: {e}")
                    return False

        # 创建示例Minecraft语言文件
        sample_file = "minecraft_lang_zh_cn.json"
        try:
            temp_manager = MinecraftLanguageManager()
            temp_manager.create_sample_json(sample_file)
            print(f"✅ 已创建示例Minecraft语言文件: {sample_file}")

            use_sample = input("是否使用示例语言文件？(y/n): ").lower().strip()
            if use_sample == 'y':
                threshold = self._get_similarity_threshold()
                max_results = self._get_max_results()

                self.terminology_manager = MinecraftLanguageManager(
                    sample_file,
                    similarity_threshold=threshold
                )
                self.terminology_manager.set_max_results(max_results)

                stats = self.terminology_manager.get_stats()
                print(f"✅ 使用示例Minecraft语言文件:")
                print(f"   - 术语数量: {stats['total_terms']} 个")
                print(f"   - 相似度阈值: {stats['similarity_threshold']}")
                print(f"   - 最大结果数: {stats['max_results']}")
                return True
            else:
                print("请准备Minecraft语言JSON文件后重新运行程序")
                return False

        except Exception as e:
            print(f"❌ 创建示例Minecraft语言文件失败: {e}")
            return False

    def _get_similarity_threshold(self) -> float:
        """获取相似度阈值设置"""
        while True:
            try:
                threshold_input = input("设置相似度阈值 (0.1-1.0, 推荐0.8): ").strip()
                if not threshold_input:
                    return 0.8  # 默认值

                threshold = float(threshold_input)
                if 0.1 <= threshold <= 1.0:
                    return threshold
                else:
                    print("❌ 阈值必须在0.1到1.0之间")
            except ValueError:
                print("❌ 请输入有效的数字")

    def _get_max_results(self) -> int:
        """获取最大结果数设置"""
        while True:
            try:
                max_input = input("设置最大结果数 (1-50, 推荐10): ").strip()
                if not max_input:
                    return 10  # 默认值

                max_results = int(max_input)
                if 1 <= max_results <= 50:
                    return max_results
                else:
                    print("❌ 最大结果数必须在1到50之间")
            except ValueError:
                print("❌ 请输入有效的整数")

    def setup_model(self) -> bool:
        """设置模型"""
        print("\n=== 模型设置 ===")
        print("支持的模型:")
        print("1. Ollama (本地运行)")
        print("2. Google Gemini (需要API Key)")

        choice = input("选择模型类型 (1-2): ").strip()

        try:
            if choice == "1":
                return self._setup_ollama()
            elif choice == "2":
                return self._setup_gemini()
            else:
                print("❌ 无效选择")
                return False
        except Exception as e:
            print(f"❌ 模型设置失败: {e}")
            return False

    def _setup_ollama(self) -> bool:
        """设置Ollama模型"""
        print("\n--- Ollama设置 ---")

        model_name = input("输入模型名称 (默认: llama2): ").strip()
        if not model_name:
            model_name = "llama2"

        base_url = input("输入Ollama服务地址 (默认: http://localhost:11434): ").strip()
        if not base_url:
            base_url = "http://localhost:11434"

        try:
            provider = OllamaProvider(model_name=model_name, base_url=base_url)
            self.translator = RAGTranslator(self.terminology_manager, provider)
            self.current_model_type = f"Ollama ({model_name})"
            print(f"✅ Ollama模型设置成功: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Ollama模型设置失败: {e}")
            print("请确保Ollama服务正在运行，并且模型已下载")
            return False

    def _setup_gemini(self) -> bool:
        """设置Gemini模型"""
        print("\n--- Gemini设置 ---")

        # 尝试从环境变量获取API Key
        api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            api_key = input("输入Google API Key: ").strip()
            if not api_key:
                print("❌ API Key不能为空")
                return False

        model_name = input("输入模型名称 (默认: gemini-1.5-flash): ").strip()
        if not model_name:
            model_name = "gemini-1.5-flash"

        try:
            provider = GeminiProvider(api_key=api_key, model_name=model_name)
            self.translator = RAGTranslator(self.terminology_manager, provider)
            self.current_model_type = f"Gemini ({model_name})"
            print(f"✅ Gemini模型设置成功: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Gemini模型设置失败: {e}")
            return False

    def show_menu(self) -> None:
        """显示主菜单"""
        print(f"\n=== TMC翻译系统 ===")
        print(f"当前模型: {self.current_model_type}")
        print(f"术语库类型: {self.terminology_type}")

        if hasattr(self.terminology_manager, 'get_stats'):
            stats = self.terminology_manager.get_stats()
            print(f"术语库: {stats['total_terms']} 个术语 (阈值: {stats['similarity_threshold']}, 最大结果: {stats['max_results']})")
        else:
            print(f"术语库: {len(self.terminology_manager.get_all_terms())} 个术语")

        print("\n选项:")
        print("1. 翻译文本")
        print("2. 查看术语库")
        print("3. 重新加载术语库")
        print("4. 调整搜索参数 (仅Minecraft语言文件)")
        print("5. 切换模型")
        print("6. 退出")

    def translate_text(self) -> None:
        """翻译文本交互"""
        print("\n=== 文本翻译 ===")
        print("输入要翻译的文本 (输入 'back' 返回主菜单):")

        while True:
            text = input("\n> ").strip()

            if text.lower() == 'back':
                break

            if not text:
                print("请输入有效文本")
                continue

            try:
                print("🔄 翻译中...")
                context = self.translator.translate(text)

                print(f"\n📄 原文: {context.source_text}")
                print(f"🔄 译文: {context.translation_result}")

                if context.relevant_terms:
                    print(f"\n📚 相关术语 ({len(context.relevant_terms)} 个):")
                    for i, term in enumerate(context.relevant_terms[:5], 1):  # 只显示前5个
                        print(f"  {i}. {term.english_name} ↔ {term.chinese_name}")

            except Exception as e:
                print(f"❌ 翻译失败: {e}")

    def show_terminology(self) -> None:
        """显示术语库"""
        print("\n=== 术语库 ===")
        terms = self.terminology_manager.get_all_terms()

        if not terms:
            print("术语库为空")
            return

        print(f"共 {len(terms)} 个术语:")
        for i, term in enumerate(terms, 1):
            print(f"\n{i}. {term.english_name} | {term.chinese_name}")
            if term.english_description:
                print(f"   EN: {term.english_description}")
            if term.chinese_description:
                print(f"   CN: {term.chinese_description}")

            if i >= 10:  # 只显示前10个，避免输出过长
                remaining = len(terms) - 10
                if remaining > 0:
                    print(f"\n... 还有 {remaining} 个术语")
                break

    def reload_terminology(self) -> None:
        """重新加载术语库"""
        print("\n=== 重新加载术语库 ===")
        if self.setup_terminology():
            # 刷新翻译器的向量存储
            if self.translator:
                self.translator.terminology_manager = self.terminology_manager
                self.translator.refresh_vector_store()
                print("✅ 术语库和向量存储已更新")

    def adjust_search_parameters(self) -> None:
        """调整搜索参数（仅适用于Minecraft语言文件）"""
        if not isinstance(self.terminology_manager, MinecraftLanguageManager):
            print("❌ 此功能仅适用于Minecraft语言文件")
            return

        print("\n=== 调整搜索参数 ===")
        current_stats = self.terminology_manager.get_stats()
        print(f"当前设置:")
        print(f"  - 相似度阈值: {current_stats['similarity_threshold']}")
        print(f"  - 最大结果数: {current_stats['max_results']}")

        # 调整相似度阈值
        adjust_threshold = input("\n是否调整相似度阈值？(y/n): ").lower().strip()
        if adjust_threshold == 'y':
            new_threshold = self._get_similarity_threshold()
            self.terminology_manager.set_similarity_threshold(new_threshold)

        # 调整最大结果数
        adjust_max = input("是否调整最大结果数？(y/n): ").lower().strip()
        if adjust_max == 'y':
            new_max = self._get_max_results()
            self.terminology_manager.set_max_results(new_max)

        updated_stats = self.terminology_manager.get_stats()
        print(f"\n✅ 参数已更新:")
        print(f"  - 相似度阈值: {updated_stats['similarity_threshold']}")
        print(f"  - 最大结果数: {updated_stats['max_results']}")

    def run(self) -> None:
        """运行应用"""
        print("🌟 欢迎使用TMC翻译系统!")
        print("这是一个基于RAG的中英文术语翻译工具")
        print("支持标准术语库(含描述)和Minecraft语言文件(高阈值搜索)")

        # 设置术语库
        if not self.setup_terminology():
            print("❌ 术语库设置失败，程序退出")
            return

        # 设置模型
        if not self.setup_model():
            print("❌ 模型设置失败，程序退出")
            return

        # 主循环
        while True:
            try:
                self.show_menu()
                choice = input("\n请选择 (1-6): ").strip()

                if choice == "1":
                    self.translate_text()
                elif choice == "2":
                    self.show_terminology()
                elif choice == "3":
                    self.reload_terminology()
                elif choice == "4":
                    self.adjust_search_parameters()
                elif choice == "5":
                    if self.setup_model():
                        print("✅ 模型切换成功")
                elif choice == "6":
                    print("👋 再见!")
                    break
                else:
                    print("❌ 无效选择，请重试")

            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见!")
                break
            except Exception as e:
                logger.error(f"程序运行错误: {e}")
                print(f"❌ 发生错误: {e}")


def main() -> None:
    """主函数入口点"""
    app = TranslationApp()
    app.run()


if __name__ == "__main__":
    main()
