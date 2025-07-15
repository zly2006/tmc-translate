import logging
import os
import sys
from typing import Optional

import dotenv

from .hybrid_terminology_manager import HybridTerminologyManager
from .rag_translator import RAGTranslator, OllamaProvider, GeminiProvider

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
        self.terminology_manager: Optional[HybridTerminologyManager] = None
        self.translator: Optional[RAGTranslator] = None
        self.current_model_type: str = ""

    def setup_terminology(self) -> bool:
        """设置术语库"""
        print("\n=== 术语库设置 ===")
        print("现在支持同时使用两种术语库类型")

        self.terminology_manager = HybridTerminologyManager()

        self._setup_standard_terminology()
        self._setup_minecraft_language()

        # 检查是否至少有一种术语库
        if not self.terminology_manager.has_any_manager():
            print("❌ 至少需要设置一种术语库")
            return False

        # 显示设置结果
        stats = self.terminology_manager.get_stats()
        print(f"\n✅ 术语库设置完成:")
        print(f"   - 总术语数: {stats['total_terms']} 个")
        print(f"   - 活跃管理器: {stats['total_managers']} 个")

        if stats['standard_manager']:
            print(f"   - 标准术语库: {stats['standard_manager']['terms_count']} 个术语")

        if stats['minecraft_manager']:
            mc_stats = stats['minecraft_manager']
            print(f"   - Minecraft语言文件: {mc_stats['terms_count']} 个术语")
            print(f"     (阈值: {mc_stats['similarity_threshold']}, 最大结果: {mc_stats['max_results']})")

        return True

    def _setup_standard_terminology(self) -> bool:
        """设置标准术语库"""
        print("\n--- 标准术语库设置 ---")

        # 检查是否有现有的术语库文件
        excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx') or f.endswith('.xls')]

        if excel_files:
            print(f"发现Excel文件: {', '.join(excel_files)}")
            choice = input("选择术语库文件 (输入文件名) 或按回车创建示例文件: ").strip()

            if choice and choice in excel_files:
                return self.terminology_manager.add_standard_terminology(choice)

        # 创建示例术语库
        sample_file = "sample_terminology.xlsx"
        try:
            from .terminology_manager import TerminologyManager
            temp_manager = TerminologyManager()
            temp_manager.create_sample_excel(sample_file)
            print(f"✅ 已创建示例标准术语库: {sample_file}")

            use_sample = input("是否使用示例术语库？(y/n): ").lower().strip()
            if use_sample == 'y':
                return self.terminology_manager.add_standard_terminology(sample_file)
            else:
                print("跳过标准术语库设置")
                return False

        except Exception as e:
            print(f"❌ 创建示例术语库失败: {e}")
            return False

    def _setup_minecraft_language(self) -> bool:
        """设置Minecraft语言文件"""
        print("\n--- Minecraft语言文件设置 ---")

        threshold = self._get_similarity_threshold()
        max_results = self._get_max_results()
        return self.terminology_manager.add_minecraft_language("assets/zh_cn_lite.json", threshold, max_results)

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
        # check argv
        if len(sys.argv) > 1:
            if '--ollama' in sys.argv:
                return self._setup_ollama()
            elif '--gemini' in sys.argv:
                return self._setup_gemini()

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

        if self.terminology_manager:
            stats = self.terminology_manager.get_stats()
            print(f"术语库: {stats['total_terms']} 个术语 ({stats['total_managers']} 个管理器)")

            if stats['standard_manager']:
                print(f"  - 标准术语库: {stats['standard_manager']['terms_count']} 个")

            if stats['minecraft_manager']:
                mc_stats = stats['minecraft_manager']
                print(f"  - Minecraft语言: {mc_stats['terms_count']} 个 (阈值: {mc_stats['similarity_threshold']})")

        # 显示向量存储状态
        if self.translator and self.translator.vector_store:
            try:
                existing_ids = self.translator._get_existing_term_ids()
                print(f"向量存储: {len(existing_ids)} 个文档已索引")
            except:
                print("向量存储: 已初始化")
        else:
            print("向量存储: 未初始化")

        print("\n选项:")
        print("1. 翻译文本")
        print("2. 查看术语库")
        print("3. 管理术语库")
        print("4. 调整Minecraft搜索参数")
        print("5. 管理向量存储")
        print("6. 切换模型")
        print("7. 退出")

    def manage_terminology(self) -> None:
        """管理术语库"""
        print("\n=== 术语库管理 ===")

        if not self.terminology_manager:
            print("❌ 未初始化术语管理器")
            return

        stats = self.terminology_manager.get_stats()
        print(f"当前状态:")
        print(f"  - 总术语数: {stats['total_terms']} 个")
        print(f"  - 标准术语库: {'已加载' if stats['standard_manager'] else '未加载'}")
        print(f"  - Minecraft语言: {'已加载' if stats['minecraft_manager'] else '未加载'}")

        print("\n管理选项:")
        print("1. 添加/重新加载标准术语库")
        print("2. 添加/重新加载Minecraft语言文件")
        print("3. 移除标准术语库")
        print("4. 移除Minecraft语言文件")
        print("5. 返回主菜单")

        choice = input("请选择 (1-5): ").strip()

        if choice == "1":
            if self.terminology_manager.has_standard_manager():
                self.terminology_manager.remove_standard_terminology()
            self._setup_standard_terminology()
        elif choice == "2":
            if self.terminology_manager.has_minecraft_manager():
                self.terminology_manager.remove_minecraft_language()
            self._setup_minecraft_language()
        elif choice == "3":
            if self.terminology_manager.remove_standard_terminology():
                print("✅ 标准术语库已移除")
            else:
                print("❌ 没有标准术语库可移除")
        elif choice == "4":
            if self.terminology_manager.remove_minecraft_language():
                print("✅ Minecraft语言文件已移除")
            else:
                print("❌ 没有Minecraft语言文件可移除")
        elif choice == "5":
            return

        # 更新翻译器
        if self.translator and self.terminology_manager.has_any_manager():
            self.translator.terminology_manager = self.terminology_manager
            self.translator.refresh_vector_store()
            print("✅ 翻译器已更新")

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

    def adjust_search_parameters(self) -> None:
        """调整Minecraft搜索参数"""
        if not self.terminology_manager or not self.terminology_manager.has_minecraft_manager():
            print("❌ 此功能需要加载Minecraft语言文件")
            return

        print("\n=== 调整Minecraft搜索参数 ===")
        stats = self.terminology_manager.get_stats()
        mc_stats = stats['minecraft_manager']

        print(f"当前设置:")
        print(f"  - 相似度阈值: {mc_stats['similarity_threshold']}")
        print(f"  - 最大结果数: {mc_stats['max_results']}")

        # 调整相似度阈值
        adjust_threshold = input("\n是否调整相似度阈值？(y/n): ").lower().strip()
        if adjust_threshold == 'y':
            new_threshold = self._get_similarity_threshold()
            self.terminology_manager.set_minecraft_similarity_threshold(new_threshold)

        # 调整最大结果数
        adjust_max = input("是否调整最大结果数？(y/n): ").lower().strip()
        if adjust_max == 'y':
            new_max = self._get_max_results()
            self.terminology_manager.set_minecraft_max_results(new_max)

        updated_stats = self.terminology_manager.get_stats()['minecraft_manager']
        print(f"\n✅ 参数已更新:")
        print(f"  - 相似度阈值: {updated_stats['similarity_threshold']}")
        print(f"  - 最大结果数: {updated_stats['max_results']}")

    def reload_terminology(self) -> None:
        """重新加载术语库"""
        print("\n=== 重新加载术语库 ===")
        if self.setup_terminology():
            # 刷新翻译器的向量存储
            if self.translator:
                self.translator.terminology_manager = self.terminology_manager
                self.translator.refresh_vector_store()
                print("✅ 术语库和向量存储已更新")

    def manage_vector_store(self) -> None:
        """管理向量存储"""
        print("\n=== 向量存储管理 ===")

        if not self.translator:
            print("❌ 翻译器未初始化")
            return

        # 显示当前状态
        if self.translator.vector_store:
            try:
                existing_ids = self.translator._get_existing_term_ids()
                print(f"当前状态: 已索引 {len(existing_ids)} 个文档")

                # 检查向量存储目录大小
                import os
                if os.path.exists("./chroma_db"):
                    total_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk("./chroma_db")
                        for filename in filenames
                    )
                    size_mb = total_size / (1024 * 1024)
                    print(f"存储大小: {size_mb:.2f} MB")
            except Exception as e:
                print(f"状态检查失败: {e}")
        else:
            print("当前状态: 向量存储未初始化")

        print("\n管理选项:")
        print("1. 增量更新向量存储")
        print("2. 强制重建向量存储")
        print("3. 清空向量存储")
        print("4. 查看向量存储统计")
        print("5. 返回主菜单")

        choice = input("请选择 (1-5): ").strip()

        if choice == "1":
            print("🔄 正在增量更新向量存储...")
            self.translator._setup_vector_store()
            print("✅ 向量存储更新完成")

        elif choice == "2":
            confirm = input("⚠️  强制重建将删除所有现有索引，是否继续？(y/n): ").lower().strip()
            if confirm == 'y':
                print("🔄 正在强制重建向量存储...")
                self.translator.force_rebuild_vector_store()
                print("✅ 向量存储重建完成")
            else:
                print("操作已取消")

        elif choice == "3":
            confirm = input("⚠️  这将删除所有向量索引，是否继续？(y/n): ").lower().strip()
            if confirm == 'y':
                if self.translator.clear_vector_store():
                    print("✅ 向量存储已清空")
                else:
                    print("❌ 清空向量存储失败")
            else:
                print("操作已取消")

        elif choice == "4":
            self._show_vector_store_stats()

        elif choice == "5":
            return

    def _show_vector_store_stats(self) -> None:
        """显示向量存储统计信息"""
        if not self.translator or not self.translator.vector_store:
            print("❌ 向量存储未初始化")
            return

        try:
            collection = self.translator.vector_store._collection
            all_docs = collection.get()

            total_docs = len(all_docs['ids']) if all_docs['ids'] else 0

            # 统计文档类型
            english_docs = 0
            chinese_docs = 0

            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if metadata and 'type' in metadata:
                        if metadata['type'] == 'english_term':
                            english_docs += 1
                        elif metadata['type'] == 'chinese_term':
                            chinese_docs += 1

            print(f"\n📊 向量存储统计:")
            print(f"  - 总文档数: {total_docs}")
            print(f"  - 英文术语文档: {english_docs}")
            print(f"  - 中文术语文档: {chinese_docs}")

            # 显示存储位置和大小
            import os
            if os.path.exists("./chroma_db"):
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk("./chroma_db")
                    for filename in filenames
                )
                size_mb = total_size / (1024 * 1024)
                print(f"  - 存储位置: ./chroma_db")
                print(f"  - 存储大小: {size_mb:.2f} MB")

        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")

    def run(self) -> None:
        """运行应用"""
        print("🌟 欢迎使用TMC翻译系统!")
        print("这是一个基于RAG的中英文术语翻译工具")
        print("支持同时使用标准术语库和Minecraft语言文件")
        print("✨ 新功能: 智能向量存储，避免重复索引")

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
                choice = input("\n请选择 (1-7): ").strip()

                if choice == "1":
                    self.translate_text()
                elif choice == "2":
                    self.show_terminology()
                elif choice == "3":
                    self.manage_terminology()
                elif choice == "4":
                    self.adjust_search_parameters()
                elif choice == "5":
                    self.manage_vector_store()
                elif choice == "6":
                    if self.setup_model():
                        print("✅ 模型切换成功")
                elif choice == "7":
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
