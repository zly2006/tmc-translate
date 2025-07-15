# TMC Translate - RAG翻译系统

基于LangChain的中英文术语翻译系统，使用RAG技术确保专业术语的准确翻译。

## 功能特性

- 🌐 **双向翻译**: 支持中文到英文和英文到中文的互相翻译
- 📚 **术语库管理**: 从Excel文件加载术语库，确保专业术语翻译的准确性
- 🤖 **多模型支持**: 支持Ollama本地模型和Google Gemini云端模型
- 🔍 **RAG技术**: 使用向量数据库进行语义搜索，提供相关术语上下文
- 💬 **交互式界面**: 提供友好的命令行交互界面
- 📊 **Excel支持**: 支持从Excel文件读取术语库数据

## 项目结构

```
src/
├── tmc_translate/
│   ├── __init__.py          # 包初始化
│   ├── main.py              # 主程序和交互界面
│   ├── models.py            # 数据模型定义
│   ├── rag_translator.py    # RAG翻译器核心
│   └── terminology_manager.py # 术语库管理器
```

## 安装

项目使用uv进行依赖管理：

```bash
# 克隆项目
git clone <repository_url>
cd tmc-translate

# 使用uv安装依赖
uv sync
```

## 术语库格式

术语库应为Excel文件(.xlsx或.xls)，包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| english_name | 英文术语名称 | Machine Learning |
| chinese_name | 中文术语名称 | 机器学习 |
| english_description | 英文描述 | A method of data analysis that automates analytical model building |
| chinese_description | 中文描述 | 一种自动化分析模型构建的数据分析方法 |

## 使用方法

### 1. 直接运行

```bash
# 使用uv运行
uv run tmc-translate

# 或者激活虚拟环境后运行
uv shell
python -m tmc_translate.main
```

### 2. 环境配置

创建`.env`文件（可选）：

```env
# Google API Key for Gemini (可选，也可以在运行时输入)
GOOGLE_API_KEY=your_google_api_key_here

# Ollama配置 (可选)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### 3. 准备术语库

- 程序首次运行时会创建示例术语库文件 `sample_terminology.xlsx`
- 您也可以准备自己的Excel术语库文件，确保包含必需的列名

### 4. 选择模型

程序支持两种模型：

#### Ollama (本地运行)
- 需要先安装并运行Ollama服务
- 推荐模型：llama2, qwen, baichuan等支持中文的模型
- 优点：本地运行，数据安全，无需API Key
- 缺点：需要本地计算资源

#### Google Gemini (云端)
- 需要Google AI Studio的API Key
- 推荐模型：gemini-pro, gemini-1.5-pro
- 优点：性能强大，响应快速
- 缺点：需要网络连接和API Key

## 主要功能

### 1. 文本翻译
- 自动检测源语言（中文/英文）
- 智能匹配相关术语
- 提供术语对照参考
- 生成高质量翻译结果

### 2. 术语库管理
- 查看所有术语
- 重新加载术语库
- 支持实时更新

### 3. 模型切换
- 运行时切换不同模型
- 支持参数调整

## 技术架构

- **LangChain**: 核心框架，提供LLM集成和链式操作
- **ChromaDB**: 向量数据库，用于术语语义搜索
- **Pandas**: 数据处理，Excel文件读取
- **dataclass**: 类型安全的数据模型
- **dotenv**: 环境变量管理

## 开发

### 添加新的模型提供者

继承`ModelProvider`抽象基类：

```python
from .rag_translator import ModelProvider

class YourModelProvider(ModelProvider):
    def get_llm(self):
        # 返回您的LLM实例
        pass
    
    def get_embeddings(self):
        # 返回您的Embeddings实例
        pass
```

### 扩展术语库格式

修改`terminology_manager.py`中的`load_from_excel`方法以支持新的列格式。

## 常见问题

### Q: Ollama连接失败怎么办？
A: 确保Ollama服务正在运行：
```bash
ollama serve
ollama pull llama2  # 下载模型
```

### Q: Gemini API调用失败？
A: 检查：
- API Key是否正确
- 网络连接是否正常
- 是否有足够的API配额

### Q: 术语库加载失败？
A: 检查Excel文件：
- 文件格式是否正确(.xlsx或.xls)
- 是否包含必需的列名
- 数据是否有空值

## 许可证

[添加您的许可证信息]

## 贡献

欢迎提交Issue和Pull Request！
