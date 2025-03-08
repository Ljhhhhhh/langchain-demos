# LangChain 聊天机器人示例

这是一个使用 LangChain 和 LangGraph 构建的聊天机器人示例项目。该项目展示了如何创建一个具有多语言支持、消息历史管理和消息修剪功能的聊天机器人。

## 功能特点

- 基于 LangChain 和 LangGraph 构建
- 使用 OpenRouter API 作为 LLM 提供者（默认使用 Gemini 模型）
- 支持多语言对话
- 聊天历史持久化存储
- 自动消息修剪，防止超出模型上下文窗口
- 支持多会话管理（基于唯一会话 ID）
- 提供命令行交互界面

## 环境要求

- Node.js v16 或更高版本
- OpenRouter API 密钥

## 安装

1. 克隆项目并进入项目目录

```
git clone <项目地址>
cd chatbot
```

2. 安装依赖

```
npm install
```

3. 设置环境变量

创建 `.env` 文件并添加以下内容：

```
# OpenRouter API密钥
OPENROUTER_API_KEY=your_openrouter_api_key

# 可选：默认语言设置
DEFAULT_LANGUAGE=中文
```

## 使用方法

### 运行示例

```
npm start
```

这将执行 `src/index.js` 中的示例，展示一个简单的聊天对话流程。

### 命令行交互界面

```
node src/cli.js
```

命令行界面支持以下特殊命令：

- `/exit` 或 `/quit`: 退出程序
- `/lang [语言]`: 切换语言 (例如: `/lang English`)
- `/new`: 开始新的会话
- `/help`: 显示帮助信息

## 项目结构说明

- `src/index.js`: 主要聊天机器人实现
- `src/cli.js`: 命令行交互界面
- `.env`: 环境变量配置

## 核心概念解析

### LangGraph 状态管理

LangGraph 提供了一个内建的持久层，使多轮对话应用的开发变得更简单。该应用程序使用 `StateGraph` 来定义聊天机器人的工作流，并使用 `MemorySaver` 来存储聊天历史。

### 提示模板

使用 ChatPromptTemplate 可以轻松定制聊天机器人的行为和个性。在本示例中，我们使用提示模板添加了系统指令和多语言支持。

### 消息修剪

为了防止聊天历史过长超出模型的上下文窗口限制，应用程序使用 `trimMessages` 函数来限制发送给模型的消息数量。

### 多会话支持

通过为每个对话分配唯一的会话 ID（通过 `uuid` 生成），应用程序可以同时管理多个独立的对话。

## 进阶开发

如果你想扩展这个聊天机器人，可以考虑以下方向：

1. 添加检索增强生成（RAG）功能，使机器人能够访问外部知识库
2. 集成代理（Agent）能力，使机器人能够采取行动
3. 实现更多的持久化存储选项（例如数据库）
4. 添加图形用户界面（GUI）或 Web API 接口

## 相关资源

- [LangChain 官方文档](https://js.langchain.com/docs/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraphjs/)
- [OpenRouter 官方文档](https://openrouter.ai/docs)
