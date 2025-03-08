/**
 * 具有检索增强生成（RAG）功能的高级聊天机器人
 *
 * 这个实现在基础聊天机器人的基础上，添加了检索增强生成（RAG）能力。
 * 机器人可以检索参考文档来回答用户问题，提供更准确和有依据的回答。
 */

import dotenv from 'dotenv';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';
import { ChatOpenAI } from '@langchain/openai';
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import {
  trimMessages,
  SystemMessage,
  HumanMessage,
  AIMessage,
} from '@langchain/core/messages';
import {
  START,
  END,
  StateGraph,
  MemorySaver,
  MessagesAnnotation,
  Annotation,
} from '@langchain/langgraph';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { StringOutputParser } from '@langchain/core/output_parsers';

// 加载环境变量
dotenv.config();

// 初始化LLM（使用OpenRouter访问Gemini模型）
const llm = new ChatOpenAI({
  modelName: 'google/gemini-2.0-flash-001',
  openAIApiKey: process.env.OPENROUTER_API_KEY,
  configuration: {
    baseURL: 'https://openrouter.ai/api/v1',
  },
});

// 初始化嵌入模型
const embeddings = new OpenAIEmbeddings({
  modelName: 'Pro/BAAI/bge-m3',
  openAIApiKey: process.env.SILICONFLOW_API_KEY,
  configuration: {
    baseURL: 'https://api.siliconflow.cn/v1',
  },
});

// 内存向量数据库
let vectorStore;

// 定义消息修剪器
const trimmer = trimMessages({
  maxTokens: 4000,
  tokenCounter: llm.getNumTokens,
  strategy: 'last',
  includeSystem: true,
  allowPartial: false,
  startOn: 'human',
});

// 系统提示
const SYSTEM_TEMPLATE = `你是一个智能助手，具有检索增强生成能力。
请用{language}回答用户的问题。

当用户提出问题时，你可以查询你的知识库。
如果知识库中有相关信息，请使用这些信息来提供准确的回答。
始终明确指出你的回答来源于检索到的知识。
如果检索结果不包含问题的答案，请坦率地告知用户你无法基于可用文档回答问题，然后尝试用你自己的知识提供帮助。

当回答时：
1. 保持回答简洁明了
2. 给出有根据的回答
3. 如果引用了检索内容，明确指出内容来源
4. 不要编造信息或提供误导性回答`;

// 聊天提示模板
const chatPromptTemplate = ChatPromptTemplate.fromMessages([
  ['system', SYSTEM_TEMPLATE],
  ['placeholder', '{messages}'],
]);

// 检索提示模板
const retrievalPromptTemplate = ChatPromptTemplate.fromMessages([
  [
    'system',
    '确定用户最后一条消息是否是查询问题，需要进行文档检索。仅返回"是"或"否"。',
  ],
  ['placeholder', '{messages}'],
]);

// RAG聊天机器人的状态定义
const RAGBotAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  language: Annotation(),
  retrieval_documents: Annotation(),
  route: Annotation(),
});

// 初始化向量数据库
async function initializeVectorStore() {
  try {
    // 定义示例文档 - 在实际场景中，您可能会从文件或数据库中加载这些
    const documents = [
      'LangChain是一个用于开发由语言模型驱动的应用程序的框架。它可以帮助构建各种应用，如聊天机器人、问答系统、摘要生成等。',
      'LangGraph是LangChain的扩展，专注于构建有状态的多步骤AI工作流。它允许开发者定义节点和边，创建复杂的状态管理流程。',
      '检索增强生成（RAG）是一种技术，它将大型语言模型与外部知识源结合起来。RAG通过检索相关文档然后使用这些文档生成回答，提高了回答的准确性。',
      '消息历史管理是构建聊天机器人的关键部分。如果不妥善管理，聊天历史可能会无限增长并超出模型的上下文窗口限制。',
      '提示模板可以帮助将原始用户输入转化为更适合语言模型处理的格式。在LangChain中，可以使用ChatPromptTemplate创建复杂的提示模板。',
      '聊天机器人通常需要支持多语言，以便服务全球用户。在LangChain中，可以通过在提示模板中添加语言参数来实现多语言支持。',
    ];

    // 文本分割器
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    // 分割文档
    const splitDocuments = await textSplitter.createDocuments(documents);

    // 创建向量存储
    vectorStore = await MemoryVectorStore.fromDocuments(
      splitDocuments,
      embeddings,
    );

    console.log('向量数据库初始化成功，已添加', documents.length, '条文档');
  } catch (error) {
    console.error('初始化向量数据库失败:', error);
    throw error;
  }
}

// 判断是否需要检索
const shouldRetrieve = async (state) => {
  // 获取最后一条用户消息
  const userMessages = state.messages.filter((m) => m.role === 'user');
  if (userMessages.length === 0) return false;

  const lastUserMessage = userMessages[userMessages.length - 1].content;

  // 使用LLM判断是否需要检索
  const response = await llm.invoke([
    {
      role: 'system',
      content:
        '这条信息是否是查询问题，需要检索信息来回答？请只回答"是"或"否"。',
    },
    { role: 'user', content: lastUserMessage },
  ]);

  return response.content.toLowerCase().includes('是');
};

// 检索相关文档
const retrieveDocuments = async (state) => {
  // 获取最后一条用户消息
  const userMessages = state.messages.filter((m) => m.role === 'user');
  if (userMessages.length === 0) return { retrieval_documents: [] };

  const lastUserMessage = userMessages[userMessages.length - 1].content;

  // 确保向量数据库已初始化
  if (!vectorStore) {
    await initializeVectorStore();
  }

  // 检索相关文档
  const retriever = vectorStore.asRetriever();
  const documents = await retriever.invoke(lastUserMessage);

  // 提取文档内容
  const documentContents = documents.map((doc) => doc.pageContent);

  return { retrieval_documents: documentContents };
};

// 生成回答
const generateResponse = async (state) => {
  // 修剪消息历史
  const trimmedMessages = await trimmer.invoke(state.messages);

  // 准备系统提示
  let systemPrompt = SYSTEM_TEMPLATE;

  // 如果有检索到的文档，将其添加到提示中
  if (state.retrieval_documents && state.retrieval_documents.length > 0) {
    systemPrompt += `\n\n以下是与查询相关的检索结果：\n${state.retrieval_documents.join(
      '\n\n',
    )}`;
  }

  // 创建提示模板
  const promptWithDocs = ChatPromptTemplate.fromMessages([
    ['system', systemPrompt],
    ['placeholder', '{messages}'],
  ]);

  // 生成提示
  const prompt = await promptWithDocs.invoke({
    messages: trimmedMessages,
    language: state.language || process.env.DEFAULT_LANGUAGE || '中文',
  });

  // 调用LLM获取响应
  const response = await llm.invoke(prompt);

  // 返回更新后的消息
  return { messages: [response] };
};

// 条件节点
const routeBasedOnRetrieval = async (state) => {
  // 判断是否需要检索
  const needsRetrieval = await shouldRetrieve(state);
  // 返回路由指令，但作为state的一部分
  return { route: needsRetrieval ? 'retrieve' : 'generate' };
};

// 创建状态图
const workflow = new StateGraph(RAGBotAnnotation)
  .addNode('router', routeBasedOnRetrieval)
  .addNode('retrieve', retrieveDocuments)
  .addNode('generate', generateResponse)
  .addEdge(START, 'router')
  .addConditionalEdges('router', (state) => state.route)
  .addEdge('retrieve', 'generate')
  .addEdge('generate', END);

// 编译工作流
const memory = new MemorySaver();
const ragChatbot = workflow.compile({ checkpointer: memory });

// 运行示例
async function runExample() {
  // 初始化向量数据库
  await initializeVectorStore();

  // 创建唯一的会话ID
  const sessionId = uuidv4();
  const config = { configurable: { thread_id: sessionId } };

  console.log('===== RAG聊天机器人示例 =====');
  console.log('会话ID:', sessionId);

  // 示例对话
  const examples = [
    {
      messages: [{ role: 'user', content: '你好！介绍一下自己。' }],
      language: '中文',
    },
    {
      messages: [{ role: 'user', content: 'LangChain是什么？' }],
    },
    {
      messages: [{ role: 'user', content: '什么是检索增强生成？' }],
    },
    {
      messages: [{ role: 'user', content: '谢谢你的解释，我明白了！' }],
    },
  ];

  // 依次执行每个示例
  for (const example of examples) {
    console.log('\n用户:', example.messages[0].content);
    console.log('处理中...');

    const output = await ragChatbot.invoke(example, config);
    console.log('机器人:', output.messages[output.messages.length - 1].content);
  }
}

// 导出聊天机器人和运行示例函数
export { ragChatbot, runExample, initializeVectorStore };

// 如果直接运行此文件，则执行示例
if (import.meta.url === `file://${process.argv[1]}`) {
  runExample().catch(console.error);
}
