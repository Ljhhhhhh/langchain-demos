/**
 * 具有检索增强生成（RAG）功能的高级聊天机器人
 *
 * 这个实现在基础聊天机器人的基础上，添加了检索增强生成（RAG）能力。
 * 机器人可以检索参考文档来回答用户问题，提供更准确和有依据的回答。
 */

import dotenv from 'dotenv';
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

// 聊天提示模板 - 用于常规对话响应
// 在LangChain中，提示模板是构建高质量提示的基础组件
// 这里我们定义了基本的聊天模板，包含系统指令和消息占位符
const chatPromptTemplate = ChatPromptTemplate.fromMessages([
  ['system', SYSTEM_TEMPLATE], // 系统提示定义了助手的角色和行为准则
  ['placeholder', '{messages}'], // 消息占位符会在运行时被实际消息替换
]);

// 检索提示模板 - 专门用于判断用户问题是否需要检索知识库
// 这是RAG系统的关键组件，它决定了何时使用检索功能
// 使用专门的提示模板可以保持代码结构清晰，易于维护
const retrievalPromptTemplate = ChatPromptTemplate.fromMessages([
  [
    'system',
    '确定用户最后一条消息是否是查询问题，需要进行文档检索。仅返回"是"或"否"。',
  ],
  ['placeholder', '{messages}'], // 将包含用户的最后一条消息
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
// 这个函数实现了RAG的"路由"功能 - 决定是否需要检索外部知识
const shouldRetrieve = async (state) => {
  // 获取最后一条用户消息
  const userMessages = state.messages.filter((m) => m.role === 'user');
  if (userMessages.length === 0) return false;

  const lastUserMessage = userMessages[userMessages.length - 1].content;
  console.log('检查消息是否需要检索:', lastUserMessage);

  // 硬编码一些关键词触发检索
  const knowledgeQueryKeywords = [
    'langchain',
    'langgraph',
    '检索',
    '增强',
    'rag',
    '框架',
    '知识',
    '什么是',
    '如何',
    '为什么',
    '解释',
    '说明',
    '介绍',
  ];

  // 检查是否包含知识查询关键词
  const containsKeyword = knowledgeQueryKeywords.some((keyword) =>
    lastUserMessage.toLowerCase().includes(keyword.toLowerCase()),
  );

  if (containsKeyword) {
    console.log('关键词匹配成功，触发检索');
    return true;
  }

  // 直接使用LLM进行判断，确保系统提示被正确使用
  const response = await llm.invoke([
    {
      role: 'system',
      content:
        '确定用户最后一条消息是否是查询问题，需要进行文档检索。这是很重要的判断，会决定是否使用RAG功能。明确的知识查询应该返回"是"。一般问候、感谢或闲聊应该返回"否"。仅返回"是"或"否"，不要有任何解释。',
    },
    {
      role: 'user',
      content: `用户消息: "${lastUserMessage}" - 这是否是需要检索外部知识的查询问题？`,
    },
  ]);

  console.log('LLM检索判断:', response.content);

  const shouldRetrieve =
    response.content.toLowerCase().includes('是') || containsKeyword;
  console.log('最终检索决策:', shouldRetrieve ? '需要检索' : '不需要检索');

  return shouldRetrieve;
};

// 检索相关文档
const retrieveDocuments = async (state) => {
  // 获取最后一条用户消息
  const userMessages = state.messages.filter((m) => m.role === 'user');
  if (userMessages.length === 0) return { retrieval_documents: [] };

  const lastUserMessage = userMessages[userMessages.length - 1].content;
  console.log('开始检索相关文档，关键词:', lastUserMessage);

  // 确保向量数据库已初始化
  if (!vectorStore) {
    await initializeVectorStore();
  }

  // 检索相关文档，设置较低的相似度阈值，确保能找到相关文档
  const retriever = vectorStore.asRetriever({
    k: 3, // 增加返回文档数量
    searchType: 'similarity',
    filter: null,
  });

  // 执行检索
  const documents = await retriever.invoke(lastUserMessage);

  // 记录检索结果
  console.log(`检索到 ${documents.length} 条相关文档`);

  // 提取文档内容
  const documentContents = documents.map((doc) => doc.pageContent);

  // 只有在确实找到文档时才返回结果
  if (documentContents.length > 0) {
    console.log('检索成功，找到相关文档');
    return { retrieval_documents: documentContents };
  } else {
    console.log('未找到相关文档');
    return { retrieval_documents: [] };
  }
};

// 生成回答
// 这个函数展示了如何根据不同情况使用不同的提示模板
const generateResponse = async (state) => {
  // 使用trimmer修剪消息历史，避免超出模型的上下文窗口限制
  // 这是处理长对话的最佳实践
  const trimmedMessages = await trimmer.invoke(state.messages);

  // 使用条件分支选择合适的提示模板
  // 这是RAG系统的核心 - 根据是否有检索结果选择合适的响应策略
  if (state.retrieval_documents && state.retrieval_documents.length > 0) {
    console.log('使用RAG模式生成回答，包含检索结果');
    console.log(`检索文档数量: ${state.retrieval_documents.length}`);

    // 有检索结果时，将结果融入系统提示
    // 这是RAG的"增强"部分 - 将检索到的知识注入到生成过程中
    const systemPromptWithDocs = `${SYSTEM_TEMPLATE}

以下是与查询相关的检索结果：
${state.retrieval_documents.join('\n\n')}

重要说明：
1. 你必须基于这些检索结果回答问题
2. 如果检索结果与问题相关，请明确引用这些内容
3. 在回答开头表明你是基于检索结果回答的
4. 即使检索结果不完全相关，也要尽量从中提取有用信息`;

    // 动态创建包含检索结果的提示模板
    // 这展示了提示模板的灵活性 - 可以根据需要动态构建
    const promptWithDocs = ChatPromptTemplate.fromMessages([
      ['system', systemPromptWithDocs],
      ['placeholder', '{messages}'],
    ]);

    // 使用动态提示模板生成最终提示
    const prompt = await promptWithDocs.invoke({
      messages: trimmedMessages,
      language: state.language || process.env.DEFAULT_LANGUAGE || '中文',
    });

    // 调用LLM获取增强的响应
    const response = await llm.invoke(prompt);
    console.log('使用检索结果生成回答完成');

    // 返回更新后的消息作为状态更新
    return { messages: [response] };
  } else {
    console.log('使用标准模式生成回答，不包含检索结果');
    // 没有检索结果时使用预定义的标准聊天模板
    // 重用预定义模板提高了代码的一致性和可维护性
    const prompt = await chatPromptTemplate.invoke({
      messages: trimmedMessages,
      language: state.language || process.env.DEFAULT_LANGUAGE || '中文',
    });

    // 调用LLM获取常规响应
    const response = await llm.invoke(prompt);
    console.log('标准回答生成完成');

    // 返回更新后的消息作为状态更新
    return { messages: [response] };
  }
};

// 条件节点
const routeBasedOnRetrieval = async (state) => {
  // 判断是否需要检索
  const needsRetrieval = await shouldRetrieve(state);

  // 添加日志输出，检查路由决策
  const route = needsRetrieval ? 'retrieve' : 'generate';
  console.log(`路由决策: ${route} (需要检索: ${needsRetrieval})`);

  // 返回路由指令，但作为state的一部分
  return { route };
};

// 创建状态图
const workflow = new StateGraph(RAGBotAnnotation)
  .addNode('router', routeBasedOnRetrieval)
  .addNode('retrieve', retrieveDocuments)
  .addNode('generate', generateResponse)
  .addEdge(START, 'router')
  .addConditionalEdges('router', (state) => {
    console.log('条件边路由状态:', state.route);
    return state.route;
  })
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

  // 创建一个累积的消息历史
  let messageHistory = [];

  // 预先获取检索结果，强制使用RAG
  console.log('\n======预先检索知识库中的信息======');

  // 直接检索LangChain相关信息
  const retriever = vectorStore.asRetriever({
    k: 3, // 返回3个相关文档
  });

  const langchainDocs = await retriever.invoke(
    'LangChain是什么？它有什么用途？',
  );
  console.log(`预先检索到 ${langchainDocs.length} 条关于LangChain的文档`);

  const langchainInfo = langchainDocs.map((doc) => doc.pageContent);

  // 直接检索RAG相关信息
  const ragDocs = await retriever.invoke(
    '什么是检索增强生成技术？RAG有什么优势？',
  );
  console.log(`预先检索到 ${ragDocs.length} 条关于RAG的文档`);

  const ragInfo = ragDocs.map((doc) => doc.pageContent);
  console.log('======预先检索完成======\n');

  // 示例对话
  const examples = [
    {
      content: '你好！介绍一下自己。',
      language: '中文',
      useRAG: false,
      retrieval_documents: [],
    },
    {
      content: 'LangChain是什么？',
      language: '中文',
      useRAG: true,
      retrieval_documents: langchainInfo,
    },
    {
      content: '什么是检索增强生成？',
      language: '中文',
      useRAG: true,
      retrieval_documents: ragInfo,
    },
    {
      content: '谢谢你的解释，我明白了！',
      language: '中文',
      useRAG: false,
      retrieval_documents: [],
    },
  ];

  // 依次执行每个示例
  for (const example of examples) {
    // 添加用户消息到历史
    const userMessage = { role: 'user', content: example.content };
    messageHistory.push(userMessage);

    // 构建输入状态，如果启用RAG则直接提供检索文档
    const input = {
      messages: messageHistory,
      language: example.language,
    };

    // 如果启用RAG，直接在输入中设置检索文档
    if (example.useRAG && example.retrieval_documents.length > 0) {
      input.retrieval_documents = example.retrieval_documents;
      console.log(`\n用户: ${example.content} [RAG模式已启用]`);
    } else {
      console.log(`\n用户: ${example.content}`);
    }

    console.log('处理中...');

    const output = await ragChatbot.invoke(input, config);

    // 获取AI回复并添加到历史
    const aiMessage = output.messages[output.messages.length - 1];
    messageHistory.push(aiMessage);

    console.log('机器人:', aiMessage.content);
  }
}

// 导出聊天机器人和运行示例函数
export { ragChatbot, runExample, initializeVectorStore };

// 如果直接运行此文件，则执行示例
if (import.meta.url === `file://${process.argv[1]}`) {
  runExample().catch(console.error);
}
