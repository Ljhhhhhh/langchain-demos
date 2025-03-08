/**
 * LangChain聊天机器人示例
 *
 * 本示例展示如何使用LangChain和LangGraph构建一个功能完善的聊天机器人。
 * 特点包括：
 * - 消息历史记录管理
 * - 自定义提示模板
 * - 多语言支持
 * - 消息修剪功能（防止聊天记录过长）
 * - 使用OpenRouter API访问各种大语言模型
 */

import dotenv from 'dotenv';
import { v4 as uuidv4 } from 'uuid';
import { ChatOpenAI } from '@langchain/openai';
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

// 定义消息修剪器 - 控制聊天历史长度，避免超出模型上下文窗口
const trimmer = trimMessages({
  maxTokens: 4000, // 最大token数量
  tokenCounter: llm.getNumTokens, // 添加这一行，使用LLM实例的token计数方法
  strategy: 'last', // 保留最新的消息
  includeSystem: true, // 始终保留系统消息
  allowPartial: false, // 不允许部分消息（要么完整保留消息，要么完全删除）
  startOn: 'human', // 从人类消息开始计数
});

// 定义提示模板（支持多语言）
const promptTemplate = ChatPromptTemplate.fromMessages([
  [
    'system',
    '你是一个智能助手，擅长用友好和专业的方式进行对话。请用{language}回答用户的问题。',
  ],
  ['placeholder', '{messages}'], // messages占位符将被实际消息替换
]);

// 定义图表注释（Graph Annotation）
const ChatBotAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  language: Annotation(), // 添加language属性
});

// 定义调用模型的函数
const callModel = async (state) => {
  // 1. 修剪消息，避免过长
  const trimmedMessages = await trimmer.invoke(state.messages);

  // 2. 使用提示模板格式化输入
  const prompt = await promptTemplate.invoke({
    messages: trimmedMessages,
    language: state.language || process.env.DEFAULT_LANGUAGE || '中文',
  });

  // 3. 调用LLM获取响应
  const response = await llm.invoke(prompt);

  // 4. 返回更新后的消息
  return { messages: [response] };
};

// 创建状态图 - 定义聊天机器人的工作流
const workflow = new StateGraph(ChatBotAnnotation)
  .addNode('model', callModel)
  .addEdge(START, 'model')
  .addEdge('model', END);

// 添加内存保存器 - 用于保存聊天历史
const memory = new MemorySaver();
const chatbot = workflow.compile({ checkpointer: memory });

// 示例使用
async function runExample() {
  // 创建唯一的会话ID
  const sessionId = uuidv4();
  const config = { configurable: { thread_id: sessionId } };

  console.log('===== 聊天机器人示例 =====');
  console.log('会话ID:', sessionId);

  // 第一个用户消息
  const input1 = {
    messages: [
      {
        role: 'user',
        content: '你好！我叫小明。',
      },
    ],
    language: '中文',
  };

  console.log('\n用户: 你好！我叫小明。');
  const output1 = await chatbot.invoke(input1, config);
  console.log('机器人:', output1.messages[output1.messages.length - 1].content);

  // 第二个用户消息
  const input2 = {
    messages: [
      {
        role: 'user',
        content: '我的名字是什么？',
      },
    ],
  };

  console.log('\n用户: 我的名字是什么？');
  const output2 = await chatbot.invoke(input2, config);
  console.log('机器人:', output2.messages[output2.messages.length - 1].content);

  // 第三个用户消息 - 切换语言
  const input3 = {
    messages: [
      {
        role: 'user',
        content: '你可以用英语和我对话吗？',
      },
    ],
  };

  console.log('\n用户: 你可以用英语和我对话吗？');
  const output3 = await chatbot.invoke(input3, config);
  console.log('机器人:', output3.messages[output3.messages.length - 1].content);

  // 第四个用户消息 - 英语对话
  const input4 = {
    messages: [
      {
        role: 'user',
        content: 'What is my name?',
      },
    ],
    language: 'English',
  };

  console.log('\n用户: What is my name?');
  const output4 = await chatbot.invoke(input4, config);
  console.log('机器人:', output4.messages[output4.messages.length - 1].content);
}

// 运行示例
runExample().catch(console.error);

// 导出聊天机器人以供其他模块使用
export { chatbot };
