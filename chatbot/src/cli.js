/**
 * 聊天机器人命令行交互界面
 *
 * 该文件提供了一个简单的命令行界面，允许用户直接与聊天机器人进行交互。
 * 支持特殊命令：
 * - /exit 或 /quit: 退出程序
 * - /lang [语言]: 切换语言 (例如: /lang English)
 * - /new: 开始新的会话
 * - /help: 显示帮助信息
 */

import readline from 'readline';
import dotenv from 'dotenv';
import { v4 as uuidv4 } from 'uuid';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { trimMessages, SystemMessage } from '@langchain/core/messages';
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

// 创建readline接口
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

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
  maxTokens: 4000,
  tokenCounter: llm.getNumTokens,
  strategy: 'last',
  includeSystem: true,
  allowPartial: false,
  startOn: 'human',
});

// 定义提示模板（支持多语言）
const promptTemplate = ChatPromptTemplate.fromMessages([
  [
    'system',
    '你是一个智能助手，擅长用友好和专业的方式进行对话。请用{language}回答用户的问题。' +
      '提供有深度和有帮助的回答，但要简洁明了。' +
      '如果你不知道答案，请承认你不知道而不是编造信息。',
  ],
  ['placeholder', '{messages}'],
]);

// 定义图表注释
const ChatBotAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  language: Annotation(),
});

// 定义调用模型的函数
const callModel = async (state) => {
  // 修剪消息
  const trimmedMessages = await trimmer.invoke(state.messages);

  // 使用提示模板格式化输入
  const prompt = await promptTemplate.invoke({
    messages: trimmedMessages,
    language: state.language || process.env.DEFAULT_LANGUAGE || '中文',
  });

  // 调用LLM获取响应
  const response = await llm.invoke(prompt);

  // 返回更新后的消息
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

// 当前会话状态
let sessionId = uuidv4();
let currentLanguage = process.env.DEFAULT_LANGUAGE || '中文';

// 显示欢迎消息
console.log('=== LangChain 聊天机器人 CLI 界面 ===');
console.log('输入您的消息与机器人交流。特殊命令：');
console.log(' - /exit 或 /quit: 退出程序');
console.log(' - /lang [语言]: 切换语言 (例如: /lang English)');
console.log(' - /new: 开始新的会话');
console.log(' - /help: 显示此帮助信息');
console.log('===============================');

// 处理用户输入
async function processUserInput(input) {
  // 检查是否是特殊命令
  if (input.startsWith('/')) {
    const command = input.split(' ')[0].toLowerCase();

    switch (command) {
      case '/exit':
      case '/quit':
        console.log('再见！感谢使用聊天机器人。');
        rl.close();
        process.exit(0);
        break;

      case '/lang':
        const newLang = input.substring(6).trim();
        if (newLang) {
          currentLanguage = newLang;
          console.log(`已切换语言到: ${currentLanguage}`);
        } else {
          console.log(`当前语言: ${currentLanguage}`);
        }
        askQuestion();
        break;

      case '/new':
        sessionId = uuidv4();
        console.log(`已开始新的会话，会话ID: ${sessionId}`);
        askQuestion();
        break;

      case '/help':
        console.log('特殊命令：');
        console.log(' - /exit 或 /quit: 退出程序');
        console.log(' - /lang [语言]: 切换语言 (例如: /lang English)');
        console.log(' - /new: 开始新的会话');
        console.log(' - /help: 显示此帮助信息');
        askQuestion();
        break;

      default:
        console.log(`未知命令: ${command}`);
        askQuestion();
        break;
    }
    return;
  }

  // 常规消息处理
  try {
    const config = { configurable: { thread_id: sessionId } };

    const userInput = {
      messages: [
        {
          role: 'user',
          content: input,
        },
      ],
      language: currentLanguage,
    };

    console.log('处理中...');
    const output = await chatbot.invoke(userInput, config);
    const response = output.messages[output.messages.length - 1].content;

    console.log('\n🤖 机器人: ' + response + '\n');
  } catch (error) {
    console.error('发生错误:', error.message);
  }

  askQuestion();
}

// 提示用户输入问题
function askQuestion() {
  rl.question('🧑 用户: ', processUserInput);
}

// 开始对话
askQuestion();

// 处理程序退出
process.on('SIGINT', () => {
  console.log('\n再见！感谢使用聊天机器人。');
  rl.close();
  process.exit(0);
});
