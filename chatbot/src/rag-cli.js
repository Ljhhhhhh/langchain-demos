/**
 * RAG聊天机器人命令行交互界面
 *
 * 该文件提供了一个命令行界面，允许用户与具有检索增强生成能力的聊天机器人进行交互。
 * 支持特殊命令：
 * - /exit 或 /quit: 退出程序
 * - /lang [语言]: 切换语言 (例如: /lang English)
 * - /new: 开始新的会话
 * - /info: 显示检索到的文档信息
 * - /help: 显示帮助信息
 */

import readline from 'readline';
import dotenv from 'dotenv';
import { v4 as uuidv4 } from 'uuid';
import { ragChatbot, initializeVectorStore } from './rag-chatbot.js';

// 加载环境变量
dotenv.config();

// 创建readline接口
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// 当前会话状态
let sessionId = uuidv4();
let currentLanguage = process.env.DEFAULT_LANGUAGE || '中文';
let lastRetrievedDocs = [];

// 初始化
async function initialize() {
  console.log('正在初始化向量数据库...');
  await initializeVectorStore();

  // 显示欢迎消息
  console.log('\n=== LangChain RAG聊天机器人 CLI 界面 ===');
  console.log(
    '这个聊天机器人具有检索增强生成（RAG）能力，可以检索文档来回答您的问题。',
  );
  console.log('输入您的消息与机器人交流。特殊命令：');
  console.log(' - /exit 或 /quit: 退出程序');
  console.log(' - /lang [语言]: 切换语言 (例如: /lang English)');
  console.log(' - /new: 开始新的会话');
  console.log(' - /info: 显示上次检索到的文档信息');
  console.log(' - /help: 显示此帮助信息');
  console.log('=======================================');

  // 开始交互
  askQuestion();
}

// 处理用户输入
async function processUserInput(input) {
  // 检查是否是特殊命令
  if (input.startsWith('/')) {
    const command = input.split(' ')[0].toLowerCase();

    switch (command) {
      case '/exit':
      case '/quit':
        console.log('再见！感谢使用RAG聊天机器人。');
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
        lastRetrievedDocs = [];
        console.log(`已开始新的会话，会话ID: ${sessionId}`);
        askQuestion();
        break;

      case '/info':
        if (lastRetrievedDocs.length > 0) {
          console.log('\n上次检索到的文档:');
          lastRetrievedDocs.forEach((doc, index) => {
            console.log(`[${index + 1}] ${doc.substring(0, 150)}...`);
          });
        } else {
          console.log('没有检索到的文档信息。');
        }
        askQuestion();
        break;

      case '/help':
        console.log('特殊命令：');
        console.log(' - /exit 或 /quit: 退出程序');
        console.log(' - /lang [语言]: 切换语言 (例如: /lang English)');
        console.log(' - /new: 开始新的会话');
        console.log(' - /info: 显示上次检索到的文档信息');
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
    const output = await ragChatbot.invoke(userInput, config);

    // 保存检索到的文档
    if (output.retrieval_documents && output.retrieval_documents.length > 0) {
      lastRetrievedDocs = output.retrieval_documents;
      console.log(`\n[系统] 检索到 ${lastRetrievedDocs.length} 条相关文档`);
    }

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

// 处理程序退出
process.on('SIGINT', () => {
  console.log('\n再见！感谢使用RAG聊天机器人。');
  rl.close();
  process.exit(0);
});

// 初始化并开始对话
initialize().catch((error) => {
  console.error('初始化失败:', error);
  process.exit(1);
});
