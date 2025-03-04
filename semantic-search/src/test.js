import { ChatOpenAI } from '@langchain/openai';

// 配置 OpenRouter 的 API 密钥和端点
const openRouterApiKey =
  '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'; // 替换为您的 OpenRouter API 密钥
const openRouterApiBase = 'https://openrouter.ai/api/v1'; // OpenRouter 的 API 端点
const modelName = 'google/gemini-2.0-flash-001'; // 替换为您在 OpenRouter 上使用的模型名称

const model = new ChatOpenAI({
  modelName,
  openAIApiKey: openRouterApiKey,
  configuration: {
    baseURL: openRouterApiBase,
  },
});

// 使用示例
async function runChat(prompt) {
  try {
    console.log('正在发送请求到 OpenRouter...');

    const response = await model.invoke([
      { role: 'system', content: '你是一个有帮助的AI助手。' },
      { role: 'user', content: prompt },
    ]);

    console.log('收到回复:');
    console.log(response.content);
    return response.content;
  } catch (error) {
    console.error('Error details:', error.message);
    console.error('Full error:', error);
  }
}

// 正确处理异步函数
(async () => {
  try {
    await runChat('请简要介绍一下 LangChain。');
    console.log('程序执行完毕');
  } catch (error) {
    console.error('主程序错误:', error);
  }
})();
