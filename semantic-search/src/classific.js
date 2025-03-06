import dotenv from 'dotenv';
import path from 'path';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { ChatOpenAI } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { z } from 'zod';

dotenv.config();

console.log(process.env.SILICONFLOW_API_KEY, 'process.env.SILICONFLOW_API_KEY');

const llm = new ChatOpenAI({
  modelName: 'google/gemini-2.0-flash-001', // 'deepseek-ai/DeepSeek-V3',
  openAIApiKey:
    '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', // process.env.SILICONFLOW_API_KEY,
  configuration: {
    baseURL: 'https://openrouter.ai/api/v1', // 'https://api.siliconflow.cn/v1',
  },
});

const classificationSchema = z.object({
  sentiment: z
    .enum(['积极', '消极', '中性'])
    .describe('文本的情感，用中文表示，如积极、消极、中性等'),
  aggressiveness: z
    .number()
    .int()
    .min(1)
    .max(10)
    .describe('文本的攻击性程度，范围从1到10。'),
  language: z
    .enum(['中文', '英文', '日文', '韩文', '其他'])
    .describe('文本所使用的语种'),
});

const llmWihStructuredOutput = llm.withStructuredOutput(classificationSchema, {
  name: 'extractor',
});

(async () => {
  const taggingPrompt = ChatPromptTemplate.fromTemplate(
    `从以下文章中提取所需信息，仅提取 classificationSchema 中提到的属性。

    文章:
{input}

请仅返回请求的属性，不要添加额外解释。
`,
  );

  const prompt1 = await taggingPrompt.invoke({
    input: '明天是不是要下雨了？那也没关系，一样可以在家里玩耍。',
  });

  const result = await llmWihStructuredOutput.invoke(prompt1);
  console.log(result, 'result');

  // // 构造提示
  // const prompt = await promptTemplate.invoke({
  //   input: '您想处理的文本',
  // });

  // console.log(prompt, 'prompt');

  // // 调用模型获取响应
  // const response = await llm.invoke(prompt);
  // console.log(response, 'response');
})();

// // Name is optional, but gives the models more clues as to what your schema represents

// async function classify() {

// }

// (async () => {
//   await classify();
// })();
