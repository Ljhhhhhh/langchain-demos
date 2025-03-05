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
  modelName: 'deepseek-ai/DeepSeek-V3',
  openAIApiKey: process.env.SILICONFLOW_API_KEY,
  configuration: {
    baseURL: 'https://api.siliconflow.cn/v1',
  },
});

const classificationSchema = z.object({
  sentiment: z.string().describe('文本的情感'),
  aggressiveness: z
    .number()
    .int()
    .min(1)
    .max(10)
    .describe('文本的攻击性程度，范围从1到10。'),
  language: z.string().describe('文本所使用的语言'),
});

const llmWihStructuredOutput = llm.withStructuredOutput(classificationSchema, {
  name: 'extractor',
});

(async () => {
  const taggingPrompt = ChatPromptTemplate.fromTemplate(
    `从以下文章中提取所需信息，仅提取${classificationSchema}中提到的属性。

    文章:
{input}
`,
  );

  const prompt1 = await taggingPrompt.invoke({
    input: '很高兴认识你！我觉得我们会成为很好的朋友！',
  });

  console.log(prompt1.toString(), 'prompt1');
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
