// import { Document } from '@langchain/core/documents';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import dotenv from 'dotenv';
import path from 'path';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

dotenv.config();

// const documents = [
//   new Document({
//     pageContent:
//       'Dogs are great companions, known for their loyalty and friendliness.',
//     metadata: { source: 'mammal-pets-doc' },
//   }),
//   new Document({
//     pageContent: 'Cats are independent pets that often enjoy their own space.',
//     metadata: { source: 'mammal-pets-doc' },
//   }),
// ];

const embeddings = new OpenAIEmbeddings({
  modelName: 'Pro/BAAI/bge-m3',
  openAIApiKey: process.env.SILICONFLOW_API_KEY,
  configuration: {
    baseURL: 'https://api.siliconflow.cn/v1',
  },
});

async function embed(query) {
  const assetsPath = path.join(process.cwd(), 'assets');
  const pdfPath = path.join(assetsPath, '00.pdf');
  const loader = new PDFLoader(pdfPath);

  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await textSplitter.splitDocuments(docs);

  const vector1 = await embeddings.embedQuery(allSplits[0].pageContent);
  const vector2 = await embeddings.embedQuery(allSplits[1].pageContent);

  const vectorStore = new MemoryVectorStore(embeddings);
  await vectorStore.addDocuments(allSplits);

  const search = await vectorStore.similaritySearch('又剩下他孤苦伶仃一人');
  console.log('search', search);
}

(async () => {
  try {
    await embed('Hello world');
    console.log('程序执行完毕');
  } catch (error) {
    console.error('主程序错误:', error);
  }
})();
