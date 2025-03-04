// import { Document } from '@langchain/core/documents';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import dotenv from 'dotenv';
import path from 'path';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';

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
  apiKey: process.env.SILICONFLOW_API_KEY,
  apiBase: 'https://api.siliconflow.cn/v1',
  basePath: '',
  model: 'Pro/BAAI/bge-m3',
});

const assetsPath = path.join(process.cwd(), 'assets');
const pdfPath = path.join(assetsPath, '00.pdf');
const loader = new PDFLoader(pdfPath);

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(docs);

try {
  const res = await embeddings.embedQuery('Hello world');
  console.log(res);
} catch (error) {
  console.log(error);
}
const vector1 = await embeddings.embedQuery(allSplits[0].pageContent);
const vector2 = await embeddings.embedQuery(allSplits[1].pageContent);

console.assert(vector1.length === vector2.length);
console.log(`Generated vectors of length ${vector1.length}\n`);
console.log(vector1.slice(0, 10));

// https://blog.csdn.net/u012899618/article/details/145620482
