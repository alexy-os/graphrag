import OpenAI from 'openai';
import { ENV } from '../../config/env';

export const chatClient = new OpenAI({
  baseURL: ENV.OPENROUTER_URL,
  apiKey: ENV.OPENROUTER_API_KEY,
});

export const embeddingClient = new OpenAI({
  baseURL: ENV.EMBEDDING_URL,
  apiKey: ENV.EMBEDDING_API_KEY,
});

export async function createEmbeddings(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  // Batch by provider capabilities; here naive per-text loop for clarity.
  const vectors: number[][] = [];
  for (const t of texts) {
    const res = await embeddingClient.embeddings.create({
      model: 'text-embedding-3-small',
      input: t,
    });
    vectors.push(res.data[0].embedding as unknown as number[]);
  }
  return vectors;
}


