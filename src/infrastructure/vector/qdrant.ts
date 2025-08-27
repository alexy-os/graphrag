import { QdrantClient } from '@qdrant/js-client-rest';
import { ENV } from '../../config/env';

let qdrantSingleton: QdrantClient | null = null;

export function getQdrantClient(): QdrantClient {
  if (!qdrantSingleton) {
    qdrantSingleton = new QdrantClient({ url: ENV.QDRANT_URL, apiKey: ENV.QDRANT_KEY });
  }
  return qdrantSingleton;
}

export async function ensureCollection(collectionName: string, vectorSize: number): Promise<void> {
  const client = getQdrantClient();
  try {
    await client.getCollection(collectionName);
    console.log(`Skipping creating collection; '${collectionName}' already exists.`);
  } catch (e: any) {
    const msg = String(e?.message || e);
    if (msg.includes('Not found')) {
      console.log(`Collection '${collectionName}' not found. Creating it now...`);
      await client.createCollection(collectionName, {
        vectors: { size: vectorSize, distance: 'Cosine' },
      } as any);
      console.log(`Collection '${collectionName}' created successfully.`);
    } else {
      console.error('Error while checking collection:', e);
      throw e;
    }
  }
}

export async function upsertEmbeddings(collectionName: string, ids: string[], vectors: number[][]): Promise<void> {
  const client = getQdrantClient();
  const points = ids.map((id, i) => ({ id, vector: vectors[i], payload: { id } }));
  await client.upsert(collectionName, { points } as any);
}

export async function searchTopK(collectionName: string, queryVector: number[], topK: number) {
  const client = getQdrantClient();
  const res = await client.search(collectionName, {
    vector: queryVector,
    limit: topK,
    with_payload: true,
  } as any);
  return res;
}


