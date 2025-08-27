import { createEmbeddings } from '../infrastructure/llm/openai';
import { fetchRelatedGraph } from '../infrastructure/graph/neo4j';
import { searchTopK } from '../infrastructure/vector/qdrant';

export async function retrieverSearch(collectionName: string, query: string, topK = 5) {
  const [vector] = await createEmbeddings([query]);
  const res = await searchTopK(collectionName, vector, topK);
  const ids = res.map((r: any) => r.payload?.id).filter(Boolean) as string[];
  const subgraph = await fetchRelatedGraph(ids);
  return { ids, subgraph };
}


