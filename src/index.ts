import { ensureCollection, getQdrantClient, upsertEmbeddings } from './infrastructure/vector/qdrant';
import { getNeo4jDriver, ingestToNeo4j } from './infrastructure/graph/neo4j';
import { extractGraphComponents } from './application/extract';
import { createEmbeddings } from './infrastructure/llm/openai';
import { retrieverSearch } from './application/retriever';
import { formatGraphContext, graphRAGRun } from './application/graphrag';

async function main() {
  console.log('Script started');
  const collectionName = 'graphRAGstoreds';
  const vectorDimension = 1536;

  // Ensure collection exists
  console.log('Creating/ensuring collection...');
  await ensureCollection(collectionName, vectorDimension);

  // Example raw text (same as Python)
  const raw = `Alice is a data scientist at TechCorp's Seattle office.
Bob and Carol collaborate on the Alpha project.
Carol transferred to the New York office last year.
Dave mentors both Alice and Bob.
TechCorp's headquarters is in Seattle.
Carol leads the East Coast team.
Dave started his career in Seattle.
The Alpha project is managed from New York.
Alice previously worked with Carol at DataCo.
Bob joined the team after Dave's recommendation.
Eve runs the West Coast operations from Seattle.
Frank works with Carol on client relations.
The New York office expanded under Carol's leadership.
Dave's team spans multiple locations.
Alice visits Seattle monthly for team meetings.
Bob's expertise is crucial for the Alpha project.
Carol implemented new processes in New York.
Eve and Dave collaborated on previous projects.
Frank reports to the New York office.
TechCorp's main AI research is in Seattle.
The Alpha project revolutionized East Coast operations.
Dave oversees projects in both offices.
Bob's contributions are mainly remote.
Carol's team grew significantly after moving to New York.
Seattle remains the technology hub for TechCorp.`;

  console.log('Extracting graph components...');
  const { nodes, relationships } = await extractGraphComponents(raw);
  console.log('Nodes:', nodes);
  console.log('Relationships:', relationships);

  console.log('Ingesting to Neo4j...');
  const nodeIdMapping = await ingestToNeo4j(nodes, relationships);
  console.log('Neo4j ingestion complete');

  console.log('Creating embeddings for Qdrant...');
  const paragraphs = raw.split('\n');
  const vectors = await createEmbeddings(paragraphs);
  const ids = Object.values(nodeIdMapping).slice(0, vectors.length);
  console.log('Upserting into Qdrant...');
  await upsertEmbeddings(collectionName, ids, vectors);
  console.log('Qdrant ingestion complete');

  const query = 'How is Bob connected to New York?';
  console.log('Starting retriever search...');
  const ret = await retrieverSearch(collectionName, query, 5);
  console.log('Retriever ids:', ret.ids);

  console.log('Formatting graph context...');
  const graphContext = formatGraphContext(ret.subgraph);
  console.log('Graph context:', graphContext);

  console.log('Running GraphRAG...');
  const answer = await graphRAGRun(graphContext, query);
  console.log('Final Answer:', answer);

  // Clean up Neo4j driver on exit
  await getNeo4jDriver().close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});


