/**
 * Build a GraphRAG agent with Neo4j and Qdrant
 * This idea is from Qdrant documentation for Python
 * https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/#build-a-graphrag-agent-with-neo4j-and-qdrant
 *
 * This is the implementation for Node.js
 */
import { ensureCollection, upsertEmbeddings } from './infrastructure/vector/qdrant';
import { getNeo4jDriver, ingestToNeo4j } from './infrastructure/graph/neo4j';
import { extractGraphComponents } from './application/extract';
import { createEmbeddings } from './infrastructure/llm/openai';
import { retrieverSearch } from './application/retriever';
import { formatGraphContext, graphRAGRun } from './application/graphrag';

/**
 * This is the main entry point for the application.
 * It will extract the graph components from the raw text,
 * ingest them to Neo4j,
 * create embeddings for the graph components,
 * upsert them to Qdrant,
 * and then run the GraphRAG algorithm to answer the query.
 */
async function main() {
  console.log('Script started');
  const collectionName = 'graphRAGstoreds'; // This is the name of the collection in Qdrant
  const vectorDimension = 1536; // This is the dimension of the embeddings

  // Ensure collection exists in Qdrant
  console.log('Creating/ensuring collection...');
  await ensureCollection(collectionName, vectorDimension);

  // Example raw text to extract graph components from
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
  // Extract graph components from the raw text
  const { nodes, relationships } = await extractGraphComponents(raw);
  console.log('Nodes:', nodes);
  console.log('Relationships:', relationships);

  console.log('Ingesting to Neo4j...');
  // Ingest the graph components to Neo4j
  const nodeIdMapping = await ingestToNeo4j(nodes, relationships);
  console.log('Neo4j ingestion complete');

  /**
   * This is the embedding creation and upsertion to Qdrant.
   * It will create embeddings for the graph components,
   * and then upsert them to Qdrant.
   */
  console.log('Creating embeddings for Qdrant...');
  // Create embeddings for the graph components
  const paragraphs = raw.split('\n');
  const vectors = await createEmbeddings(paragraphs);
  const ids = Object.values(nodeIdMapping).slice(0, vectors.length);
  console.log('Upserting into Qdrant...');
  // Upsert the embeddings to Qdrant
  await upsertEmbeddings(collectionName, ids, vectors);
  console.log('Qdrant ingestion complete');

  /**
   * This is the GraphRAG algorithm.
   * It will use the retriever to find the most relevant graph components,
   * format the graph context for the GraphRAG algorithm,
   * and then run the GraphRAG algorithm to answer the query.
   */
  const query = 'How is Bob connected to New York?'; // This is the query to ask the graph

  /**
   * This is the retriever search.
   * It will use the retriever to find the most relevant graph components,
   * and then format the graph context for the GraphRAG algorithm.
   */
  console.log('Starting retriever search...');
  const ret = await retrieverSearch(collectionName, query, 5);
  console.log('Retriever ids:', ret.ids);

  // Format the graph context for the GraphRAG algorithm
  console.log('Formatting graph context...');
  const graphContext = formatGraphContext(ret.subgraph);
  console.log('Graph context:', graphContext);

  /**
   * This is the GraphRAG algorithm.
   * It will use the retriever to find the most relevant graph components,
   * format the graph context for the GraphRAG algorithm,
   * and then run the GraphRAG algorithm to answer the query.
   */
  console.log('Running GraphRAG...');
  const answer = await graphRAGRun(graphContext, query);
  console.log('Final Answer:', answer); // This is the final answer

  // Clean up Neo4j driver on exit
  await getNeo4jDriver().close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
