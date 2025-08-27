# GraphRAG (Bun + TypeScript)

This project mirrors the Python functionality using a clean architecture in Bun/TypeScript.

## Structure
- src/config/env.ts: Environment loading and validation
- src/infrastructure/llm/openai.ts: Chat and embedding clients
- src/infrastructure/graph/neo4j.ts: Neo4j driver and graph I/O
- src/infrastructure/vector/qdrant.ts: Qdrant client, collection ensure, upsert, search
- src/application/extract.ts: LLM JSON extraction → nodes/relationships
- src/application/retriever.ts: Vector search → Neo4j subgraph
- src/application/graphrag.ts: Context formatting and answer generation
- src/index.ts: CLI entry mimicking Python flow

## Requirements
- Bun 1.2+
- Neo4j Aura (or any Neo4j instance with TLS)
- Qdrant (Cloud or self-hosted)
- OpenRouter (for chat) and an embeddings provider (OpenAI or compatible)

## Environment
Create a .env in the project root:

```
# Neo4j
NEO4J_URI=neo4j+s://qdrant.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Qdrant
QDRANT_URL=https://your-qdrant-url
QDRANT_KEY=your_qdrant_key

# Chat LLM (OpenRouter)
OPENROUTER_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_openrouter_key

# Embeddings LLM
EMBEDDING_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your_embedding_key
```

Notes:
- The embeddings model is text-embedding-3-small (dimension 1536) to match Qdrant collection size.
- NEO4J_URI supports neo4j+s:// for TLS.

## Install & Run
```
bun install
bun run typecheck
bun start
```

The script will:
1) Ensure Qdrant collection exists (1536, Cosine)
2) Extract a graph via LLM (JSON)
3) Ingest nodes/relationships into Neo4j
4) Embed paragraphs and upsert into Qdrant
5) Retrieve top-K related entities and subgraph from Neo4j
6) Generate a final answer with GraphRAG

## Troubleshooting
- Ensure all envs are set and valid. The app will fail fast if any are missing/invalid.
- Confirm your Qdrant URL/API key and collection dimensions align with the embedding size (1536).
- In Neo4j Browser, you can verify:
```
MATCH (n:Entity) RETURN count(n);
MATCH (:Entity)-[r]-() RETURN count(r);
MATCH (n:Entity) RETURN n.id, n.name LIMIT 25;
```
