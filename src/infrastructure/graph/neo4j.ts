import neo4j, { Driver } from 'neo4j-driver';
import { ENV } from '../../config/env';

let driverSingleton: Driver | null = null;

export function getNeo4jDriver(): Driver {
  if (!driverSingleton) {
    driverSingleton = neo4j.driver(
      ENV.NEO4J_URI,
      neo4j.auth.basic(ENV.NEO4J_USERNAME, ENV.NEO4J_PASSWORD)
    );
  }
  return driverSingleton;
}

type NodesMap = Record<string, string>; // name -> id
type Relationship = { source: string; target: string; type: string };

export async function ingestToNeo4j(nodes: NodesMap, relationships: Relationship[]): Promise<NodesMap> {
  const driver = getNeo4jDriver();
  const session = driver.session();
  try {
    for (const [name, id] of Object.entries(nodes)) {
      await session.run(
        'CREATE (n:Entity {id: $id, name: $name})',
        { id, name }
      );
    }
    for (const rel of relationships) {
      await session.run(
        'MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)',
        { source_id: rel.source, target_id: rel.target, type: rel.type }
      );
    }
    return nodes;
  } finally {
    await session.close();
  }
}

export async function fetchRelatedGraph(entityIds: string[]) {
  const driver = getNeo4jDriver();
  const session = driver.session();
  const query = `
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
  `;
  try {
    const result = await session.run(query, { entity_ids: entityIds });
    const subgraph: Array<{ entity: any; relationship: any; related_node: any }> = [];
    for (const record of result.records) {
      subgraph.push({
        entity: record.get('e').properties,
        relationship: record.get('r').properties,
        related_node: record.get('related').properties,
      });
      const r2 = record.get('r2');
      const n2 = record.get('n2');
      if (r2 && n2) {
        subgraph.push({
          entity: record.get('related').properties,
          relationship: r2.properties,
          related_node: n2.properties,
        });
      }
    }
    return subgraph;
  } finally {
    await session.close();
  }
}


