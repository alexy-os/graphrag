import { chatClient } from '../infrastructure/llm/openai';

export function formatGraphContext(subgraph: Array<{ entity: any; relationship: any; related_node: any }>) {
  const nodes = new Set<string>();
  const edges: string[] = [];
  for (const entry of subgraph) {
    const entity = entry.entity;
    const related = entry.related_node;
    const relationship = entry.relationship;
    if (!entity || !related || !relationship) continue;
    nodes.add(entity.name);
    nodes.add(related.name);
    edges.push(`${entity.name} ${relationship.type} ${related.name}`);
  }
  return { nodes: Array.from(nodes), edges };
}

export async function graphRAGRun(graphContext: { nodes: string[]; edges: string[] }, userQuery: string) {
  const nodesStr = graphContext.nodes.join(', ');
  const edgesStr = graphContext.edges.join('; ');
  const prompt = `
You are an intelligent assistant with access to the following knowledge graph:

Nodes: ${nodesStr}

Edges: ${edgesStr}

Using this graph, Answer the following question:

User Query: "${userQuery}"
  `;
  const completion = await chatClient.chat.completions.create({
    model: 'gpt-4',
    messages: [
      { role: 'system', content: 'Provide the answer for the following question:' },
      { role: 'user', content: prompt },
    ],
  });
  return completion.choices[0]?.message?.content ?? '';
}


