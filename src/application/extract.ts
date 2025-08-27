import { z } from 'zod';
import { chatClient } from '../infrastructure/llm/openai';
import { GraphExtraction, NodesMap, Relationship } from '../domain/types';

const GraphSchema = z.object({
  graph: z.array(
    z.object({
      node: z.string(),
      target_node: z.string(),
      relationship: z.string(),
    })
  ),
});

export async function extractGraphComponents(raw: string): Promise<{ nodes: NodesMap; relationships: Relationship[] }> {
  const system = `You are a precise graph relationship extractor. Extract all relationships from the text and format them as a JSON object with this exact structure:\n{
    "graph": [
      {"node": "Person/Entity", "target_node": "Related Entity", "relationship": "Type of Relationship"}
    ]
  }
  Include ALL relationships mentioned in the text, including implicit ones. Be thorough and precise.`;

  const completion = await chatClient.chat.completions.create({
    model: 'gpt-4o-2024-08-06',
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: `Extract nodes and relationships from the following text:\n${raw}` },
    ],
  });

  const content = completion.choices[0]?.message?.content ?? '{}';
  const parsed = GraphSchema.parse(JSON.parse(content)) as GraphExtraction;

  const nodes: NodesMap = {};
  const relationships: Relationship[] = [];
  const uuid = () => crypto.randomUUID();

  for (const entry of parsed.graph) {
    const node = entry.node;
    const target = entry.target_node;
    const rel = entry.relationship;
    if (!nodes[node]) nodes[node] = uuid();
    if (target && !nodes[target]) nodes[target] = uuid();
    if (target && rel) {
      relationships.push({ source: nodes[node], target: nodes[target], type: rel });
    }
  }

  return { nodes, relationships };
}


