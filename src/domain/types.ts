export type GraphTriple = {
  node: string;
  target_node: string;
  relationship: string;
};

export type GraphExtraction = {
  graph: GraphTriple[];
};

export type NodesMap = Record<string, string>; // name -> uuid
export type Relationship = { source: string; target: string; type: string };


