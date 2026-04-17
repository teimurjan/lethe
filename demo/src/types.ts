export type HitColor = "green" | "red" | "gray";

export type Hit = {
  id: string;
  snippet: string;
  score: number;
  color: HitColor;
};

export type SidePayload = {
  ndcg: number;
  top5?: Hit[];
  queryText?: string;
};

export type QueryRow = {
  idx: number;
  qid: string;
  baseline: SidePayload;
  lethe: SidePayload;
};

export type RunData = {
  meta: {
    fps: number;
    totalQueries: number;
    snapshotAt: number[];
  };
  queries: QueryRow[];
};

export type Side = "baseline" | "lethe";
