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
  phase?: string;
  synthetic?: boolean;
  baseline: SidePayload;
  lethe: SidePayload;
};

export type HeadlineMeta = {
  baselineNdcg: number;
  lethNdcg: number;
  deltaPct: number;
  text: string;
  baselineLabel: string;
};

export type RunData = {
  meta: {
    fps: number;
    totalQueries: number;
    snapshotAt: number[];
    nUnique?: number;
    nReplay?: number;
    nRounds?: number;
    nRoundsReal?: number;
    nRoundsSynthetic?: number;
    phaseBoundaries?: number[];
    headline?: HeadlineMeta;
  };
  queries: QueryRow[];
};

export type Side = "baseline" | "lethe";
