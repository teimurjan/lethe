import {AbsoluteFill, interpolate} from "remotion";
import {
  BASELINE,
  BG,
  FONT_MONO,
  LETHE,
  TEXT,
  TEXT_DIM,
  TEXT_FAINT,
} from "./theme";
import type {RunData} from "./types";

type Props = {
  frame: number;
  durationInFrames: number;
  run: RunData;
};

export const Outro: React.FC<Props> = ({frame, durationInFrames, run}) => {
  const alpha = interpolate(
    frame,
    [0, 12, durationInFrames - 10, durationInFrames],
    [0, 1, 1, 0.85],
    {extrapolateLeft: "clamp", extrapolateRight: "clamp"},
  );

  const n = run.queries.length;
  const bMean =
    run.queries.reduce((a, q) => a + q.baseline.ndcg, 0) / n;
  const lMean = run.queries.reduce((a, q) => a + q.lethe.ndcg, 0) / n;
  const delta = lMean - bMean;
  const pct = bMean > 0 ? (delta / bMean) * 100 : 0;

  return (
    <AbsoluteFill
      style={{
        background: BG,
        color: TEXT,
        fontFamily: FONT_MONO,
        padding: 60,
        opacity: alpha,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 28,
      }}
    >
      <div style={{textAlign: "center"}}>
        <div style={{fontSize: 14, color: TEXT_DIM}}>
          LongMemEval · 199,509 turns · {n} queries · NDCG@10
        </div>
        <div style={{fontSize: 30, marginTop: 10}}>
          lethe vs basic hybrid retrieval.
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 80,
          fontFamily: FONT_MONO,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        <Stat label="basic RRF" value={bMean} color={BASELINE} />
        <Stat label="lethe" value={lMean} color={LETHE} />
        <Stat
          label="delta"
          value={delta}
          color={delta >= 0 ? LETHE : BASELINE}
          suffix={` (${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%)`}
          sign
        />
      </div>

      <div
        style={{
          fontSize: 13,
          color: TEXT_FAINT,
          textAlign: "center",
          maxWidth: 760,
          lineHeight: 1.5,
          marginTop: 14,
        }}
      >
        hybrid BM25 + dense + cross-encoder on both sides. only lethe
        accumulates retrieval-induced forgetting across the run. full
        methodology in BENCHMARKS.md.
      </div>
    </AbsoluteFill>
  );
};

const Stat: React.FC<{
  label: string;
  value: number;
  color: string;
  suffix?: string;
  sign?: boolean;
}> = ({label, value, color, suffix = "", sign = false}) => (
  <div style={{display: "flex", flexDirection: "column", gap: 6}}>
    <span style={{fontSize: 13, color: TEXT_DIM}}>{label}</span>
    <span style={{fontSize: 26, color}}>
      {sign ? (value >= 0 ? "+" : "") : ""}
      {value.toFixed(4)}
      {suffix ? <span style={{fontSize: 16, marginLeft: 6}}>{suffix}</span> : null}
    </span>
  </div>
);
