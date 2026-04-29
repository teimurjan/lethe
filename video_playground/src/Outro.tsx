import {AbsoluteFill, interpolate, spring, useVideoConfig} from "remotion";
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
  const {fps} = useVideoConfig();
  const alpha = interpolate(
    frame,
    [0, 12, durationInFrames - 10, durationInFrames],
    [0, 1, 1, 0.85],
    {extrapolateLeft: "clamp", extrapolateRight: "clamp"},
  );

  // Hold the framing for a moment before the big number lands, so the
  // graph's final state gets to breathe.
  const pctScale = spring({
    frame: Math.max(0, frame - 30),
    fps,
    config: {damping: 14, stiffness: 90},
  });
  const numberAlpha = interpolate(pctScale, [0, 1], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const {baselineNdcg, lethNdcg, deltaPct, baselineLabel} = run.meta.headline!;

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
        gap: 32,
      }}
    >
      <div
        style={{
          fontSize: 14,
          color: TEXT_DIM,
          letterSpacing: 1,
          textTransform: "uppercase",
        }}
      >
        LongMemEval S · NDCG@10
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 20,
          transform: `scale(${0.9 + pctScale * 0.1})`,
          opacity: numberAlpha,
        }}
      >
        <span
          style={{
            fontSize: 168,
            fontWeight: 600,
            color: LETHE,
            lineHeight: 1,
            letterSpacing: -4,
          }}
        >
          +{Math.round(deltaPct)}%
        </span>
        <span style={{fontSize: 24, color: TEXT_DIM}}>NDCG</span>
      </div>

      <div
        style={{
          fontSize: 22,
          color: TEXT,
          textAlign: "center",
        }}
      >
        lethe vs{" "}
        <span style={{color: BASELINE}}>{baselineLabel}</span>
      </div>

      <div
        style={{
          display: "flex",
          gap: 64,
          fontFamily: FONT_MONO,
          fontVariantNumeric: "tabular-nums",
          marginTop: 6,
        }}
      >
        <Stat label="basic RRF" value={baselineNdcg} color={BASELINE} />
        <Stat label="lethe" value={lethNdcg} color={LETHE} />
      </div>

      <div
        style={{
          fontSize: 13,
          color: TEXT_FAINT,
          textAlign: "center",
          maxWidth: 760,
          lineHeight: 1.5,
          marginTop: 10,
        }}
      >
        hybrid BM25 + dense fusion vs. lethe (BM25 + dense + cross-encoder
        rerank + clustered RIF). numbers published in CLAUDE.md.
      </div>
    </AbsoluteFill>
  );
};

const Stat: React.FC<{label: string; value: number; color: string}> = ({
  label,
  value,
  color,
}) => (
  <div style={{display: "flex", flexDirection: "column", gap: 6}}>
    <span style={{fontSize: 13, color: TEXT_DIM}}>{label}</span>
    <span style={{fontSize: 28, color}}>{value.toFixed(4)}</span>
  </div>
);
