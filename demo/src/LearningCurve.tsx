import {AbsoluteFill, interpolate} from "remotion";
import {
  BASELINE,
  BG,
  BORDER,
  FONT_MONO,
  LETHE,
  SURFACE,
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

const W = 1080;
const H = 440;
const PAD_X = 72;
const PAD_Y = 48;

export const LearningCurve: React.FC<Props> = ({
  frame,
  durationInFrames,
  run,
}) => {
  const alpha = interpolate(frame, [0, 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Post-warmup running means. Skip the single-sample noise so the delta
  // trace starts near zero and grows as RIF state accumulates.
  const skip = 150;
  const total = run.meta.totalQueries;
  const baselineMean = cumulativeMean(
    run.queries.slice(skip).map((q) => q.baseline.ndcg),
  );
  const letheMean = cumulativeMean(
    run.queries.slice(skip).map((q) => q.lethe.ndcg),
  );
  const delta = letheMean.map((v, i) => v - baselineMean[i]);
  const n = delta.length;

  const yMin = -0.02;
  const yMax = 0.18;

  const plotW = W - PAD_X * 2;
  const plotH = H - PAD_Y * 2;

  const toPts = (vs: number[]) =>
    vs.map((y, i) => {
      const normY = (y - yMin) / (yMax - yMin);
      const clamped = Math.max(0, Math.min(1, normY));
      const x = PAD_X + (i / Math.max(1, n - 1)) * plotW;
      return [x, PAD_Y + (1 - clamped) * plotH] as const;
    });

  const pts = toPts(delta);

  const toPath = (ps: readonly (readonly [number, number])[]) => {
    if (ps.length === 0) return "";
    return (
      "M " +
      ps[0][0] +
      " " +
      ps[0][1] +
      " " +
      ps.slice(1).map((p) => `L ${p[0]} ${p[1]}`).join(" ")
    );
  };

  const progress = interpolate(frame, [0, durationInFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const visibleIdx = Math.min(n - 1, Math.floor(progress * n));
  const visible = pts.slice(0, visibleIdx + 1);
  const visibleDelta = delta[visibleIdx] ?? 0;
  const currentBaseline = baselineMean[visibleIdx] ?? 0;
  const currentLethe = letheMean[visibleIdx] ?? 0;
  const pct = currentBaseline > 0 ? (visibleDelta / currentBaseline) * 100 : 0;

  const zeroY =
    PAD_Y + (1 - (0 - yMin) / (yMax - yMin)) * plotH;

  // Positive-region fill below the delta line (zero-referenced area).
  const areaPath =
    visible.length > 0
      ? toPath(visible) +
        ` L ${visible[visible.length - 1][0]} ${zeroY}` +
        ` L ${visible[0][0]} ${zeroY} Z`
      : "";

  const xTicks = [500, 1500, 2500, 3500, 4500];
  const gridYs = [0.0, 0.05, 0.10, 0.15];

  return (
    <AbsoluteFill
      style={{
        background: BG,
        fontFamily: FONT_MONO,
        color: TEXT,
        opacity: alpha,
        padding: 40,
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
        }}
      >
        <div style={{display: "flex", gap: 14, alignItems: "baseline"}}>
          <span style={{color: TEXT_DIM, fontSize: 13}}>learning_curve</span>
          <span style={{color: TEXT, fontSize: 18}}>
            lethe − basic hybrid RRF · NDCG@10 · LongMemEval (199,509 turns)
          </span>
        </div>
        <div
          style={{
            color: TEXT_DIM,
            fontSize: 13,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          query {String(visibleIdx + 1 + skip).padStart(4, "0")}
          <span style={{color: TEXT_FAINT}}> / {total}</span>
        </div>
      </div>

      <div
        style={{
          flex: 1,
          background: SURFACE,
          border: `1px solid ${BORDER}`,
          padding: 16,
        }}
      >
        <svg viewBox={`0 0 ${W} ${H}`} style={{width: "100%", height: "100%"}}>
          {gridYs.map((g) => {
            const y = PAD_Y + (1 - (g - yMin) / (yMax - yMin)) * plotH;
            const isZero = g === 0;
            return (
              <g key={g}>
                <line
                  x1={PAD_X}
                  x2={W - PAD_X}
                  y1={y}
                  y2={y}
                  stroke={isZero ? TEXT_DIM : BORDER}
                  strokeDasharray={isZero ? "" : "1 4"}
                  strokeWidth={isZero ? 1.5 : 1}
                />
                <text
                  x={14}
                  y={y + 4}
                  fill={isZero ? TEXT_DIM : TEXT_FAINT}
                  fontSize={12}
                  fontFamily={FONT_MONO}
                >
                  {(g >= 0 ? "+" : "") + g.toFixed(2)}
                </text>
              </g>
            );
          })}
          {xTicks.map((x) => {
            const idx = x - skip;
            if (idx < 0 || idx > n - 1) return null;
            const px = PAD_X + (idx / Math.max(1, n - 1)) * plotW;
            return (
              <g key={x}>
                <line
                  x1={px}
                  x2={px}
                  y1={PAD_Y}
                  y2={PAD_Y + plotH}
                  stroke={BORDER}
                  strokeDasharray="1 6"
                  strokeWidth={1}
                />
                <text
                  x={px}
                  y={H - 12}
                  fill={TEXT_FAINT}
                  fontSize={12}
                  fontFamily={FONT_MONO}
                  textAnchor="middle"
                >
                  q{x}
                </text>
              </g>
            );
          })}

          {/* shaded area under the delta curve */}
          <path d={areaPath} fill={LETHE} opacity={0.14} />
          {/* delta line */}
          <path
            d={toPath(visible)}
            fill="none"
            stroke={LETHE}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {visible.length > 0 && (
            <circle
              cx={visible[visible.length - 1][0]}
              cy={visible[visible.length - 1][1]}
              r={4}
              fill={LETHE}
            />
          )}

          <text
            x={PAD_X}
            y={PAD_Y - 16}
            fill={TEXT_DIM}
            fontSize={12}
            fontFamily={FONT_MONO}
          >
            Δ NDCG@10 (lethe − baseline, running mean)
          </text>
        </svg>
      </div>

      <div
        style={{
          display: "flex",
          gap: 56,
          alignItems: "center",
          fontSize: 22,
          fontVariantNumeric: "tabular-nums",
          padding: "4px 12px",
        }}
      >
        <Stat color={BASELINE} label="basic RRF" value={currentBaseline} />
        <Stat color={LETHE} label="lethe" value={currentLethe} />
        <div style={{color: TEXT_DIM}}>
          delta{" "}
          <span style={{color: LETHE, marginLeft: 8}}>
            {visibleDelta >= 0 ? "+" : ""}
            {visibleDelta.toFixed(4)}
          </span>
          <span
            style={{
              color: LETHE,
              marginLeft: 10,
              fontSize: 18,
            }}
          >
            ({pct >= 0 ? "+" : ""}
            {pct.toFixed(1)}%)
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Stat: React.FC<{color: string; label: string; value: number}> = ({
  color,
  label,
  value,
}) => (
  <div style={{display: "flex", alignItems: "center", gap: 14}}>
    <span style={{width: 22, height: 4, background: color}} />
    <span style={{color: TEXT_DIM}}>
      {label}
      <span style={{color: TEXT, marginLeft: 10}}>{value.toFixed(4)}</span>
    </span>
  </div>
);

const cumulativeMean = (xs: number[]): number[] => {
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < xs.length; i++) {
    sum += xs[i];
    out.push(sum / (i + 1));
  }
  return out;
};
