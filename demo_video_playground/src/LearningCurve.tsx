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
import type {QueryRow, RunData} from "./types";

type Props = {
  frame: number;
  durationInFrames: number;
  run: RunData;
};

const W = 1080;
const H = 440;
const PAD_X = 92;
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

  const rounds = collectRounds(run.queries);
  const nRounds = rounds.length;
  const nRoundsReal = run.meta.nRoundsReal!;

  // Auto-fit y-range around the actual series so the curve fills the plot.
  const allY = rounds.flatMap((r) => [r.baseline, r.lethe]);
  const yMin = Math.max(0, Math.min(...allY) - 0.02);
  const yMax = Math.min(1, Math.max(...allY) + 0.02);

  const plotW = W - PAD_X * 2;
  const plotH = H - PAD_Y * 2;

  const xOf = (i: number) =>
    PAD_X + (i / Math.max(1, nRounds - 1)) * plotW;
  const yOf = (v: number) => {
    const norm = (v - yMin) / (yMax - yMin);
    return PAD_Y + (1 - Math.max(0, Math.min(1, norm))) * plotH;
  };

  // Draw the curve over the first ~2/3 of the scene; the remaining
  // frames hold the completed curve so the viewer can read it.
  const drawFrames = Math.max(1, Math.floor(durationInFrames * 0.65));
  const progress = interpolate(frame, [0, drawFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // Reveal one round at a time, smoothly.
  const revealed = progress * (nRounds - 1);
  const visibleFull = Math.floor(revealed);
  const fracToNext = revealed - visibleFull;

  const baselinePts: Array<readonly [number, number]> = [];
  const lethePts: Array<readonly [number, number]> = [];
  for (let i = 0; i <= Math.min(visibleFull, nRounds - 1); i++) {
    baselinePts.push([xOf(i), yOf(rounds[i].baseline)] as const);
    lethePts.push([xOf(i), yOf(rounds[i].lethe)] as const);
  }
  // Interpolate toward the next point so the line grows smoothly.
  if (visibleFull + 1 < nRounds) {
    const a = rounds[visibleFull];
    const b = rounds[visibleFull + 1];
    const x = xOf(visibleFull) + (xOf(visibleFull + 1) - xOf(visibleFull)) * fracToNext;
    const yb = yOf(a.baseline + (b.baseline - a.baseline) * fracToNext);
    const yl = yOf(a.lethe + (b.lethe - a.lethe) * fracToNext);
    baselinePts.push([x, yb] as const);
    lethePts.push([x, yl] as const);
  }

  const toPath = (ps: ReadonlyArray<readonly [number, number]>) => {
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

  // Shaded gap between the two lines.
  const gapArea =
    lethePts.length > 0
      ? toPath(lethePts) +
        " " +
        baselinePts
          .slice()
          .reverse()
          .map((p) => `L ${p[0]} ${p[1]}`)
          .join(" ") +
        " Z"
      : "";

  const currentIdx = Math.min(nRounds - 1, Math.round(revealed));
  const current = rounds[currentIdx];
  const currentDelta = current.lethe - current.baseline;
  const pct = (currentDelta / current.baseline) * 100;

  // Vertical divider between real and synthetic rounds (fades in).
  const dividerIdx = nRoundsReal;
  const dividerX =
    dividerIdx > 0 && dividerIdx < nRounds
      ? PAD_X + ((dividerIdx - 0.5) / Math.max(1, nRounds - 1)) * plotW
      : null;
  const dividerAlpha = interpolate(revealed, [dividerIdx - 1, dividerIdx], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Y-axis tick values — 4 evenly spaced.
  const yTicks = [0, 1, 2, 3].map((k) => yMin + ((yMax - yMin) * k) / 3);

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
            mean NDCG@10 per replay round · {run.meta.nUnique} queries replayed
          </span>
        </div>
        <div
          style={{
            color: TEXT_DIM,
            fontSize: 13,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          round {currentIdx + 1}
          <span style={{color: TEXT_FAINT}}> / {nRounds}</span>
          {currentIdx >= nRoundsReal ? (
            <span style={{color: TEXT_FAINT, marginLeft: 10}}>· projected</span>
          ) : null}
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
          {yTicks.map((v, k) => {
            const y = yOf(v);
            return (
              <g key={k}>
                <line
                  x1={PAD_X}
                  x2={W - PAD_X}
                  y1={y}
                  y2={y}
                  stroke={BORDER}
                  strokeDasharray="1 4"
                  strokeWidth={1}
                />
                <text
                  x={PAD_X - 12}
                  y={y + 4}
                  fill={TEXT_FAINT}
                  fontSize={12}
                  fontFamily={FONT_MONO}
                  textAnchor="end"
                >
                  {v.toFixed(3)}
                </text>
              </g>
            );
          })}
          {rounds.map((_, i) => {
            const px = xOf(i);
            return (
              <g key={i}>
                <line
                  x1={px}
                  x2={px}
                  y1={PAD_Y}
                  y2={PAD_Y + plotH}
                  stroke={BORDER}
                  strokeDasharray="1 6"
                  strokeWidth={1}
                  opacity={0.5}
                />
                <text
                  x={px}
                  y={H - 12}
                  fill={TEXT_FAINT}
                  fontSize={12}
                  fontFamily={FONT_MONO}
                  textAnchor="middle"
                >
                  {`r${i + 1}`}
                </text>
              </g>
            );
          })}

          {/* divider between real and synthetic rounds */}
          {dividerX !== null ? (
            <g opacity={dividerAlpha}>
              <line
                x1={dividerX}
                x2={dividerX}
                y1={PAD_Y}
                y2={PAD_Y + plotH}
                stroke={TEXT_DIM}
                strokeDasharray="4 4"
                strokeWidth={1}
              />
              <text
                x={dividerX + 6}
                y={PAD_Y + 14}
                fill={TEXT_DIM}
                fontSize={11}
                fontFamily={FONT_MONO}
              >
                projected →
              </text>
            </g>
          ) : null}

          {/* gap fill */}
          <path d={gapArea} fill={LETHE} opacity={0.12} />

          {/* baseline line */}
          <path
            d={toPath(baselinePts)}
            fill="none"
            stroke={BASELINE}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {baselinePts.length > 0 && (
            <circle
              cx={baselinePts[baselinePts.length - 1][0]}
              cy={baselinePts[baselinePts.length - 1][1]}
              r={3.5}
              fill={BASELINE}
            />
          )}

          {/* lethe line */}
          <path
            d={toPath(lethePts)}
            fill="none"
            stroke={LETHE}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {lethePts.length > 0 && (
            <circle
              cx={lethePts[lethePts.length - 1][0]}
              cy={lethePts[lethePts.length - 1][1]}
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
            mean NDCG@10 per round (same 100 queries replayed)
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
        <Stat color={BASELINE} label="basic RRF" value={current.baseline} />
        <Stat color={LETHE} label="lethe" value={current.lethe} />
        <div style={{color: TEXT_DIM}}>
          delta{" "}
          <span style={{color: LETHE, marginLeft: 8}}>
            {currentDelta >= 0 ? "+" : ""}
            {currentDelta.toFixed(4)}
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

type RoundMean = {phase: string; baseline: number; lethe: number};

const collectRounds = (rows: QueryRow[]): RoundMean[] => {
  const sums = new Map<string, {b: number; l: number; n: number}>();
  for (const r of rows) {
    const phase = r.phase;
    if (!phase || !phase.startsWith("warm")) continue;
    const e = sums.get(phase) ?? {b: 0, l: 0, n: 0};
    e.b += r.baseline.ndcg;
    e.l += r.lethe.ndcg;
    e.n += 1;
    sums.set(phase, e);
  }
  const sorted = [...sums.keys()].sort((a, b) => {
    const na = parseInt(a.replace("warm", ""), 10);
    const nb = parseInt(b.replace("warm", ""), 10);
    return na - nb;
  });
  return sorted.map((phase) => {
    const e = sums.get(phase)!;
    return {phase, baseline: e.b / e.n, lethe: e.l / e.n};
  });
};
