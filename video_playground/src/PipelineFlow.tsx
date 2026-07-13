import {AbsoluteFill, interpolate, spring, useVideoConfig} from "remotion";
import {
  BG,
  BORDER,
  COLOR_DISTRACTOR,
  COLOR_RELEVANT,
  FONT_MONO,
  LETHE,
  SURFACE,
  TEXT,
  TEXT_DIM,
  TEXT_FAINT,
} from "./theme";

type Props = {
  frame: number;
  durationInFrames: number;
};

const PHASES = [
  {key: "dedupe", title: "dedupe", start: 0, end: 90},
  {key: "query", title: "query", start: 90, end: 180},
  {key: "retrieve", title: "retrieve", start: 180, end: 300},
  {key: "merge", title: "merge", start: 300, end: 390},
  {key: "rerank", title: "rerank", start: 390, end: 480},
  {key: "learn", title: "learn", start: 480, end: 570},
] as const;

type PhaseKey = (typeof PHASES)[number]["key"];

type Doc = {
  id: string;
  label: string;
  kind: "good" | "bad";
};

const DOCS: Record<string, Doc> = {
  e1: {id: "e1", label: "PgBouncer transaction mode", kind: "good"},
  e2: {id: "e2", label: "asyncpg pool_size for FastAPI", kind: "good"},
  eD: {id: "eD", label: "404 — page not found", kind: "bad"},
};

const QUERY = "PostgreSQL connection pool config";

// Sparse: keyword overlap puts the distractor on top
const SPARSE = ["eD", "e1", "e2"];
// Dense: meaning matters, distractor falls
const DENSE = ["e1", "e2", "eD"];
// Merged (RRF) — the same candidates fuse by id; eD's sparse rank pulls it to #2
const MERGED = ["e1", "eD", "e2"];
// After cross-encoder, distractor crashes to the bottom
const RERANKED = ["e1", "e2", "eD"];

export const PipelineFlow: React.FC<Props> = ({frame, durationInFrames}) => {
  const {fps} = useVideoConfig();
  const fade = interpolate(frame, [0, 8], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const phase = currentPhase(frame);

  return (
    <AbsoluteFill
      style={{
        background: BG,
        fontFamily: FONT_MONO,
        color: TEXT,
        opacity: fade,
        padding: "60px 80px",
        display: "flex",
        flexDirection: "column",
        gap: 40,
      }}
    >
      <PhaseTitle frame={frame} fps={fps} />

      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: 0,
        }}
      >
        {phase === "dedupe" && <DedupeView frame={frame} fps={fps} />}
        {phase === "query" && <QueryView frame={frame} fps={fps} />}
        {phase === "retrieve" && <RetrieveView frame={frame} fps={fps} />}
        {phase === "merge" && <MergeView frame={frame} fps={fps} />}
        {phase === "rerank" && <RerankView frame={frame} fps={fps} />}
        {phase === "learn" && <LearnView frame={frame} fps={fps} />}
      </div>

      <PhaseDots phase={phase} />
    </AbsoluteFill>
  );
};

// ─────────────────────────────────────────────────────────────────────

const currentPhase = (frame: number): PhaseKey => {
  for (let i = PHASES.length - 1; i >= 0; i--) {
    if (frame >= PHASES[i].start) return PHASES[i].key;
  }
  return "dedupe";
};

const PhaseTitle: React.FC<{frame: number; fps: number}> = ({frame, fps}) => {
  const phase = currentPhase(frame);
  const idx = PHASES.findIndex((p) => p.key === phase);
  const title = PHASES[idx].title;
  const phaseStart = PHASES[idx].start;
  const s = spring({
    frame: Math.max(0, frame - phaseStart),
    fps,
    config: {damping: 200, stiffness: 220, mass: 0.6},
    durationInFrames: 14,
  });
  const opacity = interpolate(s, [0, 1], [0.4, 1]);
  return (
    <div style={{display: "flex", alignItems: "baseline", gap: 24}}>
      <span
        style={{
          color: TEXT_FAINT,
          fontSize: 28,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {`0${idx + 1}`.slice(-2)}
      </span>
      <span
        style={{
          color: LETHE,
          fontSize: 56,
          fontWeight: 500,
          letterSpacing: -1,
          opacity,
        }}
      >
        {title}
      </span>
    </div>
  );
};

const PhaseDots: React.FC<{phase: PhaseKey}> = ({phase}) => {
  const idx = PHASES.findIndex((p) => p.key === phase);
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        gap: 14,
      }}
    >
      {PHASES.map((p, i) => (
        <div
          key={p.key}
          style={{
            width: i === idx ? 32 : 10,
            height: 4,
            background: i === idx ? LETHE : i < idx ? `${LETHE}55` : BORDER,
            borderRadius: 2,
            transition: "none",
          }}
        />
      ))}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────

const DedupeView: React.FC<{frame: number; fps: number}> = ({frame, fps}) => {
  const local = frame - PHASES[0].start;
  const progress = interpolate(local, [12, 68], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const duplicateOpacity = interpolate(progress, [0.25, 0.7], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const canonicalX = interpolate(progress, [0, 1], [-170, 170]);
  const duplicateX = interpolate(progress, [0, 1], [170, -170]);
  const label = spring({
    frame: Math.max(0, local - 48),
    fps,
    config: {damping: 200, stiffness: 220, mass: 0.6},
    durationInFrames: 16,
  });

  return (
    <div style={{display: "flex", flexDirection: "column", alignItems: "center", gap: 30}}>
      <div style={{position: "relative", width: 900, height: 170}}>
        <div style={{position: "absolute", left: 0, width: 560, transform: `translateX(${canonicalX}px)`}}>
          <DocCard doc={{id: "d1", label: "PgBouncer: cap asyncpg pools at 10", kind: "good"}} size="lg" />
        </div>
        <div style={{position: "absolute", right: 0, width: 560, opacity: duplicateOpacity, transform: `translateX(${duplicateX}px)`}}>
          <DocCard doc={{id: "d2", label: "Keep asyncpg pools small behind PgBouncer", kind: "good"}} size="lg" />
        </div>
      </div>
      <div style={{opacity: label, color: LETHE, fontSize: 26}}>
        cosine ≥ 0.95 · keep one canonical memory
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────

const QueryView: React.FC<{frame: number; fps: number}> = ({frame, fps}) => {
  const local = frame - PHASES[1].start;
  const s = spring({
    frame: Math.max(0, local - 8),
    fps,
    config: {damping: 200, stiffness: 200, mass: 0.6},
    durationInFrames: 18,
  });
  const opacity = interpolate(s, [0, 1], [0, 1]);
  const ty = interpolate(s, [0, 1], [12, 0]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 28,
        alignItems: "center",
        opacity,
        transform: `translateY(${ty}px)`,
      }}
    >
      <div style={{color: TEXT_DIM, fontSize: 22}}>a user asks</div>
      <div
        style={{
          fontSize: 52,
          color: TEXT,
          letterSpacing: -0.5,
          padding: "24px 40px",
          border: `2px solid ${LETHE}66`,
          background: `${LETHE}11`,
        }}
      >
        <span style={{color: LETHE, marginRight: 16}}>$</span>
        {QUERY}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────

const RetrieveView: React.FC<{frame: number; fps: number}> = ({frame, fps}) => {
  const local = frame - PHASES[2].start;
  return (
    <div style={{display: "flex", gap: 40, width: "100%", alignItems: "stretch"}}>
      <Lane
        title="sparse"
        subtitle="keyword match"
        ranking={SPARSE}
        local={local}
        fps={fps}
        delay={0}
      />
      <Lane
        title="dense"
        subtitle="vector similarity"
        ranking={DENSE}
        local={local}
        fps={fps}
        delay={20}
      />
    </div>
  );
};

const Lane: React.FC<{
  title: string;
  subtitle: string;
  ranking: string[];
  local: number;
  fps: number;
  delay: number;
}> = ({title, subtitle, ranking, local, fps, delay}) => {
  const headerS = spring({
    frame: Math.max(0, local - delay),
    fps,
    config: {damping: 200, stiffness: 220, mass: 0.6},
    durationInFrames: 14,
  });
  const headerO = interpolate(headerS, [0, 1], [0, 1]);
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        gap: 18,
      }}
    >
      <div
        style={{
          opacity: headerO,
          display: "flex",
          flexDirection: "column",
          gap: 4,
        }}
      >
        <div style={{color: LETHE, fontSize: 32}}>{title}</div>
        <div style={{color: TEXT_DIM, fontSize: 18}}>{subtitle}</div>
      </div>
      <div style={{display: "flex", flexDirection: "column", gap: 14}}>
        {ranking.map((id, i) => {
          const s = spring({
            frame: Math.max(0, local - delay - 14 - i * 8),
            fps,
            config: {damping: 200, stiffness: 220, mass: 0.6},
            durationInFrames: 14,
          });
          const o = interpolate(s, [0, 1], [0, 1]);
          const tx = interpolate(s, [0, 1], [-12, 0]);
          return (
            <DocCard
              key={id}
              doc={DOCS[id]}
              opacity={o}
              translateX={tx}
            />
          );
        })}
      </div>
    </div>
  );
};

const DocCard: React.FC<{
  doc: Doc;
  opacity?: number;
  translateX?: number;
  size?: "lg" | "md";
  faded?: boolean;
}> = ({doc, opacity = 1, translateX = 0, size = "md", faded = false}) => {
  const accent = doc.kind === "bad" ? COLOR_DISTRACTOR : COLOR_RELEVANT;
  const fontSize = size === "lg" ? 28 : 24;
  const padding = size === "lg" ? "20px 28px" : "16px 24px";
  return (
    <div
      style={{
        opacity: faded ? opacity * 0.35 : opacity,
        transform: `translateX(${translateX}px)`,
        display: "flex",
        alignItems: "center",
        gap: 18,
        padding,
        background: SURFACE,
        borderLeft: `4px solid ${accent}`,
        fontSize,
      }}
    >
      <span style={{color: TEXT}}>{doc.label}</span>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────

const MergeView: React.FC<{frame: number; fps: number}> = ({frame, fps}) => {
  const local = frame - PHASES[3].start;
  // Step 1: show two stacks
  // Step 2: arrows indicate fusion
  // Step 3: merged single column
  const fuseProgress = interpolate(local, [10, 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const leftX = interpolate(fuseProgress, [0, 1], [0, 380]);
  const rightX = interpolate(fuseProgress, [0, 1], [0, -380]);
  const leftOp = interpolate(fuseProgress, [0.6, 1], [1, 0]);
  const rightOp = interpolate(fuseProgress, [0.6, 1], [1, 0]);
  const mergedOp = interpolate(fuseProgress, [0.7, 1], [0, 1]);

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
      }}
    >
      <div
        style={{
          position: "absolute",
          left: 0,
          top: "50%",
          transform: `translate(${leftX}px, -50%)`,
          opacity: leftOp,
          width: 480,
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
      >
        <div style={{color: TEXT_DIM, fontSize: 20, marginBottom: 4}}>
          sparse
        </div>
        {SPARSE.map((id) => (
          <DocCard key={id} doc={DOCS[id]} />
        ))}
      </div>

      <div
        style={{
          position: "absolute",
          right: 0,
          top: "50%",
          transform: `translate(${rightX}px, -50%)`,
          opacity: rightOp,
          width: 480,
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
      >
        <div style={{color: TEXT_DIM, fontSize: 20, marginBottom: 4}}>
          dense
        </div>
        {DENSE.map((id) => (
          <DocCard key={id} doc={DOCS[id]} />
        ))}
      </div>

      <div
        style={{
          position: "absolute",
          left: "50%",
          top: "50%",
          transform: "translate(-50%, -50%)",
          opacity: mergedOp,
          width: 580,
          display: "flex",
          flexDirection: "column",
          gap: 16,
          alignItems: "stretch",
        }}
      >
        <div
          style={{
            color: TEXT_DIM,
            fontSize: 20,
            textAlign: "center",
            marginBottom: 4,
          }}
        >
          same candidates fuse by id
        </div>
        {MERGED.map((id) => (
          <DocCard key={id} doc={DOCS[id]} size="lg" />
        ))}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────

const RerankView: React.FC<{frame: number; fps: number}> = ({frame, fps}) => {
  const local = frame - PHASES[4].start;
  const reorderProgress = interpolate(local, [10, 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Position before (MERGED order) vs after (RERANKED order)
  const items = MERGED.map((id) => {
    const fromIdx = MERGED.indexOf(id);
    const toIdx = RERANKED.indexOf(id);
    const idx = lerp(fromIdx, toIdx, reorderProgress);
    return {id, idx, isLast: toIdx === RERANKED.length - 1};
  });

  return (
    <div
      style={{
        position: "relative",
        width: 720,
        height: 360,
      }}
    >
      {items.map((item) => {
        const top = item.idx * 110;
        const faded = reorderProgress > 0.6 && item.isLast;
        const fadeAmount = interpolate(
          reorderProgress,
          [0.6, 1],
          [0, 1],
          {extrapolateLeft: "clamp", extrapolateRight: "clamp"},
        );
        const fadedFinal = faded ? 1 - fadeAmount * 0.65 : 1;
        return (
          <div
            key={item.id}
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              top,
              opacity: fadedFinal,
            }}
          >
            <DocCard doc={DOCS[item.id]} size="lg" />
          </div>
        );
      })}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────

const LearnView: React.FC<{frame: number; fps: number}> = ({frame, fps}) => {
  const local = frame - PHASES[5].start;
  // Two events: bad doc → suppressed, good doc → reinforced
  const badProgress = interpolate(local, [12, 50], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const goodProgress = interpolate(local, [40, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 36,
        width: 820,
      }}
    >
      <Effect
        doc={DOCS.eD}
        verb="suppressed"
        progress={badProgress}
        color={COLOR_DISTRACTOR}
      />
      <Effect
        doc={DOCS.e1}
        verb="reinforced"
        progress={goodProgress}
        color={COLOR_RELEVANT}
      />
      <ClosingLine frame={frame} fps={fps} />
    </div>
  );
};

const Effect: React.FC<{
  doc: Doc;
  verb: string;
  progress: number;
  color: string;
}> = ({doc, verb, progress, color}) => {
  const opacity = interpolate(progress, [0, 0.3], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const tx = interpolate(progress, [0, 0.3], [-10, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <div
      style={{
        opacity,
        transform: `translateX(${tx}px)`,
        display: "flex",
        alignItems: "center",
        gap: 24,
        padding: "20px 28px",
        background: SURFACE,
        borderLeft: `4px solid ${color}`,
      }}
    >
      <span
        style={{
          color: TEXT,
          fontSize: 26,
          flex: 1,
        }}
      >
        {doc.label}
      </span>
      <div
        style={{
          height: 14,
          width: 220,
          background: `${color}20`,
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: `${progress * 100}%`,
            background: color,
          }}
        />
      </div>
      <span
        style={{
          color,
          fontSize: 22,
          minWidth: 160,
          textAlign: "right",
        }}
      >
        {verb}
      </span>
    </div>
  );
};

const ClosingLine: React.FC<{frame: number; fps: number}> = ({frame, fps}) => {
  const local = frame - PHASES[5].start;
  const s = spring({
    frame: Math.max(0, local - 70),
    fps,
    config: {damping: 200, stiffness: 200, mass: 0.6},
    durationInFrames: 18,
  });
  const opacity = interpolate(s, [0, 1], [0, 1]);
  return (
    <div
      style={{
        opacity,
        textAlign: "center",
        color: LETHE,
        fontSize: 28,
        marginTop: 12,
      }}
    >
      next query starts smarter
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────

const lerp = (a: number, b: number, t: number) =>
  a + (b - a) * Math.max(0, Math.min(1, t));
