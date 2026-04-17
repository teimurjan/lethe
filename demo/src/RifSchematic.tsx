import {AbsoluteFill, interpolate, spring, useVideoConfig} from "remotion";
import {BG, BORDER, COLOR_DISTRACTOR, COLOR_RELEVANT, FONT_MONO, LETHE, SURFACE, TEXT, TEXT_DIM, TEXT_FAINT} from "./theme";

type Props = {
  frame: number;
  durationInFrames: number;
};

const PHASES = [
  {start: 0, label: "title"},
  {start: 30, label: "candidates"},
  {start: 100, label: "rerank"},
  {start: 160, label: "suppress"},
  {start: 220, label: "next"},
  {start: 280, label: "summary"},
] as const;

type Entry = {
  id: string;
  text: string;
  relevant: boolean;
  distractor: boolean;
};

const ENTRIES: Entry[] = [
  {id: "e1", text: "PgBouncer transaction mode config", relevant: true, distractor: false},
  {id: "eD", text: "404 — PostgreSQL pool page removed", relevant: false, distractor: true},
  {id: "e2", text: "asyncpg pool_size for FastAPI", relevant: true, distractor: false},
  {id: "e3", text: "SQLAlchemy max_overflow settings", relevant: true, distractor: false},
  {id: "e4", text: "psycopg3 ConnectionPool usage", relevant: true, distractor: false},
];

export const RifSchematic: React.FC<Props> = ({frame, durationInFrames}) => {
  const {fps} = useVideoConfig();
  const alpha = interpolate(
    frame,
    [0, 10, durationInFrames - 10, durationInFrames],
    [0, 1, 1, 0],
    {extrapolateLeft: "clamp", extrapolateRight: "clamp"},
  );

  const phase = PHASES.reduce(
    (cur, p) => (frame >= p.start ? p : cur),
    PHASES[0],
  );

  const showCandidates = frame >= PHASES[1].start;
  const showRerank = frame >= PHASES[2].start;
  const showSuppress = frame >= PHASES[3].start;
  const showNext = frame >= PHASES[4].start;
  const showSummary = frame >= PHASES[5].start;

  // After rerank: distractor (eD) drops from rank 2 to rank 5
  const reranked = showRerank
    ? [ENTRIES[0], ENTRIES[2], ENTRIES[3], ENTRIES[4], ENTRIES[1]]
    : ENTRIES;

  // After "next query": distractor excluded entirely
  const nextRound = showNext
    ? [ENTRIES[0], ENTRIES[2], ENTRIES[3], ENTRIES[4]]
    : null;

  const activeEntries = nextRound ?? reranked;
  const suppressedId = showSuppress && !showNext ? "eD" : null;

  return (
    <AbsoluteFill
      style={{
        background: BG,
        fontFamily: FONT_MONO,
        color: TEXT,
        opacity: alpha,
        padding: "40px 60px",
        display: "flex",
        flexDirection: "column",
        gap: 18,
      }}
    >
      <div style={{display: "flex", alignItems: "baseline", gap: 14}}>
        <span style={{color: TEXT_DIM, fontSize: 13}}>algorithm</span>
        <span style={{color: TEXT, fontSize: 20}}>
          retrieval-induced forgetting (RIF)
        </span>
      </div>

      <div style={{display: "flex", gap: 24, flex: 1, minHeight: 0}}>
        {/* Left: flow diagram */}
        <div
          style={{
            flex: 1,
            background: SURFACE,
            border: `1px solid ${BORDER}`,
            padding: 20,
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          {/* Query */}
          <Step
            visible={showCandidates}
            frame={frame}
            delay={PHASES[1].start}
            fps={fps}
          >
            <div style={{color: TEXT_DIM, fontSize: 12, marginBottom: 4}}>
              {showNext ? "query #2" : "query #1"}
            </div>
            <div
              style={{
                padding: "8px 14px",
                border: `1px solid ${LETHE}44`,
                background: `${LETHE}11`,
                fontSize: 15,
              }}
            >
              <span style={{color: LETHE}}>$</span>{" "}
              PostgreSQL connection pool config
            </div>
          </Step>

          {/* Arrow */}
          {showCandidates && (
            <div style={{color: TEXT_FAINT, fontSize: 13, paddingLeft: 20}}>
              {showRerank ? "cross-encoder reranks →" : "BM25 + vector →"}
            </div>
          )}

          {/* Candidate list */}
          {showCandidates && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 4,
                flex: 1,
              }}
            >
              {activeEntries.map((entry, i) => {
                const isDistractor = entry.distractor;
                const isSuppressed = entry.id === suppressedId;
                const fg = isDistractor
                  ? COLOR_DISTRACTOR
                  : entry.relevant
                    ? COLOR_RELEVANT
                    : TEXT_DIM;
                const sym = isDistractor ? "!" : entry.relevant ? "+" : "·";

                return (
                  <Step
                    key={entry.id}
                    visible
                    frame={frame}
                    delay={
                      showNext
                        ? PHASES[4].start + i * 3
                        : showRerank
                          ? PHASES[2].start + i * 3
                          : PHASES[1].start + 10 + i * 4
                    }
                    fps={fps}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        height: 42,
                        padding: "0 12px",
                        borderLeft: `2px solid ${fg}`,
                        background: isSuppressed
                          ? `${COLOR_DISTRACTOR}15`
                          : "rgba(255,255,255,0.012)",
                        borderBottom: `1px solid ${BORDER}`,
                        opacity: isSuppressed ? 0.5 : 1,
                      }}
                    >
                      <span style={{width: 28, color: TEXT_DIM, fontSize: 12}}>
                        #{i + 1}
                      </span>
                      <span
                        style={{
                          width: 14,
                          color: fg,
                          fontSize: 14,
                          fontWeight: 600,
                        }}
                      >
                        {sym}
                      </span>
                      <span
                        style={{
                          flex: 1,
                          fontSize: 13,
                          color: isSuppressed ? TEXT_FAINT : TEXT,
                          marginLeft: 10,
                          textDecoration: isSuppressed
                            ? "line-through"
                            : "none",
                        }}
                      >
                        {entry.text}
                      </span>
                      {isSuppressed && (
                        <span
                          style={{
                            fontSize: 11,
                            color: COLOR_DISTRACTOR,
                            marginLeft: 8,
                          }}
                        >
                          suppressed
                        </span>
                      )}
                    </div>
                  </Step>
                );
              })}
            </div>
          )}
        </div>

        {/* Right: explanation */}
        <div
          style={{
            width: 360,
            display: "flex",
            flexDirection: "column",
            gap: 12,
            paddingTop: 8,
          }}
        >
          <PhaseNote visible={showCandidates && !showRerank} frame={frame} delay={PHASES[1].start} fps={fps}>
            BM25 + vector search surfaces candidates. A distractor
            <span style={{color: COLOR_DISTRACTOR}}> (!) </span>
            ranks high on keyword overlap.
          </PhaseNote>
          <PhaseNote visible={showRerank && !showSuppress} frame={frame} delay={PHASES[2].start} fps={fps}>
            Cross-encoder reranks by semantic relevance. Distractor drops
            from #2 to #5 — but still in the pool.
          </PhaseNote>
          <PhaseNote visible={showSuppress && !showNext} frame={frame} delay={PHASES[3].start} fps={fps}>
            RIF marks the loser: it ranked high in BM25 but low after
            cross-encoder.
            <span style={{color: COLOR_DISTRACTOR}}> Suppression score grows.</span>
          </PhaseNote>
          <PhaseNote visible={showNext && !showSummary} frame={frame} delay={PHASES[4].start} fps={fps}>
            Next query: suppressed entry is penalized
            <span style={{color: LETHE}}> before </span>
            the cross-encoder sees it. Slot freed for a better candidate.
          </PhaseNote>
          <PhaseNote visible={showSummary} frame={frame} delay={PHASES[5].start} fps={fps}>
            Over time, chronic distractors accumulate suppression and fade
            from the candidate pool. Retrieval quality improves
            <span style={{color: LETHE}}> without retraining any model.</span>
          </PhaseNote>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Step: React.FC<{
  visible: boolean;
  frame: number;
  delay: number;
  fps: number;
  children: React.ReactNode;
}> = ({visible, frame, delay, fps, children}) => {
  if (!visible) return null;
  const s = spring({
    frame: Math.max(0, frame - delay),
    fps,
    config: {damping: 200, stiffness: 260, mass: 0.5},
    durationInFrames: 12,
  });
  const opacity = interpolate(s, [0, 1], [0, 1]);
  const translate = interpolate(s, [0, 1], [8, 0]);
  return (
    <div style={{opacity, transform: `translateX(${translate}px)`}}>
      {children}
    </div>
  );
};

const PhaseNote: React.FC<{
  visible: boolean;
  frame: number;
  delay: number;
  fps: number;
  children: React.ReactNode;
}> = ({visible, frame, delay, fps, children}) => {
  if (!visible) return null;
  const s = spring({
    frame: Math.max(0, frame - delay),
    fps,
    config: {damping: 200, stiffness: 200, mass: 0.6},
    durationInFrames: 14,
  });
  const opacity = interpolate(s, [0, 1], [0, 1]);
  return (
    <div
      style={{
        opacity,
        fontSize: 15,
        lineHeight: 1.6,
        color: TEXT,
        padding: "10px 14px",
        borderLeft: `2px solid ${LETHE}44`,
      }}
    >
      {children}
    </div>
  );
};
