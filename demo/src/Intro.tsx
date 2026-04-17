import {AbsoluteFill, Img, interpolate, staticFile} from "remotion";
import {BG, FONT_MONO, LETHE, TEXT, TEXT_DIM} from "./theme";

type Props = {
  frame: number;
  durationInFrames: number;
};

export const Intro: React.FC<Props> = ({frame, durationInFrames}) => {
  const alpha = interpolate(
    frame,
    [0, 10, durationInFrames - 8, durationInFrames],
    [0, 1, 1, 0],
    {extrapolateLeft: "clamp", extrapolateRight: "clamp"},
  );
  const logoRise = interpolate(frame, [0, 20], [12, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const caretVisible = Math.floor(frame / 8) % 2 === 0;

  return (
    <AbsoluteFill
      style={{
        background: BG,
        fontFamily: FONT_MONO,
        color: TEXT,
        opacity: alpha,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 36,
          transform: `translateY(${logoRise}px)`,
        }}
      >
        <Img
          src={staticFile("logo.png")}
          style={{width: 180, height: 180, imageRendering: "auto"}}
        />
        <div>
          <div style={{fontSize: 54, fontWeight: 500, letterSpacing: -1}}>
            lethe
          </div>
          <div style={{fontSize: 20, color: TEXT_DIM, marginTop: 6}}>
            self-improving memory for LLM agents
          </div>
          <div style={{fontSize: 18, marginTop: 22, color: TEXT}}>
            <span style={{color: LETHE}}>$</span>{" "}
            <span>retrieval that learns</span>
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 20,
                background: caretVisible ? LETHE : "transparent",
                marginLeft: 4,
                verticalAlign: "middle",
              }}
            />
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
