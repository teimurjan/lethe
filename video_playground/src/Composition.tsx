import {useEffect, useState} from "react";
import {
  AbsoluteFill,
  continueRender,
  delayRender,
  staticFile,
  useCurrentFrame,
} from "remotion";
import {Intro} from "./Intro";
import {Outro} from "./Outro";
import {PipelineFlow} from "./PipelineFlow";
import {BG, FONT_MONO, TEXT_DIM} from "./theme";
import type {RunData} from "./types";

export const INTRO_FRAMES = 60;
export const FLOW_FRAMES = 570;
export const OUTRO_FRAMES = 150;

export const LetheDemo: React.FC = () => {
  const [run, setRun] = useState<RunData | null>(null);
  const [handle] = useState(() => delayRender("run_replay_extended.json"));

  useEffect(() => {
    fetch(staticFile("run_replay_extended.json"))
      .then((r) => r.json() as Promise<RunData>)
      .then((data) => {
        setRun(data);
        continueRender(handle);
      })
      .catch((err) => {
        console.error("Failed to load run_replay_extended.json", err);
        continueRender(handle);
      });
  }, [handle]);

  const frame = useCurrentFrame();

  if (!run) {
    return (
      <AbsoluteFill
        style={{background: BG, fontFamily: FONT_MONO, color: TEXT_DIM}}
      >
        <div style={{padding: 48}}>loading run_replay_extended.json...</div>
      </AbsoluteFill>
    );
  }

  const flowEnd = INTRO_FRAMES + FLOW_FRAMES;

  if (frame < INTRO_FRAMES) {
    return <Intro frame={frame} durationInFrames={INTRO_FRAMES} />;
  }
  if (frame < flowEnd) {
    return (
      <PipelineFlow
        frame={frame - INTRO_FRAMES}
        durationInFrames={FLOW_FRAMES}
      />
    );
  }
  return (
    <Outro
      frame={frame - flowEnd}
      durationInFrames={OUTRO_FRAMES}
      run={run}
    />
  );
};
