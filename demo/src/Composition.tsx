import {useEffect, useState} from "react";
import {
  AbsoluteFill,
  continueRender,
  delayRender,
  staticFile,
  useCurrentFrame,
} from "remotion";
import {Intro} from "./Intro";
import {LearningCurve} from "./LearningCurve";
import {Outro} from "./Outro";
import {RifSchematic} from "./RifSchematic";
import {BG, FONT_MONO, TEXT_DIM} from "./theme";
import type {RunData} from "./types";

export const INTRO_FRAMES = 60;
export const SCHEMATIC_FRAMES = 330;
export const CURVE_FRAMES = 120;
export const OUTRO_FRAMES = 90;

export const LetheDemo: React.FC = () => {
  const [run, setRun] = useState<RunData | null>(null);
  const [handle] = useState(() => delayRender("run.json"));

  useEffect(() => {
    fetch(staticFile("run.json"))
      .then((r) => r.json() as Promise<RunData>)
      .then((data) => {
        setRun(data);
        continueRender(handle);
      })
      .catch((err) => {
        console.error("Failed to load run.json", err);
        continueRender(handle);
      });
  }, [handle]);

  const frame = useCurrentFrame();

  if (!run) {
    return (
      <AbsoluteFill
        style={{background: BG, fontFamily: FONT_MONO, color: TEXT_DIM}}
      >
        <div style={{padding: 48}}>loading run.json...</div>
      </AbsoluteFill>
    );
  }

  const schematicEnd = INTRO_FRAMES + SCHEMATIC_FRAMES;
  const curveEnd = schematicEnd + CURVE_FRAMES;

  if (frame < INTRO_FRAMES) {
    return <Intro frame={frame} durationInFrames={INTRO_FRAMES} />;
  }
  if (frame < schematicEnd) {
    return (
      <RifSchematic
        frame={frame - INTRO_FRAMES}
        durationInFrames={SCHEMATIC_FRAMES}
      />
    );
  }
  if (frame < curveEnd) {
    return (
      <LearningCurve
        frame={frame - schematicEnd}
        durationInFrames={CURVE_FRAMES}
        run={run}
      />
    );
  }
  return (
    <Outro
      frame={frame - curveEnd}
      durationInFrames={OUTRO_FRAMES}
      run={run}
    />
  );
};
