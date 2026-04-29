import {Composition} from "remotion";
import {LetheDemo} from "./Composition";

const FPS = 30;
const INTRO_FRAMES = 60;
const SCHEMATIC_FRAMES = 330;
const CURVE_FRAMES = 180;
const OUTRO_FRAMES = 120;
const DURATION = INTRO_FRAMES + SCHEMATIC_FRAMES + CURVE_FRAMES + OUTRO_FRAMES;

export const Root: React.FC = () => {
  return (
    <Composition
      id="lethe-demo"
      component={LetheDemo}
      durationInFrames={DURATION}
      fps={FPS}
      width={1280}
      height={720}
    />
  );
};
