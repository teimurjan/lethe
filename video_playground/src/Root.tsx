import {Composition} from "remotion";
import {
  FLOW_FRAMES,
  INTRO_FRAMES,
  LetheDemo,
  OUTRO_FRAMES,
} from "./Composition";

const FPS = 30;
const DURATION = INTRO_FRAMES + FLOW_FRAMES + OUTRO_FRAMES;

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
