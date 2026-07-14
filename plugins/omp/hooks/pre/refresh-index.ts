import { fileURLToPath } from "node:url";
import type { HookAPI } from "@oh-my-pi/pi-coding-agent/extensibility/hooks";

const refreshScript = fileURLToPath(new URL("../refresh-index.sh", import.meta.url));

export default function refreshIndexHook(pi: HookAPI): void {
	pi.on("before_agent_start", async () => {
		await pi.exec("bash", [refreshScript]);
	});
}
