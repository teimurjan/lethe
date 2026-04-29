#!/usr/bin/env bash
# Install the lethe Codex CLI plugin into ~/.codex/.
#
# Copies hook scripts and skills to ~/.codex/lethe/, then prints (or, with
# --auto-config, appends) the config.toml snippet that wires the hooks.

set -eu
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HOME}/.codex/lethe"
CONFIG_FILE="${HOME}/.codex/config.toml"
AUTO_CONFIG=0

for arg in "$@"; do
  case "${arg}" in
    --auto-config) AUTO_CONFIG=1 ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--auto-config]

Copies hooks and skills to ${DEST}. Without --auto-config, prints the snippet
to add to ${CONFIG_FILE}. With --auto-config, appends the snippet between
sentinel markers (idempotent — re-runs replace the previous block).
EOF
      exit 0
      ;;
    *) printf 'Unknown arg: %s\n' "${arg}" >&2; exit 2 ;;
  esac
done

mkdir -p "${DEST}"
cp -R "${SCRIPT_DIR}/hooks" "${DEST}/"
cp -R "${SCRIPT_DIR}/skills" "${DEST}/"
chmod +x "${DEST}/hooks/"*.sh

printf 'Installed lethe Codex hooks to %s\n' "${DEST}"

SNIPPET=$(cat <<EOF
# >>> lethe codex plugin >>>
[features]
codex_hooks = true

[[hooks.SessionStart]]
[[hooks.SessionStart.hooks]]
type = "command"
command = "bash ${DEST}/hooks/session-start.sh"
timeout = 15

[[hooks.UserPromptSubmit]]
[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = "bash ${DEST}/hooks/user-prompt-submit.sh"
timeout = 10

[[hooks.Stop]]
[[hooks.Stop.hooks]]
type = "command"
command = "bash ${DEST}/hooks/stop.sh"
timeout = 120

[[skills.config]]
path = "${DEST}/skills/recall"

[[skills.config]]
path = "${DEST}/skills/recall-global"
# <<< lethe codex plugin <<<
EOF
)

if [ "${AUTO_CONFIG}" -eq 0 ]; then
  cat <<EOF

Add the following to ${CONFIG_FILE} (or run with --auto-config to do it for you):

${SNIPPET}
EOF
  exit 0
fi

mkdir -p "$(dirname "${CONFIG_FILE}")"
touch "${CONFIG_FILE}"

# Remove any previous lethe block, then append the current one.
TMP="$(mktemp)"
awk '
  /^# >>> lethe codex plugin >>>/ { skip = 1; next }
  /^# <<< lethe codex plugin <<</ { skip = 0; next }
  !skip { print }
' "${CONFIG_FILE}" >"${TMP}"

# Trim trailing blank lines so the appended block is cleanly separated.
sed -e :a -e '/^$/{$d;N;ba' -e '}' "${TMP}" >"${CONFIG_FILE}"
rm -f "${TMP}"

# Ensure exactly one blank line before the new block (file may now be empty).
[ -s "${CONFIG_FILE}" ] && printf '\n' >>"${CONFIG_FILE}"
printf '%s\n' "${SNIPPET}" >>"${CONFIG_FILE}"

printf 'Patched %s with lethe hooks + skills.\n' "${CONFIG_FILE}"
