// End-to-end smoke for the napi-rs binding.
//
// Run: `npm test` (which runs `napi build` first via the script).
// Asserts that add → retrieve → save → reload → retrieve all work
// against a fresh tempdir store. Models load from the local HF cache
// (~/.cache/huggingface/hub/) — first run downloads ~50 MB.

import { MemoryStore } from "../index.js";
import os from "node:os";
import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "lethe-node-smoke-"));
let failed = false;

try {
  const store = await MemoryStore.open(tmp);

  // add() returns the assigned id on success, null on dedup.
  const id = await store.add("Rust is a systems programming language.");
  assert.equal(typeof id, "string", "first add should return an id");

  const dup = await store.add("Rust is a systems programming language.");
  assert.equal(dup, null, "exact-content add should dedup");

  await store.add("Python is a high-level interpreted language.");
  await store.add("Postgres is a relational database.");
  assert.equal(store.size(), 3, "size after three unique adds");

  // retrieve()
  const hits = await store.retrieve("Rust", 3);
  assert.ok(hits.length > 0, "retrieve should return at least one hit");
  assert.ok(
    hits[0].content.includes("Rust"),
    `top hit should mention "Rust": got ${hits[0].content}`,
  );
  assert.equal(typeof hits[0].id, "string");
  assert.equal(typeof hits[0].score, "number");

  // save()
  await store.save();

  // Reopen and verify state survived.
  const reopened = await MemoryStore.open(tmp);
  assert.equal(reopened.size(), 3, "size should persist across save/reopen");
  const dup2 = await reopened.add("Rust is a systems programming language.");
  assert.equal(dup2, null, "dedup state should persist across save/reopen");

  console.log(
    `OK — size=${store.size()}, top hit="${hits[0].content}" score=${hits[0].score.toFixed(3)}`,
  );
} catch (err) {
  failed = true;
  console.error(err);
} finally {
  await fs.rm(tmp, { recursive: true, force: true });
}

process.exit(failed ? 1 : 0);
