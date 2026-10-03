// JSON-lines bridge: one request per stdin line, one response per stdout line.
//   request:  {"id": 1, "state": "...", "questions": {qid: {type, instructions, criteria?}}}
//   response: {"id": 1, "answers": {...}, "usage": {...}}  or  {"id": 1, "error": "..."}
// Model location: LAYA_MODEL_DIR (local ONNX bundle) or the published Hugging Face bundle,
// downloaded once (~1.7 GB) into LAYA_CACHE / ~/.cache/receptron-laya.
import { createInterface } from "node:readline";
import { Laya } from "@receptron/laya";

const opts = {};
if (process.env.LAYA_MODEL_DIR) opts.modelDir = process.env.LAYA_MODEL_DIR;
if (process.env.LAYA_SUBFOLDER) opts.subfolder = process.env.LAYA_SUBFOLDER; // e.g. "multilingual"
if (process.env.LAYA_THREADS) opts.sessionOptions = { intraOpNumThreads: Number(process.env.LAYA_THREADS) };
// e.g. LAYA_PROVIDERS=cuda,cpu (Linux + NVIDIA), dml,cpu (Windows), coreml,cpu (macOS)
if (process.env.LAYA_PROVIDERS) opts.executionProviders = process.env.LAYA_PROVIDERS.split(",");
if (!opts.modelDir) {
  let last = "";
  opts.onProgress = ({ file, received, total }) => {
    const pct = total ? Math.floor((100 * received) / total) : null;
    const line = `downloading ${file} ${pct ?? Math.floor(received / 1e6) + " MB"}${pct === null ? "" : "%"}`;
    if (line !== last && (pct === null || pct % 10 === 0)) process.stderr.write(line + "\n");
    last = line;
  };
}

const laya = await Laya.load(opts);
process.stdout.write(JSON.stringify({ ready: true, modelDir: laya.modelDir }) + "\n");

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!line.trim()) continue;
  let id = null;
  try {
    const req = JSON.parse(line);
    id = req.id;
    const res = await laya.systemOne(req.state, req.questions);
    process.stdout.write(JSON.stringify({ id, answers: res.answers, usage: res.usage }) + "\n");
  } catch (e) {
    process.stdout.write(JSON.stringify({ id, error: String(e?.message ?? e) }) + "\n");
  }
}
await laya.close();
