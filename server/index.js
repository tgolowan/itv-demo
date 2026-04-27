import express from 'express';
import cors from 'cors';
import multer from 'multer';
import dotenv from 'dotenv';
import os from 'os';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import FormData from 'form-data';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const PORT = process.env.PORT || 5050;
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const VISION_MODEL = process.env.OLLAMA_VISION_MODEL || 'llava';
const TEXT_MODEL = process.env.OLLAMA_TEXT_MODEL || 'mistral';
const PYTHON_BIN = process.env.PYTHON_BIN || 'python3';
const OUTPUT_DIR = path.resolve(ROOT, process.env.OUTPUT_DIR || './outputs');

if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const DIST_DIR = path.join(ROOT, 'dist');

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Serve built frontend if present (production: `npm run build`)
if (fs.existsSync(DIST_DIR)) {
  app.use(express.static(DIST_DIR));
}

// Friendly root in dev (no dist build yet)
app.get('/', (_req, res) => {
  if (fs.existsSync(path.join(DIST_DIR, 'index.html'))) {
    return res.sendFile(path.join(DIST_DIR, 'index.html'));
  }
  res.set('Content-Type', 'text/html').send(`<!doctype html>
<html><head><title>LocalVideoGen API</title>
<style>body{font:14px/1.5 ui-monospace,monospace;background:#030712;color:#e5e7eb;padding:40px;max-width:680px;margin:auto}
a{color:#00FFD1}h1{background:linear-gradient(90deg,#00FFD1,#A855F7,#FF6B00);-webkit-background-clip:text;color:transparent}
code{background:#111827;padding:2px 6px;border-radius:4px}</style></head>
<body>
<h1>LocalVideoGen API</h1>
<p>This is the API server on port ${PORT}. The web UI runs on Vite.</p>
<p><strong>→ Open <a href="http://localhost:3000">http://localhost:3000</a></strong></p>
<p>If the UI isn't running, start it with <code>npm run dev</code> from the project root.</p>
<p>API endpoints: <a href="/api/health">/api/health</a> · <a href="/api/system">/api/system</a> · <a href="/api/ollama/status">/api/ollama/status</a> · <a href="/api/jobs">/api/jobs</a></p>
</body></html>`);
});

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (['image/jpeg', 'image/png', 'image/webp'].includes(file.mimetype)) cb(null, true);
    else cb(new Error('Only JPEG/PNG/WebP allowed'));
  },
});

// ---------- helpers ----------
const jobDir = (id) => path.join(OUTPUT_DIR, id);
const metaPath = (id) => path.join(jobDir(id), 'meta.json');
const readMeta = (id) => {
  try { return JSON.parse(fs.readFileSync(metaPath(id), 'utf8')); } catch { return null; }
};
const writeMeta = (id, data) => {
  fs.writeFileSync(metaPath(id), JSON.stringify(data, null, 2));
};

// ---------- system ----------
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', uptime: process.uptime(), memory: process.memoryUsage() });
});

app.get('/api/system', (_req, res) => {
  res.json({
    platform: os.platform(),
    arch: os.arch(),
    cpus: os.cpus().length,
    totalMemory: os.totalmem(),
    freeMemory: os.freemem(),
    hostname: os.hostname(),
  });
});

// ---------- ollama ----------
app.get('/api/ollama/status', async (_req, res) => {
  try {
    const r = await axios.get(`${OLLAMA_URL}/api/tags`, { timeout: 3000 });
    const models = (r.data.models || []).map((m) => m.name);
    const required = [VISION_MODEL, TEXT_MODEL];
    const setup = required.every((req) => models.some((m) => m.startsWith(req)));
    res.json({ status: 'online', models, required, setup });
  } catch {
    res.json({ status: 'offline', models: [], required: [VISION_MODEL, TEXT_MODEL], setup: false });
  }
});

// ---------- prompt from image (llava) ----------
app.post('/api/generate/prompt-from-image', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image uploaded' });
  try {
    const b64 = req.file.buffer.toString('base64');
    const r = await axios.post(`${OLLAMA_URL}/api/generate`, {
      model: VISION_MODEL,
      prompt: 'Describe this image in vivid cinematic detail for a video generation prompt. Include subject, environment, lighting, mood, and camera motion. Keep under 80 words.',
      images: [b64],
      stream: false,
    }, { timeout: 120000 });
    res.json({ prompt: (r.data.response || '').trim() });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------- enhance prompt (mistral) ----------
app.post('/api/generate/enhance-prompt', async (req, res) => {
  const { prompt, style } = req.body || {};
  if (!prompt) return res.status(400).json({ error: 'Missing prompt' });
  try {
    const sys = `Rewrite the user's video idea as a single vivid cinematic prompt in the "${style || 'Cinematic'}" style. Add concrete visual detail: lighting, color palette, composition, camera motion. No preamble. Under 80 words.`;
    const r = await axios.post(`${OLLAMA_URL}/api/generate`, {
      model: TEXT_MODEL,
      prompt: `${sys}\n\nIdea: ${prompt}\n\nPrompt:`,
      stream: false,
    }, { timeout: 120000 });
    res.json({ enhanced: (r.data.response || '').trim() });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------- ollama chat (local LLM playground) ----------
app.post('/api/ollama/chat', async (req, res) => {
  const { messages, model } = req.body || {};
  const m = (model || TEXT_MODEL || 'llama3.2').trim();
  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'messages[] required' });
  }
  try {
    const r = await axios.post(
      `${OLLAMA_URL}/api/chat`,
      { model: m, messages, stream: false },
      { timeout: 600000 },
    );
    const msg = r.data?.message;
    const content = typeof msg?.content === 'string' ? msg.content : '';
    res.json({ model: m, message: msg || { role: 'assistant', content }, content });
  } catch (e) {
    const detail = e.response?.data ? JSON.stringify(e.response.data) : e.message;
    res.status(500).json({ error: detail });
  }
});

// ---------- create video job ----------
app.post('/api/generate/video', upload.single('image'), async (req, res) => {
  const prompt = (req.body.prompt || '').trim();
  const style = req.body.style || 'Cinematic';
  const duration = parseInt(req.body.duration || '6', 10);
  const fps = parseInt(req.body.fps || '24', 10);
  if (!prompt && !req.file) return res.status(400).json({ error: 'Provide prompt or image' });

  const id = uuidv4();
  fs.mkdirSync(jobDir(id), { recursive: true });

  let imagePath = null;
  if (req.file) {
    imagePath = path.join(jobDir(id), `input_${req.file.originalname.replace(/[^\w.\-]/g, '_')}`);
    fs.writeFileSync(imagePath, req.file.buffer);
  }

  const job = {
    job_id: id,
    status: 'queued',
    progress: 0,
    prompt,
    style,
    duration,
    fps,
    image: imagePath ? path.basename(imagePath) : null,
    created_at: new Date().toISOString(),
  };
  writeMeta(id, job);

  // run async
  runVideoGen(id, { imagePath, prompt, duration, fps }).catch((e) => {
    const m = readMeta(id) || job;
    writeMeta(id, { ...m, status: 'failed', error: e.message, progress: 0 });
  });

  res.json(job);
});

async function runVideoGen(id, { imagePath, prompt, duration, fps }) {
  const meta = readMeta(id);
  writeMeta(id, { ...meta, status: 'running', progress: 5 });

  const outPath = path.join(jobDir(id), 'output.mp4');
  const script = path.join(__dirname, 'video_gen.py');
  const args = [script];
  if (imagePath) args.push(imagePath);
  else args.push('--text-only');
  args.push(outPath, '--duration', String(duration), '--fps', String(fps));
  if (prompt) args.push('--prompt', prompt);

  await new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, args, { cwd: __dirname });
    let stderr = '';
    let stdoutTail = '';
    proc.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      stdoutTail = (stdoutTail + text).slice(-4000);
      for (const line of text.split('\n')) {
        const m = line.match(/PROGRESS:(\d+)/);
        if (m) {
          const cur = readMeta(id);
          if (cur) writeMeta(id, { ...cur, progress: parseInt(m[1], 10) });
        }
      }
    });
    proc.stderr.on('data', (c) => { stderr = (stderr + c.toString()).slice(-4000); });
    proc.on('error', reject);
    proc.on('close', (code, signal) => {
      if (code === 0) return resolve();
      const reason = signal ? `killed by ${signal} (likely OOM)` : `exited ${code}`;
      reject(new Error(`${reason}\nstdout: ${stdoutTail}\nstderr: ${stderr}`));
    });
  });

  const cur = readMeta(id);
  writeMeta(id, {
    ...cur,
    status: 'completed',
    progress: 100,
    output: 'output.mp4',
    completed_at: new Date().toISOString(),
  });
}

// ---------- jobs ----------
app.get('/api/jobs', (_req, res) => {
  const ids = fs.existsSync(OUTPUT_DIR) ? fs.readdirSync(OUTPUT_DIR) : [];
  const jobs = ids
    .map((id) => readMeta(id))
    .filter(Boolean)
    .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
  res.json({ jobs, total: jobs.length });
});

app.get('/api/jobs/:jobId', (req, res) => {
  const m = readMeta(req.params.jobId);
  if (!m) return res.status(404).json({ error: 'Job not found' });
  res.json(m);
});

app.get('/api/jobs/:jobId/download', (req, res) => {
  const file = path.join(jobDir(req.params.jobId), 'output.mp4');
  if (!fs.existsSync(file)) return res.status(404).json({ error: 'Output not ready' });
  res.download(file, `${req.params.jobId}.mp4`);
});

app.delete('/api/jobs/:jobId', (req, res) => {
  const dir = jobDir(req.params.jobId);
  if (!fs.existsSync(dir)) return res.status(404).json({ error: 'Not found' });
  fs.rmSync(dir, { recursive: true, force: true });
  res.json({ deleted: true });
});

// ---------- startup ----------
const banner = `
╔══════════════════════════════════════════════════════╗
║   ██╗      ██████╗  ██████╗ █████╗ ██╗               ║
║   ██║     ██╔═══██╗██╔════╝██╔══██╗██║               ║
║   ██║     ██║   ██║██║     ███████║██║               ║
║   ██║     ██║   ██║██║     ██╔══██║██║               ║
║   ███████╗╚██████╔╝╚██████╗██║  ██║███████╗          ║
║   VideoGen API · Apple Silicon · Ollama + SVD        ║
╚══════════════════════════════════════════════════════╝
  http://localhost:${PORT}
  GET    /api/health
  GET    /api/system
  GET    /api/ollama/status
  POST   /api/generate/prompt-from-image
  POST   /api/generate/enhance-prompt
  POST   /api/ollama/chat
  POST   /api/generate/video
  GET    /api/jobs
  GET    /api/jobs/:id
  GET    /api/jobs/:id/download
  DELETE /api/jobs/:id
  Ollama: ${OLLAMA_URL}  models: ${VISION_MODEL}, ${TEXT_MODEL}
  Output: ${OUTPUT_DIR}
`;

const server = app.listen(PORT, () => console.log(banner));

const shutdown = () => {
  console.log('\nShutting down…');
  server.close(() => process.exit(0));
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
