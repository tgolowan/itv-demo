import { useRef, useState } from 'react';

const STYLES = ['Cinematic', 'Abstract', 'Realistic', 'Animated', 'Surreal'];

/** TTV = text-to-video (prompt drives a synthetic keyframe). ITV = image-to-video (SVD from your still). */
export default function InputPanel({ onJobCreated, ollamaReady }) {
  const fileRef = useRef(null);
  const [mode, setMode] = useState('ttv');
  const [prompt, setPrompt] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [style, setStyle] = useState('Cinematic');
  const [duration, setDuration] = useState(6);
  const [fps, setFps] = useState(24);
  const [loading, setLoading] = useState(false);
  const [enhancing, setEnhancing] = useState(false);
  const [error, setError] = useState('');

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setImage(f);
    setImagePreview(URL.createObjectURL(f));
  };

  const clearImage = () => {
    setImage(null);
    setImagePreview(null);
    if (fileRef.current) fileRef.current.value = '';
  };

  const describeImage = async () => {
    if (!image) return;
    setEnhancing(true);
    setError('');
    try {
      const fd = new FormData();
      fd.append('image', image);
      const res = await fetch('/api/generate/prompt-from-image', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.prompt) setPrompt(data.prompt);
      else setError(data.error || 'Failed to describe image');
    } catch (e) { setError(e.message); }
    setEnhancing(false);
  };

  const enhancePrompt = async () => {
    if (!prompt.trim()) return;
    setEnhancing(true);
    setError('');
    try {
      const res = await fetch('/api/generate/enhance-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, style }),
      });
      const data = await res.json();
      if (data.enhanced) setPrompt(data.enhanced);
      else setError(data.error || 'Failed to enhance prompt');
    } catch (e) { setError(e.message); }
    setEnhancing(false);
  };

  const generate = async () => {
    if (mode === 'ttv' && !prompt.trim()) {
      setError('Text-to-video needs a prompt describing the scene and motion.');
      return;
    }
    if (mode === 'itv' && !image) {
      setError('Image-to-video needs a reference still (JPEG, PNG, or WebP).');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const fd = new FormData();
      fd.append('prompt', prompt);
      fd.append('style', style);
      fd.append('duration', String(duration));
      fd.append('fps', String(fps));
      if (mode === 'itv' && image) fd.append('image', image);
      const res = await fetch('/api/generate/video', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.job_id) {
        onJobCreated(data);
        setPrompt('');
        clearImage();
      } else {
        setError(data.error || 'Failed to start generation');
      }
    } catch (e) { setError(e.message); }
    setLoading(false);
  };

  const canSubmit =
    mode === 'ttv' ? Boolean(prompt.trim()) : Boolean(image);

  return (
    <div className="gradient-border p-6 animate-slide-in space-y-5">
      <div className="flex flex-wrap gap-2">
        {[
          { id: 'ttv', label: 'TTV · Text → Video', hint: 'Prompt only; backend builds a placeholder frame then animates.' },
          { id: 'itv', label: 'ITV · Image → Video', hint: 'Upload a still; Stable Video Diffusion animates from your image.' },
        ].map((m) => (
          <button
            key={m.id}
            type="button"
            title={m.hint}
            onClick={() => {
              setMode(m.id);
              setError('');
            }}
            className={`px-3 py-2 rounded-xl text-xs font-mono border transition text-left ${
              mode === m.id
                ? 'bg-gradient-to-r from-neon-cyan/15 to-neon-purple/15 border-neon-cyan/50 text-white'
                : 'bg-white/5 border-white/10 text-gray-400 hover:text-white'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>
      <p className="text-xs text-gray-500 font-mono -mt-1">
        {mode === 'ttv'
          ? 'Best on large unified memory (e.g. 64 GB): SVD + PyTorch MPS can use tens of GB for decode.'
          : 'Use a clear, well-lit keyframe. Optional prompt nudges motion and mood.'}
      </p>
      {mode === 'ttv' && image && (
        <div className="flex items-center gap-2 text-xs font-mono text-amber-200/80">
          <span>A still is loaded but ignored in TTV.</span>
          <button type="button" onClick={clearImage} className="underline hover:text-amber-100">
            Remove
          </button>
        </div>
      )}

      <div>
        <label className="font-mono text-xs text-gray-400 uppercase tracking-wider">Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          placeholder={
            mode === 'ttv'
              ? 'A neon-lit Tokyo street at night, slow dolly, rain on asphalt…'
              : 'Optional: push in, parallax, golden hour glow, subtle wind in trees…'
          }
          className="mt-2 w-full bg-black/40 border border-white/10 rounded-xl p-3 font-mono text-sm focus:outline-none focus:border-neon-cyan/60 resize-none"
        />
        <div className="flex justify-end mt-2 gap-2">
          <button
            onClick={enhancePrompt}
            disabled={!prompt.trim() || enhancing || !ollamaReady}
            className="text-xs px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 hover:border-neon-purple/60 disabled:opacity-40 font-mono"
          >
            ✨ Enhance
          </button>
        </div>
      </div>

      <div>
        <label className="font-mono text-xs text-gray-400 uppercase tracking-wider">
          {mode === 'itv' ? 'Source image (required)' : 'Reference image (ITV only — switch mode above)'}
        </label>
        <div className="mt-2 flex items-center gap-3">
          <input
            ref={fileRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFile}
            disabled={mode === 'ttv'}
            className="block text-sm font-mono file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:bg-white/5 file:text-gray-200 hover:file:bg-white/10 disabled:opacity-40"
          />
          {image && mode === 'itv' && (
            <>
              <button onClick={describeImage} disabled={enhancing || !ollamaReady}
                className="text-xs px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 hover:border-neon-cyan/60 disabled:opacity-40 font-mono">
                👁 Describe
              </button>
              <button onClick={clearImage} className="text-xs px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 hover:border-rose-400/60 font-mono">
                ✕
              </button>
            </>
          )}
        </div>
        {imagePreview && mode === 'itv' && (
          <img src={imagePreview} alt="preview" className="mt-3 rounded-xl max-h-48 border border-white/10" />
        )}
      </div>

      <div>
        <label className="font-mono text-xs text-gray-400 uppercase tracking-wider">Style</label>
        <div className="mt-2 flex flex-wrap gap-2">
          {STYLES.map((s) => (
            <button
              key={s}
              onClick={() => setStyle(s)}
              className={`px-3 py-1.5 rounded-lg text-sm font-mono border transition ${
                style === s
                  ? 'bg-gradient-to-r from-neon-cyan/20 to-neon-purple/20 border-neon-purple/60 text-white'
                  : 'bg-white/5 border-white/10 text-gray-400 hover:text-white'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-5">
        <div>
          <div className="flex justify-between font-mono text-xs text-gray-400 uppercase tracking-wider">
            <span>Duration</span><span className="text-neon-cyan">{duration}s</span>
          </div>
          <input type="range" min="5" max="30" value={duration}
            onChange={(e) => setDuration(+e.target.value)} className="w-full mt-2" />
        </div>
        <div>
          <div className="flex justify-between font-mono text-xs text-gray-400 uppercase tracking-wider">
            <span>FPS</span><span className="text-neon-purple">{fps}</span>
          </div>
          <input type="range" min="24" max="60" value={fps}
            onChange={(e) => setFps(+e.target.value)} className="w-full mt-2" />
        </div>
      </div>

      {error && (
        <div className="px-3 py-2 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm font-mono">
          {error}
        </div>
      )}

      <button
        onClick={generate}
        disabled={loading || !canSubmit}
        className="btn-primary w-full py-3 rounded-xl text-sm uppercase tracking-widest font-mono"
      >
        {loading ? 'Submitting…' : mode === 'ttv' ? '⚡ Run TTV' : '🎬 Run ITV'}
      </button>
    </div>
  );
}
