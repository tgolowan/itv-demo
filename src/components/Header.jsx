export default function Header({ ollamaStatus }) {
  const online = ollamaStatus.status === 'online';
  const dotColor = online ? 'bg-emerald-400' : ollamaStatus.status === 'checking' ? 'bg-yellow-400' : 'bg-rose-500';
  const label = online ? 'Ollama Online' : ollamaStatus.status === 'checking' ? 'Connecting…' : 'Ollama Offline';

  return (
    <header className="border-b border-white/5 backdrop-blur-md bg-gray-950/60 sticky top-0 z-20">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-neon-cyan via-neon-purple to-neon-orange animate-pulse-glow" />
          <div>
            <h1 className="font-display text-xl font-semibold gradient-text">LocalVideoGen</h1>
            <p className="text-xs text-gray-500 font-mono">
              local TTV &amp; ITV · Mac uses CPU SVD by default (stable) · Ollama LLMs
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10">
            <span className={`w-2 h-2 rounded-full ${dotColor} ${online ? 'animate-pulse' : ''}`} />
            <span className="text-gray-300">{label}</span>
          </div>
          {ollamaStatus.models?.length > 0 && (
            <div className="hidden md:flex items-center gap-1 text-gray-500">
              <span>models:</span>
              <span className="text-neon-cyan">{ollamaStatus.models.length}</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
