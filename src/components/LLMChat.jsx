import { useEffect, useRef, useState } from 'react';

export default function LLMChat({ ollamaReady, models }) {
  const [model, setModel] = useState('');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const bottomRef = useRef(null);

  useEffect(() => {
    if (!model && models?.length) {
      const pick =
        models.find((n) => /llama|mistral|qwen|gemma|phi|deepseek/i.test(n)) || models[0];
      setModel(pick);
    }
  }, [models, model]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

  const send = async () => {
    const text = input.trim();
    if (!text || !model) return;
    setError('');
    const next = [...messages, { role: 'user', content: text }];
    setMessages(next);
    setInput('');
    setSending(true);
    try {
      const res = await fetch('/api/ollama/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages: next }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Chat failed');
      const reply = (data.content || '').trim();
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }]);
    } catch (e) {
      setError(e.message);
      setMessages((prev) => prev.slice(0, -1));
      setInput(text);
    }
    setSending(false);
  };

  const clearChat = () => {
    setMessages([]);
    setError('');
  };

  return (
    <div className="gradient-border p-6 animate-slide-in space-y-4 flex flex-col min-h-[420px] max-h-[70vh]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-display text-lg text-white">Local LLM</h2>
          <p className="text-xs text-gray-500 font-mono mt-0.5">
            Ollama · runs on your Mac · no cloud
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={!ollamaReady || !models?.length}
            className="bg-black/50 border border-white/10 rounded-lg px-3 py-2 text-sm font-mono text-gray-200 max-w-[220px]"
          >
            {!models?.length ? (
              <option value="">No models</option>
            ) : (
              models.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))
            )}
          </select>
          <button
            type="button"
            onClick={clearChat}
            className="text-xs px-3 py-2 rounded-lg bg-white/5 border border-white/10 hover:border-rose-400/50 font-mono text-gray-300"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin space-y-3 pr-1 min-h-[200px]">
        {messages.length === 0 && (
          <p className="text-sm text-gray-500 font-mono">
            Ask for shot ideas, rewrite prompts, or debug your pipeline. Start Ollama and pull a
            model (e.g. <span className="text-neon-cyan">ollama pull llama3.2</span>).
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`rounded-xl px-3 py-2 text-sm font-mono border ${
              m.role === 'user'
                ? 'bg-neon-cyan/5 border-neon-cyan/20 text-gray-100 ml-4'
                : 'bg-white/[0.03] border-white/10 text-gray-300 mr-4'
            }`}
          >
            <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">
              {m.role}
            </div>
            <div className="whitespace-pre-wrap break-words">{m.content}</div>
          </div>
        ))}
        {sending && (
          <div className="text-xs text-gray-500 font-mono animate-pulse">Thinking…</div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && (
        <div className="px-3 py-2 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm font-mono">
          {error}
        </div>
      )}

      <div className="flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (!sending && ollamaReady) send();
            }
          }}
          rows={2}
          disabled={!ollamaReady || sending}
          placeholder={ollamaReady ? 'Message… (Enter to send, Shift+Enter newline)' : 'Start Ollama to chat…'}
          className="flex-1 bg-black/40 border border-white/10 rounded-xl p-3 font-mono text-sm focus:outline-none focus:border-neon-purple/50 resize-none disabled:opacity-40"
        />
        <button
          type="button"
          onClick={send}
          disabled={!ollamaReady || sending || !input.trim() || !model}
          className="btn-primary self-end px-4 py-2 rounded-xl text-xs font-mono shrink-0"
        >
          Send
        </button>
      </div>
    </div>
  );
}
