import { useEffect, useState, useCallback, useRef } from 'react';
import Header from './components/Header.jsx';
import InputPanel from './components/InputPanel.jsx';
import JobsList from './components/JobsList.jsx';
import StatusBar from './components/StatusBar.jsx';
import GPUMonitor from './components/GPUMonitor.jsx';
import LLMChat from './components/LLMChat.jsx';

export default function App() {
  const [ollamaStatus, setOllamaStatus] = useState({ status: 'checking', models: [], required: [], setup: false });
  const [jobs, setJobs] = useState([]);
  const [activeTab, setActiveTab] = useState('generate'); // generate | llm | jobs
  const pollingRef = useRef(null);

  const fetchOllama = useCallback(async () => {
    try {
      const res = await fetch('/api/ollama/status');
      const data = await res.json();
      setOllamaStatus(data);
    } catch {
      setOllamaStatus({ status: 'offline', models: [], required: ['llava', 'mistral'], setup: false });
    }
  }, []);

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      setJobs(data.jobs || []);
    } catch {}
  }, []);

  useEffect(() => {
    fetchOllama();
    fetchJobs();
    const o = setInterval(fetchOllama, 10000);
    const j = setInterval(fetchJobs, 2000);
    pollingRef.current = { o, j };
    return () => { clearInterval(o); clearInterval(j); };
  }, [fetchOllama, fetchJobs]);

  const onJobCreated = (job) => {
    setJobs((prev) => [job, ...prev]);
    setActiveTab('jobs');
  };

  const deleteJob = async (id) => {
    await fetch(`/api/jobs/${id}`, { method: 'DELETE' });
    fetchJobs();
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header ollamaStatus={ollamaStatus} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
        <section className="lg:col-span-2 space-y-6">
          <div className="flex flex-wrap gap-2 font-mono text-sm">
            {[
              { id: 'generate', label: '⚡ TTV / ITV' },
              { id: 'llm', label: '💬 Local LLM' },
              { id: 'jobs', label: `📼 Jobs (${jobs.length})` },
            ].map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setActiveTab(t.id)}
                className={`px-4 py-2 rounded-lg transition ${
                  activeTab === t.id
                    ? 'bg-white/10 text-white border border-white/10'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {activeTab === 'generate' && (
            <InputPanel onJobCreated={onJobCreated} ollamaReady={ollamaStatus.status === 'online'} />
          )}
          {activeTab === 'llm' && (
            <LLMChat
              ollamaReady={ollamaStatus.status === 'online'}
              models={ollamaStatus.models || []}
            />
          )}
          {activeTab === 'jobs' && <JobsList jobs={jobs} onDelete={deleteJob} />}
        </section>

        <aside className="space-y-6">
          <GPUMonitor />
          <div className="gradient-border p-5">
            <h3 className="font-display text-lg mb-2">Tips</h3>
            <ul className="text-sm text-gray-400 space-y-2 font-mono">
              <li>• Mac: SVD runs on CPU by default (slow, stable — no Metal freeze/reboot)</li>
              <li>• Shorter duration / fps = faster jobs; first clip ~6s @ 24fps is a good smoke test</li>
              <li>• TTV: rich prompt; ITV: strong still + optional motion hint</li>
              <li>• GPU: set SVD_USE_MPS=1 for server — faster, riskier on memory</li>
            </ul>
          </div>
        </aside>
      </main>

      <StatusBar />
    </div>
  );
}
