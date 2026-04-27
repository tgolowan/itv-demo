import { useState } from 'react';

const STATUS_COLORS = {
  queued: 'text-yellow-300 bg-yellow-300/10 border-yellow-300/30',
  running: 'text-neon-cyan bg-neon-cyan/10 border-neon-cyan/30',
  completed: 'text-emerald-300 bg-emerald-300/10 border-emerald-300/30',
  failed: 'text-rose-300 bg-rose-300/10 border-rose-300/30',
};

export default function JobsList({ jobs, onDelete }) {
  const [expanded, setExpanded] = useState(null);

  if (!jobs.length) {
    return (
      <div className="gradient-border p-10 text-center text-gray-500 font-mono">
        No jobs yet. Generate your first video on the Generate tab.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {jobs.map((job) => {
        const isOpen = expanded === job.job_id;
        const cls = STATUS_COLORS[job.status] || STATUS_COLORS.queued;
        return (
          <div key={job.job_id} className="gradient-border p-4 animate-slide-in">
            <div className="flex items-center justify-between gap-3">
              <button onClick={() => setExpanded(isOpen ? null : job.job_id)} className="flex-1 text-left">
                <div className="flex items-center gap-3">
                  <span className={`text-xs font-mono px-2 py-1 rounded border ${cls}`}>{job.status}</span>
                  <span className="font-mono text-xs text-gray-500">{job.job_id.slice(0, 8)}</span>
                  <span className="text-sm text-gray-200 truncate">{job.prompt || '(image only)'}</span>
                </div>
                <div className="mt-2 h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-neon-cyan via-neon-purple to-neon-orange transition-all"
                    style={{ width: `${job.progress || 0}%` }}
                  />
                </div>
              </button>
              <div className="flex gap-2">
                {job.status === 'completed' && (
                  <a
                    href={`/api/jobs/${job.job_id}/download`}
                    className="text-xs px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 hover:border-neon-cyan/60 font-mono"
                  >
                    ⬇ Download
                  </a>
                )}
                <button
                  onClick={() => onDelete(job.job_id)}
                  className="text-xs px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 hover:border-rose-400/60 font-mono"
                >
                  ✕
                </button>
              </div>
            </div>

            {isOpen && (
              <div className="mt-4 pt-4 border-t border-white/5 grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs">
                <Meta label="Duration" value={`${job.duration || '-'}s`} />
                <Meta label="FPS" value={job.fps || '-'} />
                <Meta label="Style" value={job.style || '-'} />
                <Meta label="Progress" value={`${job.progress || 0}%`} />
                <div className="col-span-full">
                  <div className="text-gray-500 uppercase tracking-wider">Prompt</div>
                  <div className="text-gray-200 mt-1 whitespace-pre-wrap">{job.prompt || '—'}</div>
                </div>
                {job.error && (
                  <div className="col-span-full text-rose-300">{job.error}</div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Meta({ label, value }) {
  return (
    <div className="rounded-lg bg-black/30 border border-white/5 p-2">
      <div className="text-gray-500 uppercase tracking-wider">{label}</div>
      <div className="text-gray-100 mt-1">{value}</div>
    </div>
  );
}
