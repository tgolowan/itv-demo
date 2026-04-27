import { useEffect, useState } from 'react';

const fmtGB = (b) => (b / 1024 ** 3).toFixed(1);

export default function GPUMonitor() {
  const [system, setSystem] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [s, h] = await Promise.all([
          fetch('/api/system').then((r) => r.json()),
          fetch('/api/health').then((r) => r.json()),
        ]);
        setSystem(s);
        setHealth(h);
      } catch {}
    };
    fetchAll();
    const i = setInterval(fetchAll, 5000);
    return () => clearInterval(i);
  }, []);

  if (!system) {
    return <div className="gradient-border p-5 text-gray-500 font-mono text-sm">Loading system…</div>;
  }

  const usedRam = system.totalMemory - system.freeMemory;
  const ramPct = (usedRam / system.totalMemory) * 100;
  const heapPct = health ? (health.memory.heapUsed / health.memory.heapTotal) * 100 : 0;

  return (
    <div className="gradient-border p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-display text-lg">System Monitor</h3>
        <span className="font-mono text-xs text-emerald-400">● live</span>
      </div>

      <div className="grid grid-cols-2 gap-3 font-mono text-xs">
        <Stat label="Platform" value={`${system.platform}/${system.arch}`} />
        <Stat label="CPU Cores" value={system.cpus} />
        <Stat label="RAM Total" value={`${fmtGB(system.totalMemory)} GB`} />
        <Stat label="RAM Free" value={`${fmtGB(system.freeMemory)} GB`} />
      </div>

      <div>
        <Bar label="System RAM" pct={ramPct} from="from-neon-cyan" to="to-neon-purple"
             right={`${fmtGB(usedRam)} / ${fmtGB(system.totalMemory)} GB`} />
      </div>

      {health && (
        <div>
          <Bar label="Server Heap" pct={heapPct} from="from-neon-purple" to="to-neon-orange"
               right={`${fmtGB(health.memory.heapUsed)} / ${fmtGB(health.memory.heapTotal)} GB`} />
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg bg-black/30 border border-white/5 p-2">
      <div className="text-gray-500 uppercase tracking-wider">{label}</div>
      <div className="text-gray-100 mt-1">{value}</div>
    </div>
  );
}

function Bar({ label, pct, from, to, right }) {
  return (
    <div>
      <div className="flex justify-between font-mono text-xs">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-300">{right}</span>
      </div>
      <div className="mt-1 h-2 rounded-full bg-white/5 overflow-hidden">
        <div className={`h-full bg-gradient-to-r ${from} ${to} transition-all`} style={{ width: `${Math.min(100, pct)}%` }} />
      </div>
    </div>
  );
}
