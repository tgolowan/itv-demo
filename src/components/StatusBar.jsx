import { useEffect, useState } from 'react';

export default function StatusBar() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const i = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(i);
  }, []);
  return (
    <footer className="border-t border-white/5 bg-gray-950/60 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between font-mono text-xs text-gray-500">
        <span>localvideogen <span className="text-neon-cyan">v1.0.0</span></span>
        <span>{now.toLocaleTimeString()}</span>
        <span className="hidden md:inline">macOS · Apple Silicon · Metal</span>
      </div>
    </footer>
  );
}
