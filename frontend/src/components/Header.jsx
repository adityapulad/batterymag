import React from 'react';
import { Activity, Zap, Cpu, RefreshCw, Upload, ShieldCheck, AlertCircle } from 'lucide-react';

export default function Header({
  selectedProfile,
  onSelectProfile,
  onRefresh,
  onOpenUpload,
  loading,
  backendStatus,
}) {
  const isOnline = backendStatus?.status === 'healthy';

  return (
    <header style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '1.25rem 0',
      borderBottom: '1px solid var(--border-subtle)',
      marginBottom: '2rem',
      flexWrap: 'wrap',
      gap: '1rem',
    }}>
      {/* Brand & Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
        <div style={{
          background: 'linear-gradient(135deg, rgba(0, 242, 254, 0.15), rgba(79, 172, 254, 0.25))',
          border: '1px solid var(--cyan-primary)',
          borderRadius: '12px',
          padding: '0.65rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 20px var(--cyan-glow)',
        }}>
          <Zap size={26} color="var(--cyan-primary)" />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <h1 style={{
              fontFamily: 'var(--font-heading)',
              fontSize: '1.5rem',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              background: 'linear-gradient(to right, #ffffff, #94a3b8)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              BATTERY INTELLIGENCE
            </h1>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.68rem',
              background: 'rgba(0, 242, 254, 0.1)',
              color: 'var(--cyan-primary)',
              padding: '0.15rem 0.5rem',
              borderRadius: '4px',
              border: '1px solid rgba(0, 242, 254, 0.3)',
              fontWeight: 600,
            }}>
              PHYSICS+AI
            </span>
          </div>
          <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)' }}>
            Real-Time State of Health, Charge, RUL & Equivalent Circuit Estimation
          </p>
        </div>
      </div>

      {/* Profile Selector & Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
        {/* Backend connectivity indicator */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.45rem',
          fontSize: '0.775rem',
          fontFamily: 'var(--font-mono)',
          padding: '0.4rem 0.75rem',
          background: 'rgba(255, 255, 255, 0.04)',
          borderRadius: '8px',
          border: '1px solid var(--border-subtle)',
        }}>
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: isOnline ? 'var(--emerald-accent)' : 'var(--amber-accent)',
            boxShadow: isOnline ? '0 0 8px var(--emerald-accent)' : 'none',
          }} />
          <span style={{ color: isOnline ? 'var(--emerald-accent)' : 'var(--amber-accent)' }}>
            {isOnline ? 'API ONLINE' : 'STANDALONE'}
          </span>
        </div>

        {/* Profile Preset Switcher */}
        <div style={{
          display: 'flex',
          background: 'rgba(0, 0, 0, 0.35)',
          padding: '0.2rem',
          borderRadius: '10px',
          border: '1px solid var(--border-subtle)',
        }}>
          {[
            { id: 'healthy', label: 'Healthy B0005', cycle: '24 cyc' },
            { id: 'moderate', label: 'Moderate B0006', cycle: '95 cyc' },
            { id: 'degraded', label: 'Degraded B0007', cycle: '168 cyc' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => onSelectProfile(item.id)}
              style={{
                background: selectedProfile === item.id ? 'rgba(0, 242, 254, 0.15)' : 'transparent',
                color: selectedProfile === item.id ? 'var(--cyan-primary)' : 'var(--text-muted)',
                border: selectedProfile === item.id ? '1px solid rgba(0, 242, 254, 0.4)' : '1px solid transparent',
                borderRadius: '8px',
                padding: '0.4rem 0.75rem',
                fontSize: '0.775rem',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* Upload Custom CSV button */}
        <button
          className="btn btn-secondary"
          onClick={onOpenUpload}
          style={{ padding: '0.45rem 0.85rem' }}
          title="Upload real cycling telemetry CSV"
        >
          <Upload size={15} />
          <span>Ingest CSV</span>
        </button>

        {/* Trigger re-inference */}
        <button
          className="btn btn-primary"
          onClick={onRefresh}
          disabled={loading}
          style={{ padding: '0.45rem 1rem' }}
        >
          <RefreshCw size={15} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          <span>{loading ? 'Evaluating...' : 'Run Pipeline'}</span>
        </button>
      </div>

      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
    </header>
  );
}
