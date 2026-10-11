import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, AlertOctagon, CheckCircle2 } from 'lucide-react';

export default function FaultPanel({ faultData }) {
  const faults = faultData?.faults_detected || [];
  const overall = faultData?.overall_health || 'healthy';

  return (
    <div className="card" style={{ height: '100%' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldAlert size={18} color={faults.length ? 'var(--rose-accent)' : 'var(--emerald-accent)'} />
          <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', fontWeight: 700 }}>
            Fault & Degradation Diagnostics
          </h2>
        </div>
        <span style={{
          fontSize: '0.72rem',
          fontFamily: 'var(--font-mono)',
          padding: '0.2rem 0.5rem',
          borderRadius: '4px',
          background: faults.length ? 'rgba(244, 63, 94, 0.1)' : 'rgba(16, 185, 129, 0.1)',
          color: faults.length ? 'var(--rose-accent)' : 'var(--emerald-accent)',
          border: `1px solid ${faults.length ? 'rgba(244, 63, 94, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
        }}>
          {faults.length} ACTIVE ANOMALIES
        </span>
      </div>

      {faults.length === 0 ? (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2.5rem 1rem',
          textAlign: 'center',
          background: 'rgba(16, 185, 129, 0.03)',
          border: '1px solid rgba(16, 185, 129, 0.15)',
          borderRadius: '10px',
        }}>
          <CheckCircle2 size={40} color="var(--emerald-accent)" style={{ marginBottom: '0.75rem' }} />
          <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.05rem', color: 'var(--emerald-accent)', marginBottom: '0.3rem' }}>
            Battery System In Nominal Operating Region
          </h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', maxWidth: '340px' }}>
            All ECM equivalent circuit impedance parameters and thermal response fall within standard safety thresholds.
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {faults.map((fault, idx) => {
            const isCrit = fault.severity === 'critical';
            return (
              <div
                key={idx}
                style={{
                  background: isCrit ? 'rgba(244, 63, 94, 0.08)' : 'rgba(245, 158, 11, 0.08)',
                  border: `1px solid ${isCrit ? 'rgba(244, 63, 94, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
                  borderRadius: '10px',
                  padding: '1rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.4rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    {isCrit ? <AlertOctagon size={16} color="var(--rose-accent)" /> : <AlertTriangle size={16} color="var(--amber-accent)" />}
                    <span style={{ fontWeight: 700, fontSize: '0.85rem', color: isCrit ? 'var(--rose-accent)' : 'var(--amber-accent)' }}>
                      {fault.type?.replace(/_/g, ' ').toUpperCase()}
                    </span>
                  </div>
                  <span className={`badge ${isCrit ? 'badge-critical' : 'badge-warning'}`}>
                    {fault.severity}
                  </span>
                </div>

                <div style={{ fontSize: '0.775rem', fontFamily: 'var(--font-mono)', color: 'var(--text-main)', marginBottom: '0.4rem' }}>
                  Param: <strong>{fault.parameter}</strong> | Current: <strong>{fault.current_value}</strong> | Baseline: {fault.baseline_value} (
                  <span style={{ color: isCrit ? 'var(--rose-accent)' : 'var(--amber-accent)' }}>
                    {fault.delta_percent > 0 ? `+${fault.delta_percent}%` : `${fault.delta_percent}%`}
                  </span>)
                </div>

                <p style={{ fontSize: '0.785rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                  {fault.recommendation}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
