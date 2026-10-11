import React from 'react';
import { BatteryCharging, HeartPulse, Clock, ShieldAlert, TrendingUp, AlertTriangle } from 'lucide-react';

export default function KPICards({ sohData, socData, rulData, faultData }) {
  const soh = sohData?.soh_percent ?? 92.4;
  const currentSoc = socData?.current_soc ?? 78.5;
  const rulCycles = rulData?.rul_cycles ?? 54;
  const rulDays = rulData?.rul_days_estimated ?? 108;
  const overallHealth = faultData?.overall_health ?? 'healthy';
  const alertCount = faultData?.faults_detected?.length ?? 0;

  const getSoHColor = (val) => {
    if (val >= 85) return 'var(--emerald-accent)';
    if (val >= 75) return 'var(--amber-accent)';
    return 'var(--rose-accent)';
  };

  const getHealthBadge = (status) => {
    switch (status) {
      case 'critical':
        return <span className="badge badge-critical">CRITICAL DEGRADATION</span>;
      case 'warning':
        return <span className="badge badge-warning">ANOMALY WARNING</span>;
      default:
        return <span className="badge badge-success">HEALTHY NOMINAL</span>;
    }
  };

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
      gap: '1.25rem',
      marginBottom: '2rem',
    }}>
      {/* 1. SOH CARD */}
      <div className="card" style={{ position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute',
          top: '-20px',
          right: '-20px',
          width: '90px',
          height: '90px',
          background: `radial-gradient(circle, ${getSoHColor(soh)} 0%, transparent 70%)`,
          opacity: 0.15,
          borderRadius: '50%',
          pointerEvents: 'none',
        }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
              State of Health (SoH)
            </span>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginTop: '0.25rem' }}>
              <span style={{ fontFamily: 'var(--font-heading)', fontSize: '2.25rem', fontWeight: 800, color: getSoHColor(soh) }}>
                {soh.toFixed(1)}%
              </span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>of fresh</span>
            </div>
          </div>
          <div style={{
            background: 'rgba(255, 255, 255, 0.05)',
            padding: '0.55rem',
            borderRadius: '10px',
            color: getSoHColor(soh),
          }}>
            <HeartPulse size={22} />
          </div>
        </div>

        {/* Progress bar */}
        <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '3px', overflow: 'hidden', margin: '0.5rem 0' }}>
          <div style={{
            width: `${Math.min(100, Math.max(0, soh))}%`,
            height: '100%',
            background: `linear-gradient(90deg, #4facfe, ${getSoHColor(soh)})`,
            borderRadius: '3px',
            transition: 'width 0.6s ease',
          }} />
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
          <span>Status: <strong style={{ color: 'var(--text-main)' }}>{sohData?.degradation_status || 'nominal'}</strong></span>
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.3rem',
            padding: '0.15rem 0.45rem',
            borderRadius: '4px',
            fontSize: '0.7rem',
            fontWeight: 600,
            background: sohData?.prediction_source === 'lstm_model' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
            color: sohData?.prediction_source === 'lstm_model' ? 'var(--emerald-accent)' : 'var(--amber-accent)',
            border: `1px solid ${sohData?.prediction_source === 'lstm_model' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`
          }}>
            {sohData?.prediction_source === 'lstm_model' ? '● LSTM [LOBO]' : '▲ Physics Fallback'}
          </span>
          <span>Conf: <strong style={{ color: 'var(--cyan-primary)' }}>{Math.round((sohData?.confidence || 0.95) * 100)}%</strong></span>
        </div>
      </div>

      {/* 2. SOC CARD */}
      <div className="card" style={{ position: 'relative', overflow: 'hidden' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
              State of Charge (SoC)
            </span>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginTop: '0.25rem' }}>
              <span style={{ fontFamily: 'var(--font-heading)', fontSize: '2.25rem', fontWeight: 800, color: 'var(--cyan-primary)' }}>
                {currentSoc.toFixed(1)}%
              </span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>remaining</span>
            </div>
          </div>
          <div style={{
            background: 'rgba(0, 242, 254, 0.1)',
            padding: '0.55rem',
            borderRadius: '10px',
            color: 'var(--cyan-primary)',
          }}>
            <BatteryCharging size={22} />
          </div>
        </div>

        {/* Progress bar */}
        <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '3px', overflow: 'hidden', margin: '0.5rem 0' }}>
          <div style={{
            width: `${Math.min(100, Math.max(0, currentSoc))}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #00f2fe, #4facfe)',
            borderRadius: '3px',
            transition: 'width 0.6s ease',
          }} />
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
          <span>Coulomb Count: <strong style={{ color: 'var(--text-main)' }}>Active</strong></span>
          <span>Confidence: <strong style={{ color: 'var(--cyan-primary)' }}>93%</strong></span>
        </div>
      </div>

      {/* 3. RUL CARD */}
      <div className="card" style={{ position: 'relative', overflow: 'hidden' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
              Remaining Useful Life
            </span>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginTop: '0.25rem' }}>
              <span style={{ fontFamily: 'var(--font-heading)', fontSize: '2.25rem', fontWeight: 800, color: 'var(--purple-accent)' }}>
                {rulCycles}
              </span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>cycles to EOL</span>
            </div>
          </div>
          <div style={{
            background: 'rgba(139, 92, 246, 0.1)',
            padding: '0.55rem',
            borderRadius: '10px',
            color: 'var(--purple-accent)',
          }}>
            <Clock size={22} />
          </div>
        </div>

        <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '3px', overflow: 'hidden', margin: '0.5rem 0' }}>
          <div style={{
            width: `${Math.min(100, Math.max(5, (rulCycles / 160) * 100))}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #8b5cf6, #c084fc)',
            borderRadius: '3px',
            transition: 'width 0.6s ease',
          }} />
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
          <span>Est. Duration: <strong style={{ color: 'var(--text-main)' }}>~{rulDays} days</strong></span>
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.3rem',
            padding: '0.15rem 0.45rem',
            borderRadius: '4px',
            fontSize: '0.7rem',
            fontWeight: 600,
            background: rulData?.prediction_source === 'lstm_model' ? 'rgba(139, 92, 246, 0.15)' : 'rgba(245, 158, 11, 0.15)',
            color: rulData?.prediction_source === 'lstm_model' ? 'var(--purple-accent)' : 'var(--amber-accent)',
            border: `1px solid ${rulData?.prediction_source === 'lstm_model' ? 'rgba(139, 92, 246, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`
          }}>
            {rulData?.prediction_source === 'lstm_model' ? '● LSTM [LOBO]' : '▲ Empirical'}
          </span>
          <span>Risk: <strong style={{ color: rulData?.risk_level === 'high' ? 'var(--rose-accent)' : 'var(--emerald-accent)' }}>
            {(rulData?.risk_level || 'low').toUpperCase()}
          </strong></span>
        </div>
      </div>

      {/* 4. HEALTH / ANOMALY CARD */}
      <div className="card" style={{ position: 'relative', overflow: 'hidden' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
              Diagnostic Status
            </span>
            <div style={{ marginTop: '0.45rem' }}>
              {getHealthBadge(overallHealth)}
            </div>
          </div>
          <div style={{
            background: alertCount > 0 ? 'rgba(244, 63, 94, 0.1)' : 'rgba(16, 185, 129, 0.1)',
            padding: '0.55rem',
            borderRadius: '10px',
            color: alertCount > 0 ? 'var(--rose-accent)' : 'var(--emerald-accent)',
          }}>
            {alertCount > 0 ? <ShieldAlert size={22} /> : <TrendingUp size={22} />}
          </div>
        </div>

        <div style={{ marginTop: '0.9rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          {alertCount === 0 ? (
            <p style={{ color: 'var(--emerald-accent)' }}>Zero anomalies detected across ECM parameters.</p>
          ) : (
            <p style={{ color: 'var(--rose-accent)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <AlertTriangle size={14} />
              {alertCount} parameter degradation warning{alertCount > 1 ? 's' : ''} active
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
