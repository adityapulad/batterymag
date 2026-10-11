import React from 'react';
import { Cpu, CheckCircle2, SlidersHorizontal, Activity } from 'lucide-react';

export default function ECMParameters({ ecmData, sohData }) {
  const params = ecmData?.parameters || sohData?.ecm_params || {
    R0: 0.048,
    R1: 0.022,
    C1: 1520.0,
    tau1: 33.4,
  };

  const metrics = ecmData?.metrics || {
    RMSE_mV: 12.4,
    R2: 0.981,
    MAE_mV: 9.1,
  };

  // Compare to healthy nominal 18650 cell
  const r0_baseline = 0.045;
  const r0_delta = ((params.R0 - r0_baseline) / r0_baseline) * 100;

  return (
    <div className="card" style={{ height: '100%' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Cpu size={18} color="var(--blue-accent)" />
          <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', fontWeight: 700 }}>
            ECM Physics Circuit Extraction
          </h2>
        </div>
        <span style={{
          fontSize: '0.72rem',
          fontFamily: 'var(--font-mono)',
          background: 'rgba(79, 172, 254, 0.1)',
          color: 'var(--blue-accent)',
          padding: '0.2rem 0.5rem',
          borderRadius: '4px',
          border: '1px solid rgba(79, 172, 254, 0.3)',
        }}>
          1-RC THEVENIN MODEL
        </span>
      </div>

      {/* Circuit Schematic Visual */}
      <div style={{
        background: 'rgba(7, 9, 14, 0.6)',
        border: '1px dashed var(--border-subtle)',
        borderRadius: '10px',
        padding: '1rem',
        marginBottom: '1.25rem',
        textAlign: 'center',
      }}>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '0.5rem',
          flexWrap: 'wrap',
        }}>
          <span style={{ color: 'var(--emerald-accent)' }}>[ OCV(SoC) ]</span>
          <span>───</span>
          <span style={{ color: 'var(--cyan-primary)', borderBottom: '1px solid var(--cyan-primary)', paddingBottom: '2px' }}>
            R₀ ({params.R0.toFixed(4)} Ω)
          </span>
          <span>───</span>
          <span style={{
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '6px',
            padding: '0.25rem 0.6rem',
            color: 'var(--blue-accent)'
          }}>
            [ R₁ // C₁ ]
          </span>
          <span>───</span>
          <span style={{ color: 'var(--amber-accent)' }}>V_term</span>
        </div>
      </div>

      {/* Parameter Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '0.85rem',
        marginBottom: '1.25rem',
      }}>
        {/* R0 */}
        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Ohmic Resistance (R₀)</div>
          <div style={{ fontFamily: 'var(--font-heading)', fontSize: '1.3rem', fontWeight: 700, color: 'var(--cyan-primary)', margin: '0.2rem 0' }}>
            {params.R0.toFixed(4)} <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Ω</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: r0_delta > 20 ? 'var(--rose-accent)' : 'var(--emerald-accent)' }}>
            {r0_delta >= 0 ? `+${r0_delta.toFixed(1)}%` : `${r0_delta.toFixed(1)}%`} vs nominal
          </div>
        </div>

        {/* R1 */}
        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Charge-Transfer (R₁)</div>
          <div style={{ fontFamily: 'var(--font-heading)', fontSize: '1.3rem', fontWeight: 700, color: 'var(--blue-accent)', margin: '0.2rem 0' }}>
            {params.R1?.toFixed(4) || '0.0210'} <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Ω</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-subtle)' }}>Kinetics impedance</div>
        </div>

        {/* C1 */}
        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Capacitance (C₁)</div>
          <div style={{ fontFamily: 'var(--font-heading)', fontSize: '1.3rem', fontWeight: 700, color: 'var(--purple-accent)', margin: '0.2rem 0' }}>
            {Math.round(params.C1 || 1500)} <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>F</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-subtle)' }}>Polarization double-layer</div>
        </div>

        {/* tau1 */}
        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Time Constant (τ₁)</div>
          <div style={{ fontFamily: 'var(--font-heading)', fontSize: '1.3rem', fontWeight: 700, color: 'var(--amber-accent)', margin: '0.2rem 0' }}>
            {(params.tau1 || 32.5).toFixed(1)} <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>s</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-subtle)' }}>Relaxation latency</div>
        </div>
      </div>

      {/* Fit metrics bar */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-around',
        padding: '0.75rem',
        background: 'rgba(0, 0, 0, 0.25)',
        borderRadius: '8px',
        border: '1px solid var(--border-subtle)',
        fontSize: '0.775rem',
      }}>
        <div><span style={{ color: 'var(--text-muted)' }}>RMSE: </span><strong style={{ color: 'var(--cyan-primary)' }}>{metrics.RMSE_mV} mV</strong></div>
        <div><span style={{ color: 'var(--text-muted)' }}>R² Fit: </span><strong style={{ color: 'var(--emerald-accent)' }}>{metrics.R2}</strong></div>
        <div><span style={{ color: 'var(--text-muted)' }}>MAE: </span><strong style={{ color: 'var(--amber-accent)' }}>{metrics.MAE_mV} mV</strong></div>
      </div>
    </div>
  );
}
