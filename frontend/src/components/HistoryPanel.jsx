import React from 'react';
import { History, Download, RefreshCw, Calendar, Tag, Layers, CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react';

export default function HistoryPanel({ history, onRefresh, loading }) {
  const exportToCSV = () => {
    if (!history || history.length === 0) return;
    const headers = ["ID", "Battery ID", "Endpoint", "SoH (%)", "SoC (%)", "RUL Cycles", "Health", "Model Version", "Timestamp"];
    const rows = history.map(item => [
      item.id,
      item.battery_id,
      item.endpoint,
      item.soh_percent !== null ? item.soh_percent : "",
      item.current_soc !== null ? item.current_soc : "",
      item.rul_cycles !== null ? item.rul_cycles : "",
      item.overall_health || "",
      item.model_version || "",
      item.timestamp || "",
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `bms_prediction_history_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getHealthBadge = (health) => {
    if (health === 'critical') return <span className="badge badge-critical">Critical</span>;
    if (health === 'warning') return <span className="badge badge-warning">Warning</span>;
    return <span className="badge badge-success">Nominal</span>;
  };

  return (
    <div className="card" style={{ marginTop: '2rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <History size={20} color="var(--purple-accent)" />
          <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.15rem', fontWeight: 700 }}>
            Prediction History & Audit Log
          </h2>
          <span style={{
            fontSize: '0.75rem',
            fontFamily: 'var(--font-mono)',
            padding: '0.2rem 0.5rem',
            borderRadius: '4px',
            background: 'rgba(139, 92, 246, 0.1)',
            color: 'var(--purple-accent)',
            border: '1px solid rgba(139, 92, 246, 0.3)',
          }}>
            {history.length} RECORDS
          </span>
        </div>

        <div style={{ display: 'flex', gap: '0.6rem' }}>
          <button
            className="btn btn-secondary"
            onClick={exportToCSV}
            disabled={history.length === 0}
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
            title="Export audit log to CSV"
          >
            <Download size={14} />
            <span>Export CSV</span>
          </button>

          <button
            className="btn btn-secondary"
            onClick={onRefresh}
            disabled={loading}
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
            title="Refresh history"
          >
            <RefreshCw size={14} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Table Container */}
      <div style={{ overflowX: 'auto', border: '1px solid var(--border-subtle)', borderRadius: '8px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(0, 0, 0, 0.4)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
              <th style={{ padding: '0.75rem 1rem' }}>ID</th>
              <th style={{ padding: '0.75rem 1rem' }}>Battery ID</th>
              <th style={{ padding: '0.75rem 1rem' }}>Model / Endpoint</th>
              <th style={{ padding: '0.75rem 1rem' }}>SoH %</th>
              <th style={{ padding: '0.75rem 1rem' }}>Current SoC</th>
              <th style={{ padding: '0.75rem 1rem' }}>RUL Cycles</th>
              <th style={{ padding: '0.75rem 1rem' }}>Model Version</th>
              <th style={{ padding: '0.75rem 1rem' }}>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-subtle)' }}>
                  No historical prediction runs recorded yet. Execute inference to generate records.
                </td>
              </tr>
            ) : (
              history.map((row) => (
                <tr
                  key={row.id}
                  style={{
                    borderBottom: '1px solid var(--border-subtle)',
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ padding: '0.65rem 1rem', fontFamily: 'var(--font-mono)', color: 'var(--text-subtle)' }}>
                    #{row.id}
                  </td>
                  <td style={{ padding: '0.65rem 1rem', fontWeight: 600, color: 'var(--text-main)' }}>
                    {row.battery_id}
                  </td>
                  <td style={{ padding: '0.65rem 1rem' }}>
                    <span style={{
                      textTransform: 'uppercase',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.7rem',
                      padding: '0.15rem 0.45rem',
                      borderRadius: '4px',
                      background: 'rgba(0, 242, 254, 0.08)',
                      color: 'var(--cyan-primary)',
                      border: '1px solid rgba(0, 242, 254, 0.2)',
                    }}>
                      {row.endpoint}
                    </span>
                  </td>
                  <td style={{ padding: '0.65rem 1rem', fontFamily: 'var(--font-mono)' }}>
                    {row.soh_percent !== null ? (
                      <span style={{ color: row.soh_percent >= 80 ? 'var(--emerald-accent)' : 'var(--amber-accent)', fontWeight: 600 }}>
                        {row.soh_percent.toFixed(1)}%
                      </span>
                    ) : '—'}
                  </td>
                  <td style={{ padding: '0.65rem 1rem', fontFamily: 'var(--font-mono)' }}>
                    {row.current_soc !== null ? `${row.current_soc.toFixed(1)}%` : '—'}
                  </td>
                  <td style={{ padding: '0.65rem 1rem', fontFamily: 'var(--font-mono)', color: 'var(--purple-accent)' }}>
                    {row.rul_cycles !== null ? `${row.rul_cycles} cyc` : '—'}
                  </td>
                  <td style={{ padding: '0.65rem 1rem', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    {row.model_version || 'hybrid_v1'}
                  </td>
                  <td style={{ padding: '0.65rem 1rem', fontSize: '0.72rem', color: 'var(--text-subtle)', fontFamily: 'var(--font-mono)' }}>
                    {row.timestamp ? new Date(row.timestamp).toLocaleTimeString() : '—'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
