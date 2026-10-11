import React, { useState, useMemo } from 'react';
import { LineChart, Sliders, Layers, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

export default function TelemetryChart({ telemetry, ecmData, socData }) {
  const [activeSeries, setActiveSeries] = useState({
    voltage: true,
    predicted: true,
    current: false,
    temperature: true,
    soc: false,
  });

  const [hoverIndex, setHoverIndex] = useState(null);
  const [zoomRange, setZoomRange] = useState([0, 100]); // percentage of cycle points

  const rawTime = telemetry?.time || [];
  const rawVoltage = telemetry?.voltage || [];
  const rawCurrent = telemetry?.current || [];
  const rawTemperature = telemetry?.temperature || [];
  const rawPredicted = ecmData?.voltage_predicted || [];
  const rawSoc = socData?.soc_trajectory || [];

  const totalPoints = rawVoltage.length;
  if (totalPoints === 0) return null;

  // Compute sliced view based on zoom slider
  const { startIndex, endIndex, time, voltage, current, temperature, predicted, soc } = useMemo(() => {
    const start = Math.floor((zoomRange[0] / 100) * (totalPoints - 1));
    const end = Math.max(start + 5, Math.ceil((zoomRange[1] / 100) * totalPoints));

    return {
      startIndex: start,
      endIndex: end,
      time: rawTime.slice(start, end),
      voltage: rawVoltage.slice(start, end),
      current: rawCurrent.slice(start, end),
      temperature: rawTemperature.slice(start, end),
      predicted: rawPredicted.slice(start, end),
      soc: rawSoc.slice(start, end),
    };
  }, [zoomRange, totalPoints, rawTime, rawVoltage, rawCurrent, rawTemperature, rawPredicted, rawSoc]);

  const n = voltage.length;

  // Chart dimensions
  const width = 850;
  const height = 300;
  const padding = { top: 25, right: 60, bottom: 35, left: 55 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  // Min / max calculation for primary Y-axis (Voltage)
  const minV = Math.min(...voltage, ...(predicted.length ? predicted : [3.5])) - 0.05;
  const maxV = Math.max(...voltage, ...(predicted.length ? predicted : [4.2])) + 0.05;

  const minTemp = Math.floor(Math.min(...(temperature.length ? temperature : [20])) - 1);
  const maxTemp = Math.ceil(Math.max(...(temperature.length ? temperature : [35])) + 1);

  // Scales
  const getX = (i) => padding.left + (i / Math.max(1, n - 1)) * plotW;
  const getYVoltage = (v) => padding.top + plotH - ((v - minV) / (maxV - minV || 1)) * plotH;
  const getYTemp = (t) => padding.top + plotH - ((t - minTemp) / (maxTemp - minTemp || 1)) * plotH;
  const getYSoC = (s) => padding.top + plotH - (s / 100) * plotH;

  // SVG Paths
  const createPath = (data, getYFunc) => {
    if (!data || data.length === 0) return '';
    return data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYFunc(d).toFixed(1)}`).join(' ');
  };

  const pathVoltage = createPath(voltage, getYVoltage);
  const pathPredicted = createPath(predicted, getYVoltage);
  const pathTemp = createPath(temperature, getYTemp);
  const pathSoC = createPath(soc, getYSoC);

  const toggleSeries = (key) => {
    setActiveSeries((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const resetZoom = () => setZoomRange([0, 100]);

  return (
    <div className="card" style={{ marginBottom: '2rem' }}>
      {/* Title & Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <LineChart size={18} color="var(--cyan-primary)" />
          <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', fontWeight: 700 }}>
            Dynamic Discharge Telemetry & ECM Tracking
          </h2>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-subtle)', fontFamily: 'var(--font-mono)' }}>
            Showing {n} of {totalPoints} points ({zoomRange[0]}% - {zoomRange[1]}%)
          </span>
        </div>

        {/* Series Filter Toggles */}
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {[
            { key: 'voltage', label: 'Measured V', color: '#00f2fe' },
            { key: 'predicted', label: 'ECM V_est', color: '#4facfe', dash: true },
            { key: 'temperature', label: 'Temp (°C, Right Axis)', color: '#f59e0b' },
            { key: 'soc', label: 'SoC (%)', color: '#10b981' },
          ].map((item) => (
            <button
              key={item.key}
              onClick={() => toggleSeries(item.key)}
              style={{
                background: activeSeries[item.key] ? 'rgba(255, 255, 255, 0.08)' : 'transparent',
                color: activeSeries[item.key] ? item.color : 'var(--text-subtle)',
                border: `1px solid ${activeSeries[item.key] ? item.color : 'var(--border-subtle)'}`,
                borderRadius: '6px',
                padding: '0.25rem 0.6rem',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                transition: 'all 0.2s',
              }}
            >
              <span style={{
                width: '8px',
                height: item.dash ? '2px' : '8px',
                borderRadius: item.dash ? '0' : '50%',
                backgroundColor: item.color,
              }} />
              {item.label}
            </button>
          ))}

          {(zoomRange[0] !== 0 || zoomRange[1] !== 100) && (
            <button
              onClick={resetZoom}
              className="btn btn-secondary"
              style={{ padding: '0.25rem 0.5rem', fontSize: '0.72rem', gap: '0.25rem' }}
              title="Reset Zoom"
            >
              <RotateCcw size={12} />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>

      {/* SVG Canvas Chart */}
      <div style={{ position: 'relative', width: '100%', overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: '100%', height: 'auto', display: 'block' }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = ((e.clientX - rect.left) / rect.width) * width;
            if (relX >= padding.left && relX <= width - padding.right) {
              const idx = Math.round(((relX - padding.left) / plotW) * (n - 1));
              setHoverIndex(Math.max(0, Math.min(n - 1, idx)));
            }
          }}
          onMouseLeave={() => setHoverIndex(null)}
        >
          {/* Axis Header Labels */}
          <text
            x={padding.left}
            y={padding.top - 8}
            fill="#00f2fe"
            fontSize="10"
            fontFamily="var(--font-mono)"
            fontWeight="600"
            textAnchor="start"
          >
            ← Voltage (V)
          </text>

          {activeSeries.temperature && (
            <text
              x={width - padding.right}
              y={padding.top - 8}
              fill="#f59e0b"
              fontSize="10"
              fontFamily="var(--font-mono)"
              fontWeight="600"
              textAnchor="end"
            >
              Temperature (°C) →
            </text>
          )}

          {/* Subtle Gridlines & Left/Right Y-Axis Ticks */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio, idx) => {
            const y = padding.top + plotH * ratio;
            const voltLabel = (maxV - ratio * (maxV - minV)).toFixed(2);
            const tempLabel = (maxTemp - ratio * (maxTemp - minTemp)).toFixed(0);
            return (
              <g key={idx}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="rgba(255, 255, 255, 0.06)"
                  strokeDasharray="4 4"
                />
                {/* Left Axis: Voltage */}
                <text
                  x={padding.left - 8}
                  y={y + 4}
                  fill="var(--text-subtle)"
                  fontSize="10"
                  fontFamily="var(--font-mono)"
                  textAnchor="end"
                >
                  {voltLabel}V
                </text>
                {/* Right Axis: Temperature */}
                {activeSeries.temperature && (
                  <text
                    x={width - padding.right + 8}
                    y={y + 4}
                    fill="#f59e0b"
                    fontSize="10"
                    fontFamily="var(--font-mono)"
                    textAnchor="start"
                    opacity="0.9"
                  >
                    {tempLabel}°C
                  </text>
                )}
              </g>
            );
          })}

          {/* Time axis ticks */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio, idx) => {
            const i = Math.round(ratio * (n - 1));
            const x = getX(i);
            const tVal = time[i] !== undefined ? `${Math.round(time[i])}s` : `${i * 10}s`;
            return (
              <text
                key={idx}
                x={x}
                y={height - 12}
                fill="var(--text-subtle)"
                fontSize="10"
                fontFamily="var(--font-mono)"
                textAnchor="middle"
              >
                {tVal}
              </text>
            );
          })}

          {/* Data Lines */}
          {activeSeries.temperature && (
            <path
              d={pathTemp}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="1.8"
              opacity="0.8"
            />
          )}

          {activeSeries.soc && (
            <path
              d={pathSoC}
              fill="none"
              stroke="#10b981"
              strokeWidth="2"
              opacity="0.85"
            />
          )}

          {activeSeries.predicted && predicted.length > 0 && (
            <path
              d={pathPredicted}
              fill="none"
              stroke="#4facfe"
              strokeWidth="2"
              strokeDasharray="5 3"
              opacity="0.9"
            />
          )}

          {activeSeries.voltage && (
            <path
              d={pathVoltage}
              fill="none"
              stroke="#00f2fe"
              strokeWidth="2.4"
            />
          )}

          {/* Crosshair & Hover Tooltip */}
          {hoverIndex !== null && hoverIndex < n && (
            <g>
              <line
                x1={getX(hoverIndex)}
                y1={padding.top}
                x2={getX(hoverIndex)}
                y2={padding.top + plotH}
                stroke="rgba(0, 242, 254, 0.5)"
                strokeWidth="1"
                strokeDasharray="2 2"
              />
              <circle
                cx={getX(hoverIndex)}
                cy={getYVoltage(voltage[hoverIndex])}
                r="4"
                fill="#00f2fe"
                stroke="#07090e"
                strokeWidth="2"
              />
            </g>
          )}
        </svg>

        {/* Hover info overlay */}
        {hoverIndex !== null && hoverIndex < n && (
          <div style={{
            position: 'absolute',
            top: '15px',
            right: '25px',
            background: 'rgba(7, 9, 14, 0.92)',
            border: '1px solid var(--border-active)',
            borderRadius: '8px',
            padding: '0.5rem 0.85rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.75rem',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            gap: '1rem',
          }}>
            <div><span style={{ color: 'var(--text-subtle)' }}>Time: </span>{Math.round(time[hoverIndex])}s</div>
            <div><span style={{ color: '#00f2fe' }}>V_meas: </span>{voltage[hoverIndex]}V</div>
            {predicted[hoverIndex] !== undefined && (
              <div><span style={{ color: '#4facfe' }}>V_est: </span>{predicted[hoverIndex]}V</div>
            )}
            <div><span style={{ color: '#f59e0b' }}>Temp: </span>{temperature[hoverIndex]}°C</div>
            {soc[hoverIndex] !== undefined && (
              <div><span style={{ color: '#10b981' }}>SoC: </span>{soc[hoverIndex]}%</div>
            )}
          </div>
        )}
      </div>

      {/* Interactive Range Zoom Slider */}
      <div style={{ marginTop: '0.85rem', padding: '0 0.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-subtle)', fontFamily: 'var(--font-mono)', minWidth: '70px' }}>
          ZOOM SLICE:
        </span>
        <input
          type="range"
          min="0"
          max="80"
          value={zoomRange[0]}
          onChange={(e) => {
            const val = parseInt(e.target.value);
            setZoomRange([val, Math.max(val + 10, zoomRange[1])]);
          }}
          style={{ flex: 1, accentColor: 'var(--cyan-primary)' }}
        />
        <input
          type="range"
          min="20"
          max="100"
          value={zoomRange[1]}
          onChange={(e) => {
            const val = parseInt(e.target.value);
            setZoomRange([Math.min(zoomRange[0], val - 10), val]);
          }}
          style={{ flex: 1, accentColor: 'var(--cyan-primary)' }}
        />
      </div>
    </div>
  );
}
