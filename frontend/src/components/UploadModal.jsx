import React, { useState } from 'react';
import { X, UploadCloud, FileText, CheckCircle, AlertCircle } from 'lucide-react';

export default function UploadModal({ isOpen, onClose, onDataLoaded }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = import.meta.env.VITE_API_URL 
        ? `${import.meta.env.VITE_API_URL}/upload/dataset`
        : '/api/v1/upload/dataset';
      const res = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Failed to parse CSV');
      }

      const data = await res.json();
      onDataLoaded({
        battery_id: data.filename.replace(/\.[^/.]+$/, ''),
        voltage: data.voltage_sample,
        current: data.current_sample,
        temperature: data.temperature_sample,
        time: data.time_sample,
        current_cycle: 60,
        capacity: 1.8,
      });
      onClose();
    } catch (err) {
      // Local fallback parsing using browser FileReader if backend upload is offline
      try {
        const text = await file.text();
        const lines = text.trim().split('\n');
        if (lines.length < 3) throw new Error('File has too few rows');
        const headers = lines[0].toLowerCase().split(',').map(h => h.trim());
        const vIdx = headers.findIndex(h => h.includes('volt') || h === 'v');
        const iIdx = headers.findIndex(h => h.includes('curr') || h === 'i');
        const tIdx = headers.findIndex(h => h.includes('temp') || h === 't');

        if (vIdx === -1 || iIdx === -1) {
          throw new Error('CSV must contain voltage and current headers');
        }

        const vArr = [];
        const iArr = [];
        const tArr = [];
        const timeArr = [];

        for (let idx = 1; idx < Math.min(201, lines.length); idx++) {
          const parts = lines[idx].split(',').map(p => parseFloat(p.trim()));
          if (!isNaN(parts[vIdx]) && !isNaN(parts[iIdx])) {
            vArr.push(parts[vIdx]);
            iArr.push(parts[iIdx]);
            tArr.push(tIdx !== -1 && !isNaN(parts[tIdx]) ? parts[tIdx] : 25.0);
            timeArr.push((idx - 1) * 30);
          }
        }

        onDataLoaded({
          battery_id: file.name.replace(/\.[^/.]+$/, ''),
          voltage: vArr,
          current: iArr,
          temperature: tArr,
          time: timeArr,
          current_cycle: 50,
          capacity: 1.8,
        });
        onClose();
      } catch (clientErr) {
        setError(clientErr.message || 'Error parsing CSV');
      }
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.75)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 999,
      padding: '1rem',
    }}>
      <div className="card" style={{
        maxWidth: '520px',
        width: '100%',
        background: '#0e131f',
        border: '1px solid var(--border-active)',
        position: 'relative',
      }}>
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '1rem',
            right: '1rem',
            background: 'transparent',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
          }}
        >
          <X size={20} />
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
          <UploadCloud size={22} color="var(--cyan-primary)" />
          <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.2rem', fontWeight: 700 }}>
            Upload Battery Telemetry CSV
          </h3>
        </div>

        <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
          Ingest raw battery discharge data. The file must include columns for <code>voltage</code> and <code>current</code> (temperature and time are optional).
        </p>

        {/* Drop Area */}
        <label style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem 1rem',
          border: '2px dashed var(--border-subtle)',
          borderRadius: '10px',
          background: 'rgba(255, 255, 255, 0.02)',
          cursor: 'pointer',
          marginBottom: '1rem',
          transition: 'all 0.2s',
        }}>
          <input
            type="file"
            accept=".csv,.txt,.mat"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <FileText size={32} color={file ? 'var(--cyan-primary)' : 'var(--text-muted)'} style={{ marginBottom: '0.5rem' }} />
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: file ? 'var(--text-main)' : 'var(--text-muted)' }}>
            {file ? file.name : 'Click to select or drag CSV / MAT file'}
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-subtle)', marginTop: '0.2rem' }}>
            Supports NASA (.mat / .csv), Oxford, CALCE battery formats
          </span>
        </label>

        {error && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.775rem',
            color: 'var(--rose-accent)',
            marginBottom: '1rem',
          }}>
            <AlertCircle size={15} />
            <span>{error}</span>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            onClick={handleUpload}
            disabled={!file || uploading}
          >
            {uploading ? 'Processing...' : 'Load & Run Inference'}
          </button>
        </div>
      </div>
    </div>
  );
}
