import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import KPICards from './components/KPICards';
import TelemetryChart from './components/TelemetryChart';
import ECMParameters from './components/ECMParameters';
import FaultPanel from './components/FaultPanel';
import UploadModal from './components/UploadModal';
import HistoryPanel from './components/HistoryPanel';
import {
  generateBatteryProfile,
  checkHealth,
  predictSoH,
  predictSoC,
  predictRUL,
  analyzeECM,
  detectFaults,
  fetchPredictionHistory,
} from './api';

export default function App() {
  const [selectedProfile, setSelectedProfile] = useState('healthy');
  const [telemetry, setTelemetry] = useState(() => generateBatteryProfile('healthy'));
  const [loading, setLoading] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);

  // Model inference outputs
  const [sohData, setSohData] = useState(null);
  const [socData, setSocData] = useState(null);
  const [rulData, setRulData] = useState(null);
  const [ecmData, setEcmData] = useState(null);
  const [faultData, setFaultData] = useState(null);

  // Prediction History
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  // Poll backend health once on mount and every 30s
  useEffect(() => {
    const check = async () => {
      const status = await checkHealth();
      setBackendStatus(status);
    };
    check();
    const timer = setInterval(check, 30000);
    return () => clearInterval(timer);
  }, []);

  const loadHistory = useCallback(async () => {
    setLoadingHistory(true);
    try {
      const records = await fetchPredictionHistory();
      setHistory(records);
    } finally {
      setLoadingHistory(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // Execute inference pipeline for current telemetry
  const runInferencePipeline = useCallback(async (currentTel) => {
    setLoading(true);
    try {
      const payload = {
        battery_id: currentTel.battery_id,
        voltage: currentTel.voltage,
        current: currentTel.current,
        temperature: currentTel.temperature,
        time: currentTel.time,
        current_cycle: currentTel.current_cycle,
        capacity: currentTel.capacity,
      };

      // Run parallel inferences
      const [sohRes, socRes, rulRes, ecmRes] = await Promise.all([
        predictSoH(payload),
        predictSoC(payload),
        predictRUL(payload),
        analyzeECM({
          battery_id: payload.battery_id,
          model_type: 'RC',
          voltage: payload.voltage,
          current: payload.current,
          time: payload.time,
        }),
      ]);

      setSohData(sohRes);
      setSocData(socRes);
      setRulData(rulRes);
      setEcmData(ecmRes);

      // Fault detection based on extracted params
      const faultPayload = {
        battery_id: payload.battery_id,
        current_params: {
          ...(ecmRes?.parameters || sohRes?.ecm_params || { R0: 0.05 }),
          capacity: payload.capacity,
        },
      };
      const faultRes = await detectFaults(faultPayload);
      setFaultData(faultRes);

      // Refresh history records after saving
      setTimeout(() => {
        loadHistory();
      }, 500);
    } catch (err) {
      console.error('Inference pipeline error:', err);
    } finally {
      setLoading(false);
    }
  }, [loadHistory]);

  // Run when profile changes or on initial mount
  useEffect(() => {
    runInferencePipeline(telemetry);
  }, [telemetry, runInferencePipeline]);

  const handleSelectProfile = (profileId) => {
    setSelectedProfile(profileId);
    const newProfile = generateBatteryProfile(profileId);
    setTelemetry(newProfile);
  };

  const handleCustomDataLoaded = (customTelemetry) => {
    setSelectedProfile('custom');
    setTelemetry(customTelemetry);
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <Header
        selectedProfile={selectedProfile}
        onSelectProfile={handleSelectProfile}
        onRefresh={() => runInferencePipeline(telemetry)}
        onOpenUpload={() => setIsUploadOpen(true)}
        loading={loading}
        backendStatus={backendStatus}
      />

      {/* Main KPI Stat Cards */}
      <KPICards
        sohData={sohData}
        socData={socData}
        rulData={rulData}
        faultData={faultData}
      />

      {/* Discharge Telemetry & Tracking Chart (with Zoom Slider) */}
      <TelemetryChart
        telemetry={telemetry}
        ecmData={ecmData}
        socData={socData}
      />

      {/* 2-Column Split: ECM Physics Circuit & Fault Diagnostics */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
        gap: '1.5rem',
      }}>
        <ECMParameters ecmData={ecmData} sohData={sohData} />
        <FaultPanel faultData={faultData} />
      </div>

      {/* Prediction Audit History Log */}
      <HistoryPanel
        history={history}
        onRefresh={loadHistory}
        loading={loadingHistory}
      />

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onDataLoaded={handleCustomDataLoaded}
      />
    </div>
  );
}
