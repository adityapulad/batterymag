"""
Fault Detector — monitors ECM parameter degradation for anomalies.

Thresholds are based on typical Li-ion degradation patterns from NASA dataset.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class FaultAlert:
    type: str
    parameter: str
    current_value: float
    baseline_value: float
    change_percent: float
    severity: str  # "info", "warning", "critical"
    message: str


# Default thresholds (percent change from baseline)
DEFAULT_THRESHOLDS = {
    "R0": {"warning": 20.0, "critical": 50.0},
    "R1": {"warning": 25.0, "critical": 60.0},
    "C1": {"warning": -20.0, "critical": -40.0},  # C1 *decreases* with degradation
    "capacity": {"warning": -10.0, "critical": -20.0},
}


class FaultDetector:
    """Detect battery faults by comparing current ECM params to baseline."""

    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def detect(
        self,
        current_params: Dict[str, float],
        baseline_params: Dict[str, float],
    ) -> List[FaultAlert]:
        """
        Compare current ECM parameters against baseline.

        Args:
            current_params: Latest extracted ECM parameters.
            baseline_params: Reference (healthy) ECM parameters.

        Returns:
            List of FaultAlert objects.
        """
        alerts: List[FaultAlert] = []

        for param, thresholds in self.thresholds.items():
            current = current_params.get(param)
            baseline = baseline_params.get(param)

            if current is None or baseline is None or baseline == 0:
                continue

            change_pct = ((current - baseline) / abs(baseline)) * 100

            # For C1 / capacity — decreasing is bad
            if thresholds.get("warning", 0) < 0:
                # Negative threshold → trigger on negative change
                if change_pct <= thresholds["critical"]:
                    severity = "critical"
                elif change_pct <= thresholds["warning"]:
                    severity = "warning"
                else:
                    continue
            else:
                # Positive threshold → trigger on positive change (resistance increase)
                if change_pct >= thresholds["critical"]:
                    severity = "critical"
                elif change_pct >= thresholds["warning"]:
                    severity = "warning"
                else:
                    continue

            message = self._generate_message(param, change_pct, severity)
            alerts.append(FaultAlert(
                type=self._fault_type(param),
                parameter=param,
                current_value=round(current, 6),
                baseline_value=round(baseline, 6),
                change_percent=round(change_pct, 2),
                severity=severity,
                message=message,
            ))

        return alerts

    @staticmethod
    def _fault_type(param: str) -> str:
        mapping = {
            "R0": "high_resistance",
            "R1": "charge_transfer_degradation",
            "C1": "capacity_loss",
            "capacity": "capacity_fade",
        }
        return mapping.get(param, "parameter_anomaly")

    @staticmethod
    def _generate_message(param: str, change_pct: float, severity: str) -> str:
        messages = {
            "R0": f"Internal resistance (R0) changed by {change_pct:+.1f}% — potential degradation",
            "R1": f"Charge-transfer resistance (R1) changed by {change_pct:+.1f}% — slower kinetics",
            "C1": f"Polarization capacitance (C1) changed by {change_pct:+.1f}% — reduced capacity",
            "capacity": f"Battery capacity changed by {change_pct:+.1f}% — degradation detected",
        }
        return messages.get(param, f"Parameter {param} changed by {change_pct:+.1f}%")

    @staticmethod
    def overall_status(alerts: List[FaultAlert]) -> str:
        if any(a.severity == "critical" for a in alerts):
            return "critical"
        if any(a.severity == "warning" for a in alerts):
            return "warning"
        return "healthy"

    @staticmethod
    def alerts_to_dicts(alerts: List[FaultAlert]) -> List[Dict]:
        return [asdict(a) for a in alerts]
