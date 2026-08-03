"""Security validation status (分层验收)。

分三层独立记录：

* ``provider_redaction_tests_passed``：Provider 级脱敏测试（log / telemetry
  不出现密钥）。Mock 即可验证。
* ``mock_artifact_scan_passed``：Mock 全链路产物（logs/telemetry/report/
  cache/analysis/validation）扫描未发现密钥 / 完整 Base64 / Bearer 值。
* ``real_run_artifact_scan_passed``：真实外部运行后产物的安全扫描。仅在
  ``real_vision_api_used=true`` 之后才能为 true。

``security_checks_passed`` 是总体布尔：

* release_candidate 阶段允许基于前两层为 true；
* production_ready 必须前三层全部为 true；
* blocked_external 状态下仍要求前两层为 true。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SecurityValidationStatus:
    """三层独立的安全验收状态。"""

    provider_redaction_tests_passed: bool = False
    mock_artifact_scan_passed: bool = False
    real_run_artifact_scan_passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "provider_redaction_tests_passed": self.provider_redaction_tests_passed,
            "mock_artifact_scan_passed": self.mock_artifact_scan_passed,
            "real_run_artifact_scan_passed": self.real_run_artifact_scan_passed,
            "notes": list(self.notes),
        }

    @property
    def security_checks_passed(self) -> bool:
        """``release_candidate`` 允许：前两层 + Provider redaction 全 true。"""
        return (
            self.provider_redaction_tests_passed
            and self.mock_artifact_scan_passed
        )


__all__ = ["SecurityValidationStatus"]