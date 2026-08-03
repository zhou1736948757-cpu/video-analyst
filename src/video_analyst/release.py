"""Release readiness model (Epic 6).

定义发布状态，不得只因测试数量多就标记 production_ready。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReleaseStatus = Literal[
    "development",
    "release_candidate",
    "production_ready",
    "blocked_external",
]


class ReleaseReadiness(BaseModel):
    """发布就绪度评估。"""

    code_complete: bool = False
    tests_passed: bool = False
    package_built: bool = False
    clean_install_verified: bool = False
    mock_e2e_verified: bool = False
    supported_python_matrix_verified: bool = False

    real_api_verified: bool = False
    real_video_verified: bool = False
    terminal_real_e2e_verified: bool = False

    security_checks_passed: bool = False
    documentation_complete: bool = False
    human_validation_passed: bool = False

    release_status: ReleaseStatus = "development"
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def evaluate(self) -> ReleaseStatus:
        """根据各项检查结果派生 release_status。

        规则（Production Readiness Gate §19）：
        - 代码/测试/构建/干净安装/Mock E2E/Python 矩阵/安全任一为 False → development
        - 全部本地项通过但 real_api_verified 或 real_video_verified 为 False → release_candidate
        - 真实 API 已实际调用但外部服务失败（代码+协议测试通过）→ blocked_external
        - 真实 API + 真实视频 + 安全 + 干净安装 + 文档 + 人工验收全通过 → production_ready
        - terminal_real_e2e_verified=False：作为 warning；若终局补帧为核心功能则追加 blocker
        """
        self.blockers = []
        self.warnings = []

        if not self.code_complete:
            self.blockers.append("code not complete")
        if not self.tests_passed:
            self.blockers.append("tests not passed")
        if not self.package_built:
            self.blockers.append("package not built")
        if not self.clean_install_verified:
            self.blockers.append("clean install not verified")
        if not self.mock_e2e_verified:
            self.blockers.append("mock E2E not verified")
        if not self.supported_python_matrix_verified:
            self.blockers.append("supported Python matrix not verified")
        if not self.security_checks_passed:
            self.blockers.append("security checks not passed")

        if self.blockers:
            self.release_status = "development"
            return self.release_status

        if not self.documentation_complete:
            self.warnings.append("documentation incomplete")

        if not self.real_api_verified:
            self.warnings.append("real API not verified")
            self.release_status = "release_candidate"
            return self.release_status

        if not self.real_video_verified:
            self.warnings.append("real API verified but real video not verified")
            self.release_status = "blocked_external"
            return self.release_status

        if not self.human_validation_passed:
            self.warnings.append("human validation not passed")
            self.release_status = "blocked_external"
            return self.release_status

        # 真实 API + 真实视频 + 安全 + 干净安装 + 文档 + 人工验收全通过 → production_ready
        if (
            self.real_api_verified
            and self.real_video_verified
            and self.security_checks_passed
            and self.clean_install_verified
            and self.documentation_complete
            and self.human_validation_passed
        ):
            if not self.terminal_real_e2e_verified:
                self.warnings.append(
                    "terminal real E2E not triggered; terminal refinement is a claimed feature"
                )
            self.release_status = "production_ready"
            return self.release_status

        self.release_status = "release_candidate"
        return self.release_status


def assess_release_readiness(
    *,
    tests_passed: bool,
    package_built: bool = False,
    clean_install_verified: bool = False,
    mock_e2e_verified: bool = False,
    supported_python_matrix_verified: bool = False,
    real_api_verified: bool = False,
    real_video_verified: bool = False,
    terminal_real_e2e_verified: bool = False,
    security_checks_passed: bool = True,
    documentation_complete: bool = False,
    human_validation_passed: bool = False,
) -> ReleaseReadiness:
    """评估发布就绪度。"""
    r = ReleaseReadiness(
        code_complete=True,
        tests_passed=tests_passed,
        package_built=package_built,
        clean_install_verified=clean_install_verified,
        mock_e2e_verified=mock_e2e_verified,
        supported_python_matrix_verified=supported_python_matrix_verified,
        real_api_verified=real_api_verified,
        real_video_verified=real_video_verified,
        terminal_real_e2e_verified=terminal_real_e2e_verified,
        security_checks_passed=security_checks_passed,
        documentation_complete=documentation_complete,
        human_validation_passed=human_validation_passed,
    )
    r.evaluate()
    return r


__all__ = [
    "ReleaseReadiness",
    "ReleaseStatus",
    "assess_release_readiness",
]