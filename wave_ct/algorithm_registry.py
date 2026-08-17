"""Final-version algorithm registry and promotion policy for WaveCT."""

from __future__ import annotations

from typing import Final


WAVECT_VERSION: Final = "3.0 Final"


ALGORITHMS: Final = (
    {
        "id": "sirt_automatic_tuning",
        "name": "SIRT 增广系统迭代反演（CT_shi 脚本兼容主算法）",
        "status": "production_default",
        "entrypoint": "WaveCT GUI -> python -m wave_ct.inversion --solver-method sirt --script-compatible-sirt",
        "note": "GUI 默认复现 CT_shi_SIRT_Automatic_tuning_global_optimum.py 的 10 m 绝对慢度 SIRT + DE 调参 + 300 次最终迭代；量化 velocity 与显示场严格分离。",
    },
    {
        "id": "auto_tv_family",
        "name": "自动事件分组 CV：TV / Haar / 分层网格 / 差分到时",
        "status": "experimental_opt_in",
        "entrypoint": "python -m wave_ct.auto_select",
        "note": "实验可选；由同一数据集的事件分组交叉验证选择，不能替代正式 SIRT 主入口。",
    },
    {
        "id": "deep_coordinate_dnr",
        "name": "2026 坐标网络深度重参数化 DNR",
        "status": "experimental_opt_in",
        "entrypoint": "WaveCT 高级参数中的 deep_reparameterization",
        "note": "严格固定350轮复核为4/6折改善、最差劣化2.30%；保留研究入口，不是默认方案。",
    },
    {
        "id": "eikonal_forward",
        "name": "Eikonal 快速行进正演",
        "status": "diagnostic_only",
        "entrypoint": "python -m wave_ct.tools.eikonal_probe",
        "note": "阶段9正演门槛失败；仅用于校准与诊断。",
    },
    {
        "id": "bent_ray_dnr",
        "name": "弯曲射线 DNR 重线性化",
        "status": "diagnostic_only",
        "entrypoint": "python -m wave_ct.tools.bent_ray_dnr_screen",
        "note": "重追踪误差显著高于直射线，禁止进入正式自动选模。",
    },
    {
        "id": "bent_ray_gauss_newton",
        "name": "弯曲射线 Gauss-Newton 延拓",
        "status": "diagnostic_only",
        "entrypoint": "python -m wave_ct.tools.bent_ray_gauss_newton",
        "note": "未完成全对比度且出现追踪失败。",
    },
    {
        "id": "source_relocation",
        "name": "水平震源重定位",
        "status": "diagnostic_only",
        "entrypoint": "python -m wave_ct.tools.source_relocation_cv",
        "note": "嵌套未见台站验证失败；不与正式速度反演联合。",
    },
    {
        "id": "receiver_static",
        "name": "台站静校正",
        "status": "diagnostic_only",
        "entrypoint": "python -m wave_ct.tools.receiver_static_cv",
        "note": "改善折数未达门槛。",
    },
    {
        "id": "global_anisotropy",
        "name": "全局水平弱各向异性",
        "status": "diagnostic_only",
        "entrypoint": "python -m wave_ct.tools.global_anisotropy_cv",
        "note": "全新留出仅1/2改善，证据不足。",
    },
    {
        "id": "dnr_strict_suite",
        "name": "DNR 严格泛化实验套件",
        "status": "research_suite",
        "entrypoint": "wave_ct.tools.dnr_*_strict",
        "note": "包含轮次、集成、稳健损失、结构、Fourier、覆盖、平滑、学习率、背景、网格与差分损失。",
    },
)


STATUS_LABELS: Final = {
    "production_default": "正式默认",
    "experimental_opt_in": "实验可选",
    "diagnostic_only": "仅诊断",
    "research_suite": "研究套件",
}


def algorithm_status_text() -> str:
    """Return a readable, stable final-version algorithm inventory."""

    lines = [
        f"WaveCT {WAVECT_VERSION} 算法整合状态",
        "",
        "整合原则：稳定算法进入自动选模；未通过阶段9门槛的方法保留为可复现实验入口，不参与正式默认反演。",
        "",
    ]
    for item in ALGORITHMS:
        label = STATUS_LABELS[str(item["status"])]
        lines.extend(
            [
                f"[{label}] {item['name']}",
                f"入口：{item['entrypoint']}",
                f"说明：{item['note']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def validate_algorithm_registry() -> None:
    """Raise if final-version registry invariants are broken."""

    identifiers = [str(item["id"]) for item in ALGORITHMS]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("algorithm registry contains duplicate IDs")
    if not any(item["status"] == "production_default" for item in ALGORITHMS):
        raise ValueError("algorithm registry has no production default")
    unknown = {
        str(item["status"]) for item in ALGORITHMS
    } - set(STATUS_LABELS)
    if unknown:
        raise ValueError(f"algorithm registry has unknown statuses: {unknown}")
