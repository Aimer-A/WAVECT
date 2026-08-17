"""
生成反演数据与结果的可复核质控图。

输出:
- 表观速度直方图: 检查保留射线的稳健速度分布。
- 距离-到时散点图: 检查到时与当前数据参考速度区间的关系。
- 典型波形拾取图: 检查拾取点是否避开前段数值噪声。
- 文本报告: 汇总数量、速度范围和切片统计。
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 220


ROOT = Path(__file__).resolve().parents[1]
INV_CSV = ROOT / "标波输出数据集" / "反演数据集.csv"
DETAIL_CSV = ROOT / "标波输出数据集" / "反演数据集_detail.csv"
SLICE_REPORT = ROOT.parent / "成像效果展示" / "slice_report.txt"
OUT_DIR = ROOT.parent / "成像效果展示"
WAVEFORM_ROOT = ROOT / "正演波形数据集"
REFERENCE_VMIN = 0.0
REFERENCE_VMAX = 0.0


def fnum(text: str) -> float:
    return float(str(text).strip())


def fnum_or_none(text: object) -> Optional[float]:
    raw = str(text).strip()
    if raw == "" or raw.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_inversion_rows() -> Tuple[List[Dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    rows: List[Dict[str, str]] = []
    distances: List[float] = []
    times_s: List[float] = []
    velocities: List[float] = []

    with INV_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sx = fnum(row["震源坐标-x"])
            sy = fnum(row["震源坐标-y"])
            sz = fnum(row["震源坐标-z"])
            rx = fnum(row["台站坐标-x"])
            ry = fnum(row["台站坐标-y"])
            rz = fnum(row["台站坐标-z"])
            tt = fnum(row["震源-台站传播时间"]) / 1000.0
            dist = float(np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2 + (sz - rz) ** 2))
            rows.append(row)
            distances.append(dist)
            times_s.append(tt)
            velocities.append(dist / tt)

    return rows, np.array(distances), np.array(times_s), np.array(velocities)


def anomaly_length_in_box(p1: np.ndarray, p2: np.ndarray, lo: float = 400.0, hi: float = 600.0) -> float:
    d = p2 - p1
    a0 = 0.0
    a1 = 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if p1[i] < lo or p1[i] > hi:
                return 0.0
            continue
        aa = (lo - p1[i]) / d[i]
        bb = (hi - p1[i]) / d[i]
        if aa > bb:
            aa, bb = bb, aa
        a0 = max(a0, aa)
        a1 = min(a1, bb)
    if a1 <= a0:
        return 0.0
    return float(np.linalg.norm(d) * (a1 - a0))


def synthetic_truth_residuals(rows: List[Dict[str, str]]) -> np.ndarray:
    residuals: List[float] = []
    for row in rows:
        p1 = np.array([fnum(row["震源坐标-x"]), fnum(row["震源坐标-y"]), fnum(row["震源坐标-z"])], dtype=np.float64)
        p2 = np.array([fnum(row["台站坐标-x"]), fnum(row["台站坐标-y"]), fnum(row["台站坐标-z"])], dtype=np.float64)
        total_len = float(np.linalg.norm(p2 - p1))
        anomaly_len = anomaly_length_in_box(p1, p2)
        truth_s = (total_len - anomaly_len) / 4000.0 + anomaly_len / 4500.0
        picked_s = fnum(row["震源-台站传播时间"]) / 1000.0
        residuals.append((picked_s - truth_s) * 1000.0)
    return np.array(residuals, dtype=np.float64)


def load_detail_rows() -> List[Dict[str, str]]:
    if not DETAIL_CSV.exists():
        return []
    with DETAIL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_velocity_hist(velocities: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.0), facecolor="white")
    ax.hist(velocities, bins=30, color="#d44a1c", edgecolor="white")
    ax.axvline(REFERENCE_VMIN, color="#1f77b4", linestyle="--", linewidth=2.0,
               label=f"参考下限 {REFERENCE_VMIN:.0f} m/s")
    ax.axvline(REFERENCE_VMAX, color="#d60000", linestyle="--", linewidth=2.0,
               label=f"参考上限 {REFERENCE_VMAX:.0f} m/s")
    ax.set_title("速度分布直方图", fontsize=16, fontweight="bold")
    ax.set_xlabel("表观速度 (m/s)")
    ax.set_ylabel("射线数量")
    ax.legend()
    ax.grid(alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e00000")
        spine.set_linewidth(1.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_表观速度直方图.png")
    plt.close(fig)


def save_time_distance_plot(distances: np.ndarray, times_s: np.ndarray) -> None:
    order = np.argsort(distances)
    d_sorted = distances[order]
    fig, ax = plt.subplots(figsize=(7.8, 5.0), facecolor="white")
    ax.scatter(distances, times_s * 1000.0, s=16, alpha=0.65, color="#8b1e1e", label="保留拾取")
    ax.plot(d_sorted, d_sorted / REFERENCE_VMAX * 1000.0, color="#d60000", linewidth=2.4,
            label=f"{REFERENCE_VMAX:.0f} m/s 参考线")
    ax.plot(d_sorted, d_sorted / REFERENCE_VMIN * 1000.0, color="#1f77b4", linewidth=2.4,
            label=f"{REFERENCE_VMIN:.0f} m/s 参考线")
    ax.fill_between(
        d_sorted,
        d_sorted / REFERENCE_VMAX * 1000.0,
        d_sorted / REFERENCE_VMIN * 1000.0,
        color="#59A14F",
        alpha=0.13,
        label="物理合理区间",
    )
    ax.set_title("时间 vs 距离质控图", fontsize=16, fontweight="bold")
    ax.set_xlabel("震源-台站距离 (m)")
    ax.set_ylabel("P波传播时间 (ms)")
    ax.legend()
    ax.grid(alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e00000")
        spine.set_linewidth(1.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_距离到时散点图.png")
    plt.close(fig)


def save_pick_examples(detail_rows: List[Dict[str, str]]) -> None:
    kept = [r for r in detail_rows if r.get("keep") == "1"]
    if not kept:
        return

    examples = []
    for source_id in ("1", "42", "100", "178"):
        hit = next((r for r in kept if r.get("source_id") == source_id or r.get("source") == f"source_{source_id}"), None)
        if hit is not None:
            examples.append(hit)
    if len(examples) < 4:
        examples = kept[:4]

    fig, axes = plt.subplots(len(examples), 1, figsize=(9.2, 2.3 * len(examples)), sharex=False, facecolor="white")
    if len(examples) == 1:
        axes = [axes]

    for ax, row in zip(axes, examples):
        source_name = row.get("source_name") or row.get("source") or ""
        station = row.get("station", "")
        file_name = row.get("file_name", "")
        if not file_name and source_name and station:
            hits = sorted((WAVEFORM_ROOT / source_name).glob(f"*.{station}.FXZ.semv"))
            file_name = hits[0].name if hits else ""
        waveform_path = WAVEFORM_ROOT / source_name / file_name
        if not waveform_path.exists():
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"未找到波形: {source_name} {station}", ha="center", va="center")
            continue
        data = np.loadtxt(waveform_path, dtype=np.float64)
        t_ms = data[:, 0] * 1000.0
        amp = data[:, 1]
        # 波形横轴是 semv 原始时间，必须画原始拾取时刻；传播时间包含
        # time_origin/source_delay 修正，只能用于距离-到时图和反演。
        pick_sec = fnum_or_none(row.get("raw_pick_time_s"))
        if pick_sec is None:
            raw_pick_ms = fnum_or_none(row.get("raw_pick_time_ms"))
            pick_sec = raw_pick_ms / 1000.0 if raw_pick_ms is not None else None
        if pick_sec is None:
            pick_sec = fnum_or_none(row.get("p_time_sec") or row.get("p_time_s"))
        if pick_sec is None:
            pick_sec = fnum_or_none(row.get("travel_time_sec"))
        if pick_sec is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"缺少拾取时间: {source_name} {station}", ha="center", va="center")
            continue

        tmin_sec = fnum_or_none(row.get("expected_t_min_sec") or row.get("search_min_s"))
        tmax_sec = fnum_or_none(row.get("expected_t_max_sec") or row.get("search_max_s"))
        pick_ms = pick_sec * 1000.0

        if tmin_sec is not None and tmax_sec is not None:
            tmin_ms = tmin_sec * 1000.0
            tmax_ms = tmax_sec * 1000.0
            view_min = tmin_ms - 35.0
            view_max = tmax_ms + 45.0
        else:
            tmin_ms = None
            tmax_ms = None
            view_min = pick_ms - 45.0
            view_max = pick_ms + 95.0

        mask = (t_ms >= view_min) & (t_ms <= view_max)
        ax.plot(t_ms[mask], amp[mask], color="#222222", linewidth=1.0)
        if tmin_ms is not None and tmax_ms is not None:
            ax.axvspan(tmin_ms, tmax_ms, color="#fff176", alpha=0.55, label="拾取搜索窗")
        ax.axvline(pick_ms, color="#d60000", linewidth=2.0, label="最终拾取")
        method = row.get("pick_method") or row.get("method", "")
        ax.set_title(f"{source_name} {station}  {method}")
        ax.set_ylabel("振幅")
        ax.grid(alpha=0.22)
        for spine in ax.spines.values():
            spine.set_edgecolor("#e00000")
            spine.set_linewidth(1.2)

    axes[-1].set_xlabel("时间 (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(OUT_DIR / "03_典型波形拾取检查.png")
    plt.close(fig)


def add_arrow(fig: plt.Figure, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        transform=fig.transFigure,
        arrowstyle="simple",
        mutation_scale=34,
        color="#ff0000",
        linewidth=0,
    )
    fig.patches.append(arrow)


def draw_menu_box(ax: plt.Axes, title: str, items: List[str], highlight: int = 0) -> None:
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, facecolor="#f8fbff", edgecolor="#b7c7d9", linewidth=1.2))
    ax.text(0.06, 0.91, title, fontsize=12, fontweight="bold", color="#111827")
    y = 0.78
    for i, item in enumerate(items):
        color = "#dbeafe" if i == highlight else "#ffffff"
        ax.add_patch(Rectangle((0.08, y - 0.055), 0.78, 0.09, facecolor=color, edgecolor="#cbd5e1", linewidth=0.8))
        ax.text(0.11, y, item, va="center", fontsize=10, color="#111827")
        y -= 0.12


def save_workflow_slide() -> None:
    fig = plt.figure(figsize=(12, 6.8), facecolor="white")
    fig.text(0.10, 0.91, "数据输入", fontsize=22, fontweight="bold")
    fig.text(0.34, 0.91, "数据管理", fontsize=22, fontweight="bold")
    fig.text(0.58, 0.91, "生成模型网格", fontsize=22, fontweight="bold")
    fig.text(0.84, 0.91, "反演", fontsize=22, fontweight="bold")

    boxes = [
        (0.05, 0.50, ["设置文件夹路径...", "数据输入...", "模型输入...", "退出"], "文件(F)"),
        (0.30, 0.50, ["数据过滤...", "时间 vs 距离", "速度分布直方图", "关闭数据"], "数据(D)"),
        (0.55, 0.50, ["生成速度模型", "提取数据", "模型网格", "射线及穿过节点统计", "模型对比"], "模型(M)"),
        (0.80, 0.50, ["反演参数设置...", "开始反演", "保存3DM结果"], "反演"),
    ]
    for x, y, items, title in boxes:
        ax = fig.add_axes([x, y, 0.16, 0.28])
        draw_menu_box(ax, title, items, 1)

    add_arrow(fig, 0.22, 0.64, 0.29, 0.64)
    add_arrow(fig, 0.47, 0.64, 0.54, 0.64)
    add_arrow(fig, 0.72, 0.64, 0.79, 0.64)

    fig.text(0.37, 0.34, "Wave CT Studio 反演流程", color="#e00000", fontsize=26, fontweight="bold")
    fig.text(0.18, 0.24, "输入CSV/3DD数据", fontsize=14)
    fig.text(0.41, 0.24, "检查时间-距离与速度分布", fontsize=14)
    fig.text(0.63, 0.24, "设置网格范围、节点数与速度", fontsize=14)
    fig.text(0.84, 0.24, "输出切片和报告", fontsize=14)
    fig.savefig(OUT_DIR / "04_软件流程示意图.png", bbox_inches="tight")
    plt.close(fig)


def save_model_grid_slide() -> None:
    fig = plt.figure(figsize=(12, 6.8), facecolor="white")
    fig.text(0.07, 0.91, "模型网格:", fontsize=22, fontweight="bold")

    ax = fig.add_axes([0.08, 0.20, 0.36, 0.62])
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.00, 0.00), 1.00, 1.00, facecolor="#f5f7fb", edgecolor="#cbd5e1", linewidth=1.2))
    ax.text(0.27, 0.94, "Form_ModelGenerate", fontsize=12)
    ax.add_patch(Rectangle((0.08, 0.47), 0.78, 0.38, facecolor="#ffffff", edgecolor="#008a9a", linewidth=3))
    labels = ["X方向节点个数", "Y方向节点个数", "Z方向节点个数"]
    vals = ["21", "21", "21"]
    for i, (label, val) in enumerate(zip(labels, vals)):
        y = 0.77 - i * 0.11
        ax.text(0.12, y, label, fontsize=10, va="center")
        ax.add_patch(Rectangle((0.47, y - 0.035), 0.22, 0.07, facecolor="#ffffff", edgecolor="#94a3b8"))
        ax.text(0.58, y, val, ha="center", va="center", color="#0066cc")

    ax.add_patch(Rectangle((0.08, 0.20), 0.78, 0.24, facecolor="#ffffff", edgecolor="#e00000", linewidth=3))
    ranges = [("X方向范围", "0.00", "1000.00"), ("Y方向范围", "0.00", "1000.00"), ("Z方向范围", "0.00", "1000.00")]
    for i, (label, a, b) in enumerate(ranges):
        y = 0.38 - i * 0.07
        ax.text(0.12, y, label, fontsize=10, va="center")
        ax.add_patch(Rectangle((0.40, y - 0.025), 0.16, 0.05, facecolor="#ffffff", edgecolor="#94a3b8"))
        ax.add_patch(Rectangle((0.66, y - 0.025), 0.16, 0.05, facecolor="#ffffff", edgecolor="#94a3b8"))
        ax.text(0.48, y, a, ha="center", va="center", fontsize=9)
        ax.text(0.61, y, "-->", ha="center", va="center", fontsize=9)
        ax.text(0.74, y, b, ha="center", va="center", fontsize=9)

    add_arrow(fig, 0.46, 0.62, 0.58, 0.73)
    add_arrow(fig, 0.46, 0.39, 0.58, 0.42)
    fig.text(0.60, 0.71, "网格个数设置: 将反演体离散成三维数据体，\n节点越多分辨率越高，但计算更慢。", fontsize=17, color="#0096c7")
    fig.text(0.60, 0.41, "反演体范围: X/Y/Z 的空间范围，\n应覆盖全部震源、台站和主要射线路径。", fontsize=17, color="#e00000")
    fig.text(
        0.60,
        0.18,
        "反演最终结果保存为切片图和统计报告，\n核心计算由 wave_ct.inversion 完成。",
        fontsize=17,
        color="#111827",
    )
    fig.savefig(OUT_DIR / "05_模型网格设置说明.png", bbox_inches="tight")
    plt.close(fig)


def save_result_composite(rows: List[Dict[str, str]], velocities: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 6.8), facecolor="white")
    fig.text(0.04, 0.91, "反演后处理成果图", fontsize=22, fontweight="bold")

    ax_map = fig.add_axes([0.06, 0.16, 0.38, 0.62])
    ax_map.set_facecolor("#fff200")
    if rows:
        sx = np.array([fnum(r["震源坐标-x"]) for r in rows])
        sy = np.array([fnum(r["震源坐标-y"]) for r in rows])
        rx = np.array([fnum(r["台站坐标-x"]) for r in rows])
        ry = np.array([fnum(r["台站坐标-y"]) for r in rows])
        x_all = np.concatenate([sx, rx])
        y_all = np.concatenate([sy, ry])
        x_pad = max(20.0, float(np.ptp(x_all)) * 0.06)
        y_pad = max(20.0, float(np.ptp(y_all)) * 0.06)
        for i in range(0, len(rows), max(1, len(rows) // 120)):
            ax_map.plot([sx[i], rx[i]], [sy[i], ry[i]], color="black", alpha=0.18, linewidth=0.8)
        h = ax_map.hist2d(sx, sy, bins=38, cmap="jet", alpha=0.78)
        fig.colorbar(h[3], ax=ax_map, fraction=0.046, pad=0.02)
        ax_map.set_xlim(float(x_all.min() - x_pad), float(x_all.max() + x_pad))
        ax_map.set_ylim(float(y_all.min() - y_pad), float(y_all.max() + y_pad))
    ax_map.set_title("输入射线覆盖与震源分布")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.grid(color="black", alpha=0.18)

    ax_hist = fig.add_axes([0.55, 0.53, 0.34, 0.28])
    ax_hist.hist(velocities, bins=26, color="#d44a1c", edgecolor="white")
    ax_hist.set_title("反演输入速度统计")
    ax_hist.set_xlabel("m/s")
    ax_hist.set_ylabel("数量")
    ax_hist.axvline(REFERENCE_VMIN, color="blue", linestyle="--")
    ax_hist.axvline(REFERENCE_VMAX, color="red", linestyle="--")

    fig.text(0.54, 0.32, "验证要点:", fontsize=20, fontweight="bold")
    fig.text(0.54, 0.25, "1. 参考速度区间由当前数据稳健统计或用户参数确定。", fontsize=15)
    fig.text(0.54, 0.19, "2. 射线覆盖稀疏区域的反演结果只能作参考。", fontsize=15)
    fig.text(0.54, 0.13, "3. 切片图应结合 RMS 收敛和射线覆盖共同判断。", fontsize=15)
    fig.savefig(OUT_DIR / "06_反演成果综合图.png", bbox_inches="tight")
    plt.close(fig)


def write_report(rows: List[Dict[str, str]], distances: np.ndarray, times_s: np.ndarray, velocities: np.ndarray, detail_rows: List[Dict[str, str]]) -> None:
    kept = sum(1 for r in detail_rows if r.get("keep") == "1") if detail_rows else len(rows)
    total = len(detail_rows) if detail_rows else len(rows)
    low = int(np.sum(velocities < REFERENCE_VMIN))
    high = int(np.sum(velocities > REFERENCE_VMAX))

    method_counts: Dict[str, int] = {}
    drop_counts: Dict[str, int] = {}
    for row in detail_rows:
        method = row.get("pick_method") or row.get("method", "")
        method_counts[method] = method_counts.get(method, 0) + 1
        reason = row.get("drop_reason", "")
        if reason:
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
    lines = [
        "验证报告",
        "=" * 60,
        f"反演输入CSV: {INV_CSV}",
        f"拾取详情CSV: {DETAIL_CSV}",
        f"总候选记录: {total}",
        f"保留记录: {kept}",
        f"表观速度范围: {velocities.min():.2f}-{velocities.max():.2f} m/s",
        f"表观速度均值: {velocities.mean():.2f} m/s",
        f"参考速度区间: {REFERENCE_VMIN:.2f}-{REFERENCE_VMAX:.2f} m/s",
        f"低于参考下限数量: {low}",
        f"高于参考上限数量: {high}",
        f"传播时间范围: {times_s.min() * 1000.0:.2f}-{times_s.max() * 1000.0:.2f} ms",
        f"距离范围: {distances.min():.2f}-{distances.max():.2f} m",
        "",
        "说明",
        "  本报告只统计人工走时CSV、反演日志和实际输出图，不使用合成异常体作为精度依据。",
        "  反演精度应结合人工P波拾取质量、速度质控、RMS收敛和射线覆盖共同判断。",
        "",
        "拾取方法计数",
    ]
    for key, value in sorted(method_counts.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("剔除原因计数")
    for key, value in sorted(drop_counts.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {key}: {value}")

    if SLICE_REPORT.exists():
        lines.extend(["", "切片报告摘录", "-" * 60])
        lines.extend(SLICE_REPORT.read_text(encoding="utf-8").splitlines())

    (OUT_DIR / "验证报告.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global INV_CSV, DETAIL_CSV, SLICE_REPORT, OUT_DIR, WAVEFORM_ROOT, REFERENCE_VMIN, REFERENCE_VMAX

    parser = argparse.ArgumentParser(description="生成震动波CT反演成果与验证报告")
    parser.add_argument("--input-csv", type=Path, default=INV_CSV)
    parser.add_argument("--detail-csv", type=Path, default=DETAIL_CSV)
    parser.add_argument("--slice-report", type=Path, default=SLICE_REPORT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--waveform-root", type=Path, default=WAVEFORM_ROOT)
    parser.add_argument("--reference-vmin", type=float, default=0.0)
    parser.add_argument("--reference-vmax", type=float, default=0.0)
    args = parser.parse_args()

    INV_CSV = args.input_csv
    DETAIL_CSV = args.detail_csv
    SLICE_REPORT = args.slice_report
    OUT_DIR = args.out_dir
    WAVEFORM_ROOT = args.waveform_root

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, distances, times_s, velocities = load_inversion_rows()
    detail_rows = load_detail_rows()
    if velocities.size == 0:
        raise RuntimeError("反演输入CSV中没有可用于验证的有效走时。")
    auto_low, auto_high = np.percentile(velocities, [5.0, 95.0])
    REFERENCE_VMIN = args.reference_vmin if args.reference_vmin > 0 else float(auto_low)
    REFERENCE_VMAX = args.reference_vmax if args.reference_vmax > REFERENCE_VMIN else float(auto_high)
    if REFERENCE_VMAX <= REFERENCE_VMIN:
        REFERENCE_VMIN = max(float(velocities.min()), 1.0)
        REFERENCE_VMAX = max(float(velocities.max()), REFERENCE_VMIN + 1.0)

    save_velocity_hist(velocities)
    save_time_distance_plot(distances, times_s)
    save_pick_examples(detail_rows)
    save_workflow_slide()
    save_model_grid_slide()
    save_result_composite(rows, velocities)
    write_report(rows, distances, times_s, velocities, detail_rows)

    print(f"验证报告目录: {OUT_DIR}")
    print(f"表观速度范围: {velocities.min():.2f}-{velocities.max():.2f} m/s")
    print(f"表观速度均值: {velocities.mean():.2f} m/s")


if __name__ == "__main__":
    main()
