from __future__ import annotations

import csv
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk

import zipfile
import numpy as np
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, Iterable, List, Optional, Tuple

import customtkinter as ctk  # 引入现代化 UI 框架

from wave_ct.config import (
    MODEL_PARAMETER_KEYS,
    app_settings_path,
    discover_accoreconsole,
    load_app_settings,
    load_project_config,
    project_path_for_csv,
    save_app_settings,
    save_project_config,
)
from wave_ct.auto_strategy import extract_dataset_features, recommend_workface_grid
from wave_ct.algorithm_registry import (
    WAVECT_VERSION,
)
from wave_ct.deliverables import collect_final_result_images
from wave_ct.image_gallery import adjacent_index, can_move, fit_zoom, sort_gallery_paths, zoom_by

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

ROOT = Path(__file__).resolve().parent

# --- 设置 CustomTkinter 全局主题 ---
ctk.set_appearance_mode("Light")  
ctk.set_default_color_theme("blue")  

# --- 极简全白高级感 UI 调色板 ---
COLORS = {
    "bg": "#F8FAFC",              # 全局极浅灰背景
    "card": "#FFFFFF",            # 纯白卡片背景
    "border": "#E2E8F0",          # 极浅柔和边框
    "text_main": "#0F172A",       # 深色主文字
    "text_muted": "#64748B",      # 灰色次要文字
    "primary": "#0F172A",         # 主题色改为极客黑（高级感）
    "primary_hover": "#334155",   
    "accent": "#2563EB",          # 强调色：纯净亮蓝
    "accent_hover": "#1D4ED8",    
    "success": "#10B981",         # 成功绿
    "console_bg": "#0B1120",      # 控制台深色（在全白 UI 中形成强反差高级感）
    "console_header": "#1E293B",  # 控制台顶部
}


def path_text(path: Path) -> str:
    return str(path.resolve())


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def apparent_velocity_stats(csv_path: Path) -> Optional[Tuple[int, float, float, float]]:
    if not csv_path.exists():
        return None
    values = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                dist = ((float(row["震源坐标-x"]) - float(row["台站坐标-x"])) ** 2 +
                        (float(row["震源坐标-y"]) - float(row["台站坐标-y"])) ** 2 +
                        (float(row["震源坐标-z"]) - float(row["台站坐标-z"])) ** 2) ** 0.5
                tt = float(row["震源-台站传播时间"]) / 1000.0
                if tt > 0:
                    values.append(dist / tt)
            except:
                continue
    if not values:
        return None
    return len(values), min(values), sum(values) / len(values), max(values)


def coordinate_bounds_from_csv(csv_path: Path) -> Optional[Tuple[float, float, float, float, float, float]]:
    if not csv_path.exists():
        return None
    axes = {"x": [], "y": [], "z": []}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                for axis in axes:
                    axes[axis].extend((
                        float(row[f"震源坐标-{axis}"]),
                        float(row[f"台站坐标-{axis}"]),
                    ))
            except (KeyError, TypeError, ValueError):
                continue
    if not all(axes.values()):
        return None
    result = []
    for axis in ("x", "y", "z"):
        low, high = min(axes[axis]), max(axes[axis])
        span = high - low
        margin = max(span * 0.05, 1.0)
        result.extend((low - margin, high + margin))
    return tuple(result)


def acquisition_geometry_from_csv(csv_path: Path) -> Optional[dict]:
    if not csv_path.exists():
        return None
    geometry = {
        "source_x": [], "source_y": [], "source_z": [],
        "station_x": [], "station_y": [], "station_z": [],
        "row_count": 0,
    }
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                geometry["source_x"].append(float(row["震源坐标-x"]))
                geometry["source_y"].append(float(row["震源坐标-y"]))
                geometry["source_z"].append(float(row["震源坐标-z"]))
                geometry["station_x"].append(float(row["台站坐标-x"]))
                geometry["station_y"].append(float(row["台站坐标-y"]))
                geometry["station_z"].append(float(row["台站坐标-z"]))
                geometry["row_count"] += 1
            except (KeyError, TypeError, ValueError):
                continue
    return geometry if geometry["row_count"] else None


def boundary_bounds_from_csv(path: Path) -> Optional[Tuple[float, float, float, float]]:
    if not path.is_file():
        return None
    points = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                points.append((float(row["x"]), float(row["y"])))
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if len(points) < 3:
        return None
    xs, ys = zip(*points)
    return min(xs), max(xs), min(ys), max(ys)


class ModelDialog(ctk.CTkToplevel):
    def __init__(self, app: "CtApp"):
        super().__init__(app)
        self.app = app
        self.title("构建初始速度模型网格")
        self.geometry("560x680")
        self.minsize(540, 600)
        self.configure(fg_color=COLORS["bg"])
        self.transient(app)
        self.grab_set()
        self._build()

    def _entry(self, parent, row, label, key_a, key_b=None):
        lbl = ctk.CTkLabel(parent, text=label, text_color=COLORS["text_main"], font=ctk.CTkFont(size=13, weight="bold"))
        lbl.grid(row=row, column=0, sticky="w", padx=(20, 10), pady=12)
        
        entry_a = ctk.CTkEntry(parent, textvariable=self.app.vars[key_a], width=120, height=36, justify="center", corner_radius=6, border_color=COLORS["border"])
        entry_a.grid(row=row, column=1, padx=5, pady=12, sticky="w")
        
        if key_b:
            ctk.CTkLabel(parent, text="至", text_color=COLORS["text_muted"], font=ctk.CTkFont(size=13)).grid(row=row, column=2, padx=5)
            entry_b = ctk.CTkEntry(parent, textvariable=self.app.vars[key_b], width=120, height=36, justify="center", corner_radius=6, border_color=COLORS["border"])
            entry_b.grid(row=row, column=3, padx=5, pady=12, sticky="w")

    def _build(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(24, 16))
        
        ctk.CTkLabel(header, text="🔲 反演空间与网格参数", text_color=COLORS["text_main"], font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text="配置反演三维空间坐标范围与离散网格节点数", text_color=COLORS["text_muted"], font=ctk.CTkFont(size=13)).pack(anchor="w", pady=(4, 0))

        body = ctk.CTkFrame(self, corner_radius=12, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        box = ctk.CTkFrame(body, fg_color="transparent")
        box.pack(fill="both", expand=True, padx=10, pady=10)

        auto_chk = ctk.CTkCheckBox(
            box,
            text="根据射线数据自动优化计算范围与网格（推荐）",
            variable=self.app.vars["auto_bounds"],
            onvalue="1", offvalue="0",
            command=self._refresh_auto_bounds,
            text_color=COLORS["primary"],
            font=ctk.CTkFont(size=13, weight="bold"),
            checkbox_width=22, checkbox_height=22,
            border_color=COLORS["border"],
            fg_color=COLORS["primary"]
        )
        auto_chk.grid(row=0, column=0, columnspan=4, sticky="w", padx=16, pady=(20, 20))

        self._entry(box, 1, "X 方向节点数", "nx_nodes")
        self._entry(box, 2, "Y 方向节点数", "ny_nodes")
        self._entry(box, 3, "Z 方向节点数", "nz_nodes")
        self._entry(box, 4, "X 坐标范围 (m)", "x_min", "x_max")
        self._entry(box, 5, "Y 坐标范围 (m)", "y_min", "y_max")
        self._entry(box, 6, "Z 坐标范围 (m)", "z_min", "z_max")

        btn_save = ctk.CTkButton(
            self, text="💾 保存并更新模型步长", 
            command=self._ok, 
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            font=ctk.CTkFont(size=14, weight="bold"), 
            height=46, corner_radius=8
        )
        btn_save.pack(fill="x", padx=24, pady=(0, 24))

    def _refresh_auto_bounds(self):
        if self.app.vars["auto_bounds"].get() == "1":
            self.app.update_bounds_from_csv(show_message=True)

    def _ok(self):
        try:
            if self.app.vars["auto_bounds"].get() == "1":
                self.app.update_bounds_from_csv(show_message=False)
            nx = max(2, int(float(self.app.vars["nx_nodes"].get())))
            ny = max(2, int(float(self.app.vars["ny_nodes"].get())))
            nz = max(2, int(float(self.app.vars["nz_nodes"].get())))
            self.app.vars["dx"].set(f"{(float(self.app.vars['x_max'].get()) - float(self.app.vars['x_min'].get())) / (nx - 1):.12g}")
            self.app.vars["dy"].set(f"{(float(self.app.vars['y_max'].get()) - float(self.app.vars['y_min'].get())) / (ny - 1):.12g}")
            self.app.vars["dz"].set(f"{(float(self.app.vars['z_max'].get()) - float(self.app.vars['z_min'].get())) / (nz - 1):.12g}")
            self.app.write_console(
                f">>> 模型网格已更新: dx={self.app.vars['dx'].get()}m, dy={self.app.vars['dy'].get()}m, dz={self.app.vars['dz'].get()}m\n",
                color="success")
            self.app.refresh_summary()
            self.app.save_current_project(silent=True)
            self.destroy()
        except ValueError:
            messagebox.showerror("输入错误", "网格参数必须是合法的数字。")


class ResultImageViewer(ctk.CTkToplevel):
    """Reusable result viewer with navigation, zoom, fit and pan controls."""

    def __init__(self, master, paths: List[Path], index: int = 0):
        super().__init__(master)
        self.paths = list(paths)
        self.index = max(0, min(index, len(self.paths) - 1))
        self.original = None
        self.photo = None
        self.zoom = 1.0
        self._fit_after_id = None
        self.title("WaveCT 成果图查看器")
        self.geometry("1180x820")
        self.minsize(760, 520)
        self.configure(fg_color=COLORS["bg"])
        self.transient(master)

        toolbar = ctk.CTkFrame(self, height=52, fg_color=COLORS["card"], corner_radius=0)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)
        self.previous_button = ctk.CTkButton(toolbar, text="‹ 上一张", width=76, command=lambda: self.move(-1))
        self.previous_button.pack(side="left", padx=(12, 5), pady=9)
        self.next_button = ctk.CTkButton(toolbar, text="下一张 ›", width=76, command=lambda: self.move(1))
        self.next_button.pack(side="left", padx=5, pady=9)
        ctk.CTkButton(toolbar, text="−", width=34, command=lambda: self.change_zoom(1 / 1.2)).pack(side="left", padx=(14, 3), pady=9)
        ctk.CTkButton(toolbar, text="+", width=34, command=lambda: self.change_zoom(1.2)).pack(side="left", padx=3, pady=9)
        ctk.CTkButton(toolbar, text="适应窗口", width=82, command=self.fit_to_window).pack(side="left", padx=(8, 3), pady=9)
        ctk.CTkButton(toolbar, text="1:1", width=46, command=lambda: self.set_zoom(1.0)).pack(side="left", padx=3, pady=9)
        self.title_label = ctk.CTkLabel(toolbar, text="", text_color=COLORS["text_main"], anchor="w")
        self.title_label.pack(side="left", fill="x", expand=True, padx=12)

        body = tk.Frame(self, bg="#FFFFFF")
        body.pack(fill="both", expand=True, padx=12, pady=(10, 4))
        self.canvas = tk.Canvas(body, background="#FFFFFF", highlightthickness=0)
        self.x_scroll = tk.Scrollbar(body, orient="horizontal", command=self.canvas.xview)
        self.y_scroll = tk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.x_scroll.set, yscrollcommand=self.y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.y_scroll.grid(row=0, column=1, sticky="ns")
        self.x_scroll.grid(row=1, column=0, sticky="ew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self.info_label = ctk.CTkLabel(self, text="", text_color=COLORS["text_muted"], anchor="w")
        self.info_label.pack(fill="x", padx=16, pady=(0, 8))

        self.canvas.bind("<ButtonPress-1>", lambda event: self.canvas.scan_mark(event.x, event.y))
        self.canvas.bind("<B1-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1))
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        for key, callback in {
            "<Left>": lambda event: self.move(-1), "<Right>": lambda event: self.move(1),
            "<plus>": lambda event: self.change_zoom(1.2), "<equal>": lambda event: self.change_zoom(1.2),
            "<minus>": lambda event: self.change_zoom(1 / 1.2), "<0>": lambda event: self.fit_to_window(),
            "<1>": lambda event: self.set_zoom(1.0), "<Escape>": lambda event: self.destroy(),
        }.items():
            self.bind(key, callback)
        self.load_index(self.index, fit=True)

    def set_paths(self, paths: List[Path], index: int) -> None:
        self.paths = list(paths)
        self.load_index(index, fit=True)

    def move(self, delta: int) -> None:
        if not self.paths:
            return
        self.load_index(adjacent_index(self.index, delta, len(self.paths)), fit=True)

    def load_index(self, index: int, fit: bool = False) -> None:
        if Image is None or not self.paths:
            return
        self.index = max(0, min(index, len(self.paths) - 1))
        path = self.paths[self.index]
        try:
            with Image.open(path) as source:
                self.original = source.convert("RGB").copy()
        except Exception as exc:
            self.canvas.delete("all")
            self.info_label.configure(text=f"图片读取失败：{exc}")
            return
        self.title(path.name)
        self.title_label.configure(text=f"{self.index + 1}/{len(self.paths)}  {path.name}")
        self.previous_button.configure(state="normal" if can_move(self.index, -1, len(self.paths)) else "disabled")
        self.next_button.configure(state="normal" if can_move(self.index, 1, len(self.paths)) else "disabled")
        if fit:
            self.after_idle(self.fit_to_window)
        else:
            self.render()

    def _on_canvas_configure(self, _event) -> None:
        if self._fit_after_id is not None:
            self.after_cancel(self._fit_after_id)
        self._fit_after_id = self.after(180, self.fit_to_window)

    def _on_mousewheel(self, event) -> None:
        self.change_zoom(1.15 if event.delta > 0 else 1 / 1.15)

    def fit_to_window(self) -> None:
        self._fit_after_id = None
        if self.original is None:
            return
        self.set_zoom(fit_zoom(self.original.size, (self.canvas.winfo_width(), self.canvas.winfo_height())))

    def change_zoom(self, factor: float) -> None:
        self.set_zoom(zoom_by(self.zoom, factor))

    def set_zoom(self, zoom: float) -> None:
        self.zoom = max(0.1, min(8.0, float(zoom)))
        self.render()

    def render(self) -> None:
        if self.original is None or ImageTk is None:
            return
        width = max(1, round(self.original.width * self.zoom))
        height = max(1, round(self.original.height * self.zoom))
        rendered = self.original.resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(rendered)
        viewport_w = max(1, self.canvas.winfo_width())
        viewport_h = max(1, self.canvas.winfo_height())
        full_w, full_h = max(width, viewport_w), max(height, viewport_h)
        x, y = max(0, (viewport_w - width) // 2), max(0, (viewport_h - height) // 2)
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, full_w, full_h))
        self.info_label.configure(text=f"{self.paths[self.index]}    |    缩放 {self.zoom * 100:.0f}%    |    滚轮缩放，拖拽平移，←/→ 切换")


class CtApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.configure(fg_color=COLORS["bg"])
        self.title(f"WaveCT {WAVECT_VERSION} - 三维 P 波走时层析成像")
        self.geometry("1480x900")
        self.minsize(1180, 720)

        self.proc = None
        self.log_queue = queue.Queue()
        self.vars = {}
        self.app_settings = load_app_settings()
        self.project_file: Optional[Path] = None
        self.dataset_config: dict = {}
        self.project_model_extras: dict = {}
        self.result_image_paths: List[Path] = []
        self.image_viewer: Optional[ResultImageViewer] = None
        self._task_sequence = 0

        self._init_vars()
        self._apply_app_settings_environment()
        self._build_ui()
        
        default_csv = Path(self.vars["output_csv"].get())
        default_project = project_path_for_csv(default_csv)
        if default_project.is_file():
            self.load_project(default_project, show_message=False)
        else:
            self.project_file = default_project
            self.vars["project_file"].set(str(default_project))
            
        self.refresh_all()
        self.refresh_result_images()
        self.after(100, self._poll_log_queue)

    def _init_vars(self):
        paths = {
            "waveform_root": ROOT / "正演波形数据集", "model": ROOT / "picker_model.onnx",
            "station_file": ROOT / "台站（台站编号-台网代码-Y-X-0-Z）.txt",
            "output_csv": ROOT / "标波输出数据集" / "反演数据集.csv",
            "detail_csv": ROOT / "标波输出数据集" / "反演数据集_detail.csv",
            "pick_report": ROOT / "标波输出数据集" / "反演数据集_report.txt",
            "slice_dir": ROOT.parent / "成像效果展示", "verify_dir": ROOT.parent / "成像效果展示"
        }
        for k, v in paths.items(): self.vars[k] = tk.StringVar(value=path_text(v))
        self.vars["boundary_file"] = tk.StringVar(value="")
        self.vars["mapa_file"] = tk.StringVar(value="")
        self.vars["dwg_file"] = tk.StringVar(value="")
        self.vars["basemap_file"] = tk.StringVar(value="")
        self.vars["pick_audit_csv"] = tk.StringVar(value="")
        self.vars["evidence_csv"] = tk.StringVar(value="")
        self.vars["project_name"] = tk.StringVar(value="")
        self.vars["project_file"] = tk.StringVar(value="")
        self.vars["report_template"] = tk.StringVar(value="auto")
        self.vars["cad_x_offset"] = tk.StringVar(value="0")
        self.vars["cad_y_offset"] = tk.StringVar(value="0")

        vals = {
            "expected_sources": "0", "expected_stations_per_source": "0", "waveform_pattern": "*.FXZ.semv",
            "source_pattern": "source_*", "source_coord_filename": "output_list_sources.txt",
            "vmin": "0", "vmax": "0", "confidence": "-1", "min_snr": "3.0",
            "picker_mode": "signal", "search_vmin": "2500", "search_vmax": "8000",
            "onset_fraction": "0.70", "smooth_samples": "15", "min_consecutive": "3",
            "mode": "generic", "dx": "50", "dy": "50", "dz": "50",
            "n_outer": "18", "n_lsqr": "300", "solver_method": "sirt",
            "sirt_iterations": "300", "sirt_omega": "0.30", "sirt_step_damp": "1.0",
            "sirt_tolerance": "1e-8", "sirt_auto_tune": "0", "sirt_tune_maxiter": "15",
            "sirt_tune_popsize": "6", "sirt_tune_iterations": "20",
            # The verified 728 probe is the production visual baseline.  It
            # deliberately stays explicit rather than allowing a saved project
            # to silently switch the one-click workflow back to a different
            # DE optimum.
            "reference_sirt_profile": "probe_728",
            "alpha_reg": "4.0", "step_damp": "0.2",
            "vmin_qc": "0", "vmax_qc": "0", "vmin_model": "0", "vmax_model": "0",
            "background_velocity": "0", "min_ray_coverage": "0", "coverage_weight_exponent": "1.5", "min_rays": "20",
            "validation_fraction": "0.2", "huber_delta": "1.5", "background_damping": "1.5",
            "model_damping": "0",
            "regularize_total_model": "1", "curvature_reg_factor": "0.25", "curvature_z_factor": "0.5",
            "source_static_damping": "10.0", "global_time_damping": "1.0",
            "max_time_correction": "0.05", "edge_preserving_tv": "0",
            "joint_sparsity": "0", "wavelet_levels": "2", "wavelet_threshold_factor": "0.80",
            "hierarchical_parameterization": "0", "hierarchical_split_rays": "5",
            "hierarchical_min_block_x": "3", "hierarchical_min_block_y": "3",
            "differential_times": "0", "differential_weight": "0.5",
            "ray_length_normalization": "0", "allow_outside_rays": "1",
            "event_centered_qc": "1",
            "auto_algorithm": "0", "auto_cv_seeds": "11,23,41",
            "auto_pilot_outer": "24", "auto_pilot_lsqr": "160",
            "deep_reparameterization": "0",
            "deep_reparam_width": "24",
            "deep_reparam_depth": "3",
            "deep_reparam_full_epochs": "350",
            "deep_reparam_starts": "3",
            "deep_reparam_device": "cpu",
            "nx_nodes": "21", "ny_nodes": "21", "nz_nodes": "21",
            "auto_bounds": "1", "x_min": "0", "x_max": "0", "y_min": "0", "y_max": "0", "z_min": "0", "z_max": "0",
            "slice_z": "", "plot_style": "rectangular",
            "presentation_vmin": "0", "presentation_vmax": "0",
            # The reference-SIRT presentation used a 0.65-cell local blur.
            # Larger defaults visually merge nearby anomalies into broad blocks.
            "presentation_sigma": "0.65", "anomaly_limit": "0.30",
            # Zero means that the presentation window follows the declared
            # workface boundary exactly, matching the reference-SIRT probe.
            "workface_view_padding": "0.0",
        }
        for k, v in vals.items(): self.vars[k] = tk.StringVar(value=v)

    def _apply_app_settings_environment(self) -> None:
        core_console = str(self.app_settings.get("autocad_core_console", "")).strip()
        cache_dir = str(self.app_settings.get("cad_cache_dir", "")).strip()
        if core_console:
            os.environ["AUTOCAD_CORE_CONSOLE"] = core_console
        if cache_dir:
            os.environ["WAVECT_CAD_CACHE"] = cache_dir

    def _create_top_nav_btn(self, parent, step_num, text, command, btn_type="default"):
            """用于顶部导航栏的按钮样式"""
            styles = {
                "default": ("transparent", COLORS["bg"], COLORS["text_main"], 1, COLORS["border"]),
                # 修复：CustomTkinter 不允许 border_color 为 "transparent"，因为 border_width 为 0，换成对应的底色即可
                "primary": (COLORS["primary"], COLORS["primary_hover"], "#FFFFFF", 0, COLORS["primary"]),
                "accent": (COLORS["accent"], COLORS["accent_hover"], "#FFFFFF", 0, COLORS["accent"]),
            }
            fg, hover, txt_color, border, border_color = styles.get(btn_type, styles["default"])
            
            prefix = f"{step_num} " if step_num else ""
            btn = ctk.CTkButton(
                parent, 
                text=f"{prefix}{text}", 
                command=command, 
                fg_color=fg, hover_color=hover, text_color=txt_color,
                border_width=border, border_color=border_color,
                height=38, corner_radius=8,
                font=ctk.CTkFont(size=13, weight="bold")
            )
            return btn

    def _build_ui(self):
        # ================= 1. 顶部纯白导航栏 (Top Navbar) =================
        header = ctk.CTkFrame(self, height=72, corner_radius=0, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Logo 区域
        logo_area = ctk.CTkFrame(header, fg_color="transparent")
        logo_area.pack(side="left", padx=24, fill="y", pady=12)
        
        logo_icon = ctk.CTkFrame(logo_area, fg_color=COLORS["primary"], corner_radius=8, width=40, height=40)
        logo_icon.pack(side="left", padx=(0, 12))
        logo_icon.pack_propagate(False)
        ctk.CTkLabel(logo_icon, text="CT", font=ctk.CTkFont(size=15, weight="bold"), text_color="white").place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(logo_area, text="WaveCT", text_color=COLORS["text_main"], font=ctk.CTkFont("Segoe UI", 18, "bold")).pack(side="left")

        # 导航按钮区域 (靠右对齐)
        nav_area = ctk.CTkFrame(header, fg_color="transparent")
        nav_area.pack(side="right", padx=24, fill="y", pady=16)

        # 基础配置组
        self._create_top_nav_btn(nav_area, "", "项目设置", self.open_project_settings).pack(side="left", padx=6)
        self._create_top_nav_btn(nav_area, "", "导入数据", self.import_raw_dataset).pack(side="left", padx=6)
        
        # 分割线
        sep = ctk.CTkFrame(nav_area, width=1, fg_color=COLORS["border"])
        sep.pack(side="left", fill="y", padx=12, pady=4)

        # 核心流程组
        self._create_top_nav_btn(nav_area, "01", "数据复核", self.open_manual_picker).pack(side="left", padx=6)
        self._create_top_nav_btn(nav_area, "02", "网格设置", lambda: ModelDialog(self)).pack(side="left", padx=6)
        self._create_top_nav_btn(nav_area, "03", "一键处理", self.run_one_click_inversion, btn_type="accent").pack(side="left", padx=6)

        # ================= 2. 主工作区 =================
        main_area = ctk.CTkFrame(self, fg_color="transparent")
        main_area.pack(fill="both", expand=True, padx=24, pady=(16, 24))

        # 当前项目路径展示 (超简窄条)
        path_bar = ctk.CTkFrame(main_area, height=44, corner_radius=8, fg_color="transparent")
        path_bar.pack(fill="x", pady=(0, 12))
        path_bar.pack_propagate(False)
        ctk.CTkLabel(path_bar, text="当前走时数据:", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLORS["text_muted"]).pack(side="left")
        ctk.CTkLabel(path_bar, textvariable=self.vars["output_csv"], text_color=COLORS["accent"], font=ctk.CTkFont("Consolas", 13, "bold")).pack(side="left", padx=12)
        ctk.CTkButton(path_bar, text="🔄 刷新全局状态", command=self.refresh_all, width=120, height=32, corner_radius=6, fg_color="transparent", border_color=COLORS["border"], border_width=1, text_color=COLORS["text_main"], hover_color=COLORS["card"]).pack(side="right")

        # 核心分栏布局 (Grid - 左边紧凑，右边宽大)
        content_split = ctk.CTkFrame(main_area, fg_color="transparent")
        content_split.pack(fill="both", expand=True)
        
        # 设置黄金比例：左侧仪表盘占 2.5 份，右侧画廊/终端占 7.5 份
        content_split.grid_columnconfigure(0, weight=25, minsize=320)
        content_split.grid_columnconfigure(1, weight=75, minsize=800)
        content_split.grid_rowconfigure(0, weight=1)

        # ----------------- A. 左侧：精简版仪表盘 -----------------
        dashboard_panel = ctk.CTkFrame(content_split, corner_radius=12, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        dashboard_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        dash_header = ctk.CTkFrame(dashboard_panel, height=54, corner_radius=12, fg_color="transparent")
        dash_header.pack(fill="x", padx=16, pady=(4, 0))
        dash_header.pack_propagate(False)
        ctk.CTkLabel(dash_header, text="项目概览", text_color=COLORS["text_main"], font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        ctk.CTkButton(dash_header, text="参数设置", command=self.open_advanced_params, width=76, height=28, corner_radius=6, fg_color="transparent", border_width=1, border_color=COLORS["border"], text_color=COLORS["text_main"], hover_color=COLORS["bg"]).pack(side="right", pady=12)

        self.summary_scroll = ctk.CTkScrollableFrame(dashboard_panel, fg_color="transparent")
        self.summary_scroll.pack(fill="both", expand=True, padx=8, pady=8)
        self._build_summary_cards(self.summary_scroll)

        # ----------------- B. 右侧：控制台 + 成果画廊 -----------------
        right_panel = ctk.CTkFrame(content_split, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        # 控制台占比调小，画廊占比调大 (3:7)
        right_panel.grid_rowconfigure(0, weight=3) 
        right_panel.grid_rowconfigure(1, weight=7) 

        # 1. 任务日志
        console_container = ctk.CTkFrame(right_panel, corner_radius=12, fg_color=COLORS["console_bg"], border_width=1, border_color="#334155")
        console_container.grid(row=0, column=0, sticky="nsew", pady=(0, 16))
        
        console_head = ctk.CTkFrame(console_container, height=36, corner_radius=12, fg_color=COLORS["console_header"])
        console_head.pack(fill="x")
        console_head.pack_propagate(False)
        
        ctk.CTkLabel(console_head, text="任务日志", text_color="#E2E8F0", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=16)
        ctk.CTkButton(console_head, text="清空", command=lambda: self.console.delete(1.0, "end"), width=50, height=22, corner_radius=4, fg_color="transparent", border_width=1, border_color="#475569", text_color="#CBD5E1", hover_color="#334155").pack(side="right", padx=12)

        self.console = ctk.CTkTextbox(console_container, font=ctk.CTkFont("Cascadia Code", 13), wrap="word", fg_color="transparent", text_color="#F8FAFC", border_width=0)
        self.console.pack(fill="both", expand=True, padx=8, pady=8)
        
        self.console.tag_config("info", foreground="#94A3B8")       
        self.console.tag_config("success", foreground="#34D399")    
        self.console.tag_config("error", foreground="#F87171")      
        self.console.tag_config("cmd", foreground="#38BDF8")        
        self.console.tag_config("warning", foreground="#FBBF24")    

        # 2. 成果预览巨幕画廊 (全白卡片)
        preview_container = ctk.CTkFrame(right_panel, corner_radius=12, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        preview_container.grid(row=1, column=0, sticky="nsew")

        preview_header = ctk.CTkFrame(preview_container, height=48, corner_radius=12, fg_color="transparent")
        preview_header.pack(fill="x", padx=16, pady=4)
        preview_header.pack_propagate(False)
        ctk.CTkLabel(preview_header, text="成果图浏览", text_color=COLORS["text_main"], font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        self.gallery_count_label = ctk.CTkLabel(preview_header, text="", text_color=COLORS["text_muted"], font=ctk.CTkFont(size=12))
        self.gallery_count_label.pack(side="left", padx=10)
        ctk.CTkButton(preview_header, text="🔄 刷新图片", command=self.refresh_result_images, width=80, height=30, corner_radius=6, fg_color="transparent", border_width=1, border_color=COLORS["border"], text_color=COLORS["text_main"], hover_color=COLORS["bg"]).pack(side="right")

        self.preview_scroll = ctk.CTkScrollableFrame(preview_container, fg_color="transparent")
        self.preview_scroll.pack(fill="both", expand=True, padx=8, pady=8)
        self.preview_images: List[object] = []

        # ================= 3. 底部极简状态栏 =================
        self.status_bar = ctk.CTkFrame(self, height=36, corner_radius=0, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(self.status_bar, text="🟢 就绪", text_color=COLORS["success"], font=ctk.CTkFont(size=12, weight="bold"))
        self.status_label.pack(side="left", padx=24)
        ctk.CTkLabel(self.status_bar, text=f"WaveCT {WAVECT_VERSION} | P 波走时层析成像", text_color=COLORS["text_muted"], font=ctk.CTkFont(size=11)).pack(side="right", padx=24)


    def _build_summary_cards(self, parent):
        def create_card(title, icon):
            card = ctk.CTkFrame(parent, corner_radius=8, fg_color=COLORS["bg"], border_width=1, border_color=COLORS["border"])
            card.pack(fill="x", pady=(0, 12))
            
            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=12, pady=(10, 4))
            ctk.CTkLabel(header, text=f"{icon} {title}", text_color=COLORS["text_main"], font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
            
            content = ctk.CTkFrame(card, fg_color="transparent")
            content.pack(fill="x", padx=12, pady=(2, 10))
            
            content.grid_columnconfigure(0, weight=1)
            content.grid_columnconfigure(1, weight=1)
            return content

        self._card_row_index = 0
        def create_row(parent, label, default_val):
            lbl = ctk.CTkLabel(parent, text=label, text_color=COLORS["text_muted"], font=ctk.CTkFont(size=12))
            lbl.grid(row=self._card_row_index, column=0, sticky="w", pady=2)
            val_lbl = ctk.CTkLabel(parent, text=default_val, text_color=COLORS["text_main"], font=ctk.CTkFont(size=12, weight="bold"), anchor="e", justify="right")
            val_lbl.grid(row=self._card_row_index, column=1, sticky="e", pady=2)
            self._card_row_index += 1
            return val_lbl

        data_content = create_card("数据状态", "📊")
        self._card_row_index = 0
        self.lbl_project_name = create_row(data_content, "项目名", "未命名")
        self.lbl_csv_name = create_row(data_content, "CSV", "未导入")
        self.lbl_basemap = create_row(data_content, "底图", "无")
        self.lbl_row_count = create_row(data_content, "记录数", "0")
        self.lbl_velocity_range = create_row(data_content, "视速区间", "暂无")
        self.lbl_velocity_mean = create_row(data_content, "平均速度", "暂无")

        grid_content = create_card("网格设定", "🌐")
        self._card_row_index = 0
        self.lbl_grid_nodes = create_row(grid_content, "节点数", "21×21×21")
        self.lbl_grid_steps = create_row(grid_content, "步长", "50,50,50")
        self.lbl_grid_range = create_row(grid_content, "范围", "0-1000m")

        qc_content = create_card("物理质控", "🛡️")
        self._card_row_index = 0
        self.lbl_qc_input = create_row(qc_content, "异常过滤", "自动")
        self.lbl_qc_model = create_row(qc_content, "模型速度", "自动")
        self.lbl_time_correction = create_row(qc_content, "时间修正", "联合反演")

        out_content = create_card("输出", "📁")
        self.lbl_output_dir = ctk.CTkLabel(out_content, text="暂无", text_color=COLORS["text_muted"], font=ctk.CTkFont("Consolas", 11), justify="left", wraplength=200)
        self.lbl_output_dir.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

    def _build_params(self, parent, variables: Optional[Dict[str, tk.StringVar]] = None):
        variables = self.vars if variables is None else variables
        frame = ctk.CTkFrame(
            parent, corner_radius=10, border_width=1,
            border_color="#E2E8F0", fg_color="#FFFFFF",
        )
        frame.pack(fill="x", padx=10, pady=(0, 12))
        ctk.CTkLabel(
            frame, text="SIRT 复现与成果显示设置",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#0F172A",
        ).pack(anchor="w", padx=14, pady=(12, 6))
        ctk.CTkCheckBox(
            frame, text="实验：自动比较候选算法（生产一键处理默认使用 SIRT）",
            variable=variables["auto_algorithm"], onvalue="1", offvalue="0",
            text_color="#0F172A",
        ).pack(anchor="w", padx=14, pady=(2, 10))
        grid = ctk.CTkFrame(frame, fg_color="#FFFFFF")
        grid.pack(fill="x", padx=14, pady=(0, 12))
        grid.columnconfigure(1, weight=1)
        fields = [
            ("SIRT 参数档案（probe_728=已验证 728 基线；auto=按几何选择）", "reference_sirt_profile"),
            ("切片标高（逗号分隔；留空输出全部网格层）", "slice_z"),
            ("模型速度下限 m/s（0 为自动；原生求解器）", "vmin_model"),
            ("模型速度上限 m/s（0 为自动；原生求解器）", "vmax_model"),
            ("成果色标下限 m/s（0 为自动；仅显示）", "presentation_vmin"),
            ("成果色标上限 m/s（0 为自动；仅显示）", "presentation_vmax"),
            ("展示平滑强度（建议 0.65）", "presentation_sigma"),
            ("工作面图边缘留白比例（0=紧贴边界）", "workface_view_padding"),
        ]
        for row, (label, key) in enumerate(fields):
            ctk.CTkLabel(grid, text=label, text_color="#475569").grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=5,
            )
            ctk.CTkEntry(
                grid, textvariable=variables[key], width=150,
                fg_color="#FFFFFF", text_color="#0F172A", border_color="#CBD5E1",
            ).grid(row=row, column=1, sticky="e", pady=5)
        ctk.CTkLabel(
            parent,
            text=("说明：兼容 SIRT 仅使用其网格、迭代、参数档案和切片设置；"
                  "模型约束、静校正、TV 等由原生求解器使用。成果色标和平滑只影响显示，不修改定量速度模型。"),
            text_color="#64748B", wraplength=470, justify="left",
        ).pack(fill="x", padx=18, pady=(0, 12))

    def open_advanced_params(self):
        win = ctk.CTkToplevel(self)
        win.title("参数设置")
        win.geometry("600x760")
        win.transient(self)
        win.configure(fg_color=COLORS["bg"])

        header = ctk.CTkFrame(win, height=60, corner_radius=0, fg_color=COLORS["card"])
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="参数设置", text_color=COLORS["text_main"], font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=20)

        scroll_area = ctk.CTkScrollableFrame(win, fg_color="transparent")
        scroll_area.pack(fill="both", expand=True, pady=10)
        pending = {key: tk.StringVar(value=value.get()) for key, value in self.vars.items()}
        self._build_params(scroll_area, pending)

        footer = ctk.CTkFrame(win, fg_color="transparent")
        footer.pack(fill="x", padx=20, pady=15)
        def apply_params() -> None:
            for key, value in pending.items():
                self.vars[key].set(value.get())
            self.refresh_summary()
            self.save_current_project(silent=True)
            win.destroy()

        ctk.CTkButton(footer, text="取消", height=38, corner_radius=8, fg_color="transparent", border_width=1, border_color=COLORS["border"], text_color=COLORS["text_main"], command=win.destroy).pack(side="right", padx=(8, 0))
        ctk.CTkButton(footer, text="应用", height=38, corner_radius=8, fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"], font=ctk.CTkFont(weight="bold", size=13), command=apply_params).pack(side="right")

    def write_console(self, text, color="info"):
        self.console.insert("end", text, tags=color)
        self.console.see("end")

    def set_status(self, text, state="idle"):
        palette = {"idle": (COLORS["success"], "🟢"), "running": ("#F59E0B", "⚡"), "error": ("#EF4444", "🔴")}
        color, icon = palette.get(state, palette["idle"])
        self.status_label.configure(text=f"{icon} {text}", text_color=color)

    def refresh_summary(self):
        csv_path = Path(self.vars["output_csv"].get())
        row_count = max(0, count_csv_rows(csv_path))
        stats = apparent_velocity_stats(csv_path)
        
        self.lbl_project_name.configure(text=self.vars["project_name"].get().strip() or "未命名")
        if csv_path.exists():
            self.lbl_csv_name.configure(text=csv_path.name, text_color=COLORS["accent"])
        else:
            self.lbl_csv_name.configure(text="未导入", text_color="#EF4444")

        basemap_text = self.vars["basemap_file"].get().strip()
        if basemap_text and Path(basemap_text).is_file():
            self.lbl_basemap.configure(text=Path(basemap_text).name, text_color=COLORS["success"])
        else:
            self.lbl_basemap.configure(text="无", text_color=COLORS["text_muted"])

        self.lbl_row_count.configure(text=f"{row_count:,}")

        if stats:
            self.lbl_velocity_range.configure(text=f"{stats[1]:.0f}-{stats[3]:.0f}")
            self.lbl_velocity_mean.configure(text=f"{stats[2]:.0f} m/s")
        else:
            self.lbl_velocity_range.configure(text="暂无")
            self.lbl_velocity_mean.configure(text="暂无")

        nx, ny, nz = self.vars['nx_nodes'].get(), self.vars['ny_nodes'].get(), self.vars['nz_nodes'].get()
        self.lbl_grid_nodes.configure(text=f"{nx}×{ny}×{nz}")

        dx, dy, dz = self.vars['dx'].get(), self.vars['dy'].get(), self.vars['dz'].get()
        self.lbl_grid_steps.configure(text=f"{dx},{dy},{dz}")
        
        try:
            x_min, x_max = float(self.vars['x_min'].get()), float(self.vars['x_max'].get())
            y_min, y_max = float(self.vars['y_min'].get()), float(self.vars['y_max'].get())
            self.lbl_grid_range.configure(text=f"X:{x_min:.0f}-{x_max:.0f} Y:{y_min:.0f}-{y_max:.0f}")
        except ValueError:
             self.lbl_grid_range.configure(text="未配置")

        try:
            qc_low = float(self.vars["vmin_qc"].get())
            qc_high = float(self.vars["vmax_qc"].get())
            model_low = float(self.vars["vmin_model"].get())
            model_high = float(self.vars["vmax_model"].get())
            qc_text = "自动" if qc_low <= 0 or qc_high <= 0 else f"{qc_low:g}-{qc_high:g}"
            model_text = "自动" if model_low <= 0 or model_high <= 0 else f"{model_low:g}-{model_high:g}"
        except ValueError:
            qc_text = model_text = "参数无效"
        
        self.lbl_qc_input.configure(text=qc_text)
        self.lbl_qc_model.configure(text=model_text)
        self.lbl_time_correction.configure(text="联合反演")
        self.lbl_output_dir.configure(text=self.vars["slice_dir"].get())

    def refresh_all(self):
        if self.vars["auto_bounds"].get() == "1":
            self.update_bounds_from_csv(show_message=False)
        self.refresh_summary()
        stats = apparent_velocity_stats(Path(self.vars["output_csv"].get()))
        row_count = count_csv_rows(Path(self.vars['output_csv'].get()))
        self.write_console(f"--- 系统状态刷新 ---\n", color="cmd")
        self.write_console(f"当前输入表: {self.vars['output_csv'].get()}\n", color="info")
        if stats:
            self.write_console(
                f"数据总数: {row_count} 条\n视速度统计: Min={stats[1]:.1f}, Mean={stats[2]:.1f}, Max={stats[3]:.1f} (m/s)\n\n",
                color="info")
        else:
            self.write_console(f"未检测到有效数据，请先导入CSV。\n\n", color="info")

    def refresh_result_images(self):
        for child in self.preview_scroll.winfo_children():
            child.destroy()
        self.preview_images = []
        previous_path = self.result_image_paths[self.image_viewer.index] if self.image_viewer and self.image_viewer.winfo_exists() and self.result_image_paths else None

        out_dir = Path(self.vars["slice_dir"].get())
        images = [(path, "最终成果图") for path in sort_gallery_paths(collect_final_result_images(out_dir))]
        self.result_image_paths = [path for path, _ in images]
        if hasattr(self, "gallery_count_label"):
            self.gallery_count_label.configure(text=f"{len(images)} 张")
        
        if not images:
            if hasattr(self, "gallery_count_label"):
                self.gallery_count_label.configure(text="0 张")
            ctk.CTkLabel(self.preview_scroll, text="输出目录中暂无结果图。", text_color=COLORS["text_muted"]).pack(pady=40)
            return

        if Image is None:
            ctk.CTkLabel(self.preview_scroll, text="缺少 Pillow 库。", text_color="#F87171").pack(pady=30)
            return

        grid_frame = ctk.CTkFrame(self.preview_scroll, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True)

        for idx, (image_path, image_source) in enumerate(images):
            card = ctk.CTkFrame(grid_frame, corner_radius=10, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
            card.grid(row=idx // 2, column=idx % 2, padx=10, pady=10, sticky="nsew")

            try:
                with Image.open(image_path) as source:
                    pil_image = source.convert("RGB").copy()
                pil_image.thumbnail((360, 220), Image.Resampling.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
                self.preview_images.append(ctk_img)
                
                img_lbl = ctk.CTkLabel(card, image=ctk_img, text="", cursor="hand2")
                img_lbl.pack(padx=12, pady=(12, 6))
                img_lbl.bind("<Button-1>", lambda event, p=image_path: self.open_result_image(p))

                title = self._result_image_title(image_path)
                ctk.CTkLabel(card, text=title, text_color=COLORS["text_main"], font=ctk.CTkFont(size=13, weight="bold"), wraplength=300).pack(padx=12, pady=(0, 12))
            except Exception as exc:
                ctk.CTkLabel(card, text=f"加载失败\n{image_path.name}").pack(padx=12, pady=24)

        for col in range(2):
            grid_frame.columnconfigure(col, weight=1)
        if self.image_viewer is not None and self.image_viewer.winfo_exists():
            retained = self.result_image_paths.index(previous_path) if previous_path in self.result_image_paths else 0
            self.image_viewer.set_paths(self.result_image_paths, retained)

    @staticmethod
    def _result_image_title(image_path: Path) -> str:
        stem = image_path.stem
        fixed = {
            "source_distribution": "震源分布图",
            "ray_coverage_plan": "反演射线覆盖图（平面）",
            "ray_coverage_3d": "反演射线覆盖图（三维）",
            "rms_convergence": "反演收敛曲线",
        }
        if stem in fixed:
            return fixed[stem]
        if stem.startswith("DNR初始化不确定性_z"):
            return f"{stem.removeprefix('DNR初始化不确定性_z')}标高DNR分散度"
        if stem.startswith("射线可靠覆盖_z"):
            return f"{stem.removeprefix('射线可靠覆盖_z')}标高射线覆盖边界"
        if stem.startswith("velocity_z"):
            return f"{stem.removeprefix('velocity_z')}标高速度反演图"
        if stem.startswith("yanbei_velocity_z"):
            return f"{stem.removeprefix('yanbei_velocity_z')}标高煤层CT结果"
        if stem.startswith("surfer_style_velocity_z"):
            return f"{stem.removeprefix('surfer_style_velocity_z')}标高CT平滑图"
        if stem.startswith("anomaly_z"):
            return f"{stem.removeprefix('anomaly_z')}标高波速异常 An"
        if stem.startswith("gradient_z"):
            return f"{stem.removeprefix('gradient_z')}标高波速梯度图"
        return image_path.name

    def _collect_report_images(
        self,
        out_dir: Path,
        newer_than: float = float("-inf"),
    ) -> List[Tuple[Path, str]]:
        report_roots = [out_dir]
        if out_dir.parent != out_dir:
            report_roots.append(out_dir.parent)

        report_files: List[Path] = []
        for root in report_roots:
            if root.is_dir():
                report_files.extend(root.glob("*.docx"))

        collected: List[Tuple[Path, str]] = []
        seen_reports = set()
        cache_root = Path(tempfile.gettempdir()) / "ct_report_previews"
        for report_path in sorted(report_files):
            try:
                if report_path.stat().st_mtime < newer_than:
                    continue
            except OSError:
                continue
            resolved = str(report_path.resolve()).lower()
            if resolved in seen_reports:
                continue
            seen_reports.add(resolved)
            try:
                stamp = f"{report_path.stem}_{report_path.stat().st_mtime_ns}"
                safe_stamp = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stamp)
                report_cache = cache_root / safe_stamp
                report_cache.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(report_path) as archive:
                    media = [name for name in archive.namelist() if name.startswith("word/media/")]
                    for media_name in media:
                        suffix = Path(media_name).suffix.lower()
                        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                            continue
                        target = report_cache / Path(media_name).name
                        if not target.exists():
                            target.write_bytes(archive.read(media_name))
                        collected.append((target, f"报表图片 · {report_path.stem}"))
            except (OSError, zipfile.BadZipFile, KeyError) as exc:
                self.write_console(f">>> 报表图片读取失败: {report_path.name}: {exc}\n", color="warning")
        return collected

    def open_result_image(self, image_path: Path):
        if Image is None or not image_path.exists():
            return
        paths = self.result_image_paths or [image_path]
        try:
            index = paths.index(image_path)
        except ValueError:
            paths, index = [image_path], 0
        if self.image_viewer is not None and self.image_viewer.winfo_exists():
            self.image_viewer.set_paths(paths, index)
            self.image_viewer.deiconify()
            self.image_viewer.lift()
            self.image_viewer.focus_force()
            return
        self.image_viewer = ResultImageViewer(self, paths, index)

    # ================= 核心工具方法补充完整 =================
    def _tool_dialog(self, title: str, width: int = 500, height: int = 320) -> tuple[ctk.CTkToplevel, ctk.CTkFrame]:
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry(f"{width}x{height}")
        win.configure(fg_color=COLORS["bg"])
        win.transient(self)

        header = ctk.CTkFrame(win, height=56, corner_radius=0, fg_color=COLORS["card"])
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text=title, font=ctk.CTkFont(size=16, weight="bold"), text_color=COLORS["text_main"]).pack(side="left", padx=20)

        body = ctk.CTkFrame(win, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=20)
        return win, body

    def _dialog_action(self, parent, text: str, command, primary: bool = False):
        btn = ctk.CTkButton(
            parent, text=text, command=command, height=42, corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["primary"] if primary else "transparent",
            border_width=0 if primary else 1,
            border_color=COLORS["border"],
            text_color="white" if primary else COLORS["text_main"],
            hover_color=COLORS["primary_hover"] if primary else COLORS["card"]
        )
        btn.pack(fill="x", pady=6)
        return btn

    def _current_project_config(self) -> dict:
        def number(key: str) -> float:
            try:
                return float(self.vars[key].get())
            except (TypeError, ValueError):
                return 0.0

        csv_text = self.vars["output_csv"].get().strip()
        project_name = self.vars["project_name"].get().strip()
        if not project_name and csv_text:
            project_name = Path(csv_text).parent.name or Path(csv_text).stem
        model_config = dict(self.project_model_extras)
        model_config.update({
            key: self.vars[key].get()
            for key in MODEL_PARAMETER_KEYS
            if key in self.vars
        })
        return {
            "project_name": project_name,
            "report_template": self.vars["report_template"].get().strip() or "auto",
            "dataset": dict(self.dataset_config),
            "inputs": {
                "travel_time_csv": csv_text,
                "detail_csv": self.vars["detail_csv"].get().strip(),
                "pick_audit_csv": self.vars["pick_audit_csv"].get().strip(),
                "waveform_root": self.vars["waveform_root"].get().strip(),
                "station_file": self.vars["station_file"].get().strip(),
                "evidence_csv": self.vars["evidence_csv"].get().strip(),
            },
            "workface": {
                "boundary_file": self.vars["boundary_file"].get().strip(),
                "basemap_file": (
                    self.vars["basemap_file"].get().strip()
                    or self.vars["dwg_file"].get().strip()
                    or self.vars["mapa_file"].get().strip()
                ),
                "mapa_file": self.vars["mapa_file"].get().strip(),
                "cad_x_offset": number("cad_x_offset"),
                "cad_y_offset": number("cad_y_offset"),
            },
            "outputs": {"directory": self.vars["slice_dir"].get().strip()},
            "model": model_config,
        }

    def _apply_project_config(self, config: dict, project_path: Path) -> None:
        self.project_file = project_path.resolve()
        self.vars["project_file"].set(str(self.project_file))
        self.vars["project_name"].set(str(config.get("project_name", "")))
        self.vars["report_template"].set(str(config.get("report_template", "auto")))
        self.dataset_config = dict(config.get("dataset", {}))

        inputs = config.get("inputs", {})
        for config_key, var_key in (
            ("travel_time_csv", "output_csv"),
            ("detail_csv", "detail_csv"),
            ("pick_audit_csv", "pick_audit_csv"),
            ("waveform_root", "waveform_root"),
            ("station_file", "station_file"),
            ("evidence_csv", "evidence_csv"),
        ):
            if config_key in inputs:
                self.vars[var_key].set(str(inputs.get(config_key, "")))

        workface = config.get("workface", {})
        for config_key, var_key in (
            ("boundary_file", "boundary_file"),
            ("mapa_file", "mapa_file"),
            ("cad_x_offset", "cad_x_offset"),
            ("cad_y_offset", "cad_y_offset"),
        ):
            if config_key in workface:
                self.vars[var_key].set(str(workface.get(config_key, "")))
        basemap = str(workface.get("basemap_file", "")).strip()
        self.vars["basemap_file"].set(basemap)
        if basemap.lower().endswith(".dat"):
            self.vars["mapa_file"].set(basemap)
            self.vars["dwg_file"].set("")
        else:
            self.vars["dwg_file"].set(basemap)

        output_dir = config.get("outputs", {}).get("directory", "")
        if output_dir:
            self.vars["slice_dir"].set(str(output_dir))
            self.vars["verify_dir"].set(str(output_dir))
        project_model = config.get("model", {})
        self.project_model_extras = {
            key: value for key, value in project_model.items() if key not in self.vars
        }
        for key, value in project_model.items():
            if key in self.vars:
                self.vars[key].set("1" if value is True else "0" if value is False else str(value))

        csv_text = self.vars["output_csv"].get().strip()
        if csv_text:
            self._configure_picker_paths_for_csv(Path(csv_text))
        self.refresh_all()

    def load_project(self, project_path: Path, show_message: bool = True) -> bool:
        try:
            config = load_project_config(project_path)
            self._apply_project_config(config, project_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if show_message:
                messagebox.showerror("项目文件错误", str(exc))
            return False
        self.write_console(f">>> 已打开项目配置: {project_path}\n", color="success")
        return True

    def select_project_file(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=str(Path(self.vars["output_csv"].get() or ROOT).parent),
            filetypes=[("Wave CT项目", "*.json"), ("所有文件", "*.*")],
        )
        if selected:
            self.load_project(Path(selected))

    def import_raw_dataset(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showwarning("任务正在运行", "请等待当前任务完成后再导入原始数据集。")
            return
        selected = filedialog.askdirectory(
            title="选择待自动识别的原始数据集根目录",
            initialdir=str(ROOT.parent),
        )
        if not selected:
            return
        dataset_root = Path(selected)
        configured_root = str(self.app_settings.get("default_output_root", "")).strip()
        initial_output_parent = Path(configured_root) if configured_root else ROOT
        if not initial_output_parent.is_dir():
            initial_output_parent = ROOT
        selected_parent = filedialog.askdirectory(
            title="选择处理结果父目录（软件将自动新建数据集处理结果目录）",
            initialdir=str(initial_output_parent),
        )
        if not selected_parent:
            return
        output_root = Path(selected_parent) / f"{dataset_root.name}处理结果"
        self.write_console(
            f">>> 正在自动识别原始数据集: {dataset_root}\n"
            f">>> 标准项目将写入: {output_root}\n",
            color="cmd",
        )
        self._run_command(
            [
                sys.executable,
                "-m",
                "wave_ct.tools.import_dataset",
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output_root),
            ],
            "导入原始数据集并生成标准项目",
            on_success=lambda: self._load_imported_project(output_root),
        )

    def _load_imported_project(self, output_root: Path) -> None:
        projects = list(output_root.rglob("wave_ct_project.json"))
        if not projects:
            messagebox.showerror(
                "导入结果不完整",
                f"导入命令已结束，但没有找到 wave_ct_project.json：\n{output_root}",
            )
            return
        project = max(
            projects,
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )
        if not self.load_project(project):
            return
        self.write_console(
            f">>> 原始数据集导入完成，共生成 {len(projects)} 个标准项目；"
            f"当前已加载: {project}\n",
            color="success",
        )
        if len(projects) > 1:
            messagebox.showinfo(
                "导入完成",
                f"共生成 {len(projects)} 个独立时段项目，已自动加载最新项目：\n"
                f"{project}\n\n可在“打开项目”中切换其他时段。",
            )

    def save_current_project(self, silent: bool = False, target: Path | None = None) -> Optional[Path]:
        csv_text = self.vars["output_csv"].get().strip()
        project_path = target or self.project_file
        if project_path is None and csv_text:
            project_path = project_path_for_csv(Path(csv_text))
        if project_path is None:
            if silent:
                return None
            selected = filedialog.asksaveasfilename(
                initialdir=str(ROOT), initialfile="wave_ct_project.json",
                defaultextension=".json", filetypes=[("Wave CT项目", "*.json")],
            )
            if not selected:
                return None
            project_path = Path(selected)
        try:
            saved = save_project_config(project_path, self._current_project_config())
        except OSError as exc:
            if not silent:
                messagebox.showerror("项目保存失败", str(exc))
            return None
        self.project_file = saved
        self.vars["project_file"].set(str(saved))
        if not silent:
            self.write_console(f">>> 项目配置已保存: {saved}\n", color="success")
            messagebox.showinfo("项目已保存", str(saved))
        return saved

    def _save_runtime_project_snapshot(self) -> None:
        self.save_current_project(silent=True)
        output_text = self.vars["slice_dir"].get().strip()
        if not output_text:
            return
        try:
            output_dir = Path(output_text)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_project_config(
                output_dir / "wave_ct_project_snapshot.json",
                self._current_project_config(),
            )
        except OSError as exc:
            self.write_console(f">>> 警告: 无法写入项目快照: {exc}\n", color="error")

    def select_basemap_file(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=str(Path(self.vars["output_csv"].get() or ROOT).parent),
            filetypes=[
                ("矿井底图", "*.dwg *.dxf *.dat"),
                ("AutoCAD DWG", "*.dwg"),
                ("AutoCAD DXF", "*.dxf"),
                ("SOS Mapa", "*.dat"),
                ("所有文件", "*.*"),
            ],
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() == ".dat":
            self.vars["mapa_file"].set(selected)
            self.vars["dwg_file"].set("")
        else:
            self.vars["dwg_file"].set(selected)
        self.vars["basemap_file"].set(selected)
        self.save_current_project(silent=True)
        self.write_console(f">>> 已设置矿井底图: {selected}\n", color="success")

    def detect_current_project_assets(self) -> None:
        csv_text = self.vars["output_csv"].get().strip()
        if not csv_text or not Path(csv_text).is_file():
            messagebox.showwarning("缺少走时CSV", "请先选择当前项目的走时CSV。")
            return
        self._auto_detect_workface_assets(Path(csv_text), force=True)
        self.save_current_project(silent=True)


    # ================= 优化设置弹窗 =================
    def open_project_settings(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("项目与软件环境设置")
        win.geometry("900x720")
        win.minsize(850, 700)
        win.transient(self)
        win.configure(fg_color=COLORS["bg"])

        # 头部
        header = ctk.CTkFrame(win, height=64, corner_radius=0, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="⚙️ 项目全局配置", text_color=COLORS["text_main"], font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=24)

        tabview = ctk.CTkTabview(win, corner_radius=12, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        tabview.pack(fill="both", expand=True, padx=24, pady=24)
        
        tab_proj = tabview.add("当前项目参数")
        tab_validation = tabview.add("验证输入")
        tab_soft = tabview.add("软件与CAD环境")

        def path_row(parent, row, label, variable, command):
            lbl = ctk.CTkLabel(parent, text=label, text_color=COLORS["text_main"], font=ctk.CTkFont(size=13, weight="bold"))
            lbl.grid(row=row, column=0, sticky="w", pady=10, padx=(20, 15))
            entry = ctk.CTkEntry(parent, textvariable=variable, corner_radius=6, border_color=COLORS["border"], height=36, text_color=COLORS["primary"])
            entry.grid(row=row, column=1, sticky="ew", pady=10)
            btn = ctk.CTkButton(parent, text="📂 浏览", command=command, width=80, height=36, corner_radius=6, fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"])
            btn.grid(row=row, column=2, padx=(10, 20), pady=10)

        # ---------------- 项目面板 ----------------
        tab_proj.columnconfigure(1, weight=1)
        
        ctk.CTkLabel(tab_proj, text="项目名称", text_color=COLORS["text_main"], font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, sticky="w", pady=10, padx=(20, 15))
        ctk.CTkEntry(tab_proj, textvariable=self.vars["project_name"], corner_radius=6, height=36).grid(row=0, column=1, columnspan=2, sticky="ew", pady=10, padx=(0, 20))
        
        ctk.CTkLabel(tab_proj, text="报表模板", text_color=COLORS["text_main"], font=ctk.CTkFont(size=13, weight="bold")).grid(row=1, column=0, sticky="w", pady=10, padx=(20, 15))
        ctk.CTkComboBox(tab_proj, variable=self.vars["report_template"], values=["auto", "generic", "yanbei"], corner_radius=6, height=36).grid(row=1, column=1, columnspan=2, sticky="ew", pady=10, padx=(0, 20))

        path_row(tab_proj, 2, "走时 CSV", self.vars["output_csv"], self.select_input_csv)
        path_row(tab_proj, 3, "波形目录", self.vars["waveform_root"], self.select_waveform_root)
        path_row(tab_proj, 4, "台站坐标文件", self.vars["station_file"], 
                 lambda: self._select_file_for_var("station_file", [("坐标文件", "*.txt *.csv *.xlsx"), ("所有文件", "*.*")]))
        path_row(tab_proj, 5, "工作面边界 CSV", self.vars["boundary_file"], self.select_boundary_file)
        path_row(tab_proj, 6, "矿井底图", self.vars["basemap_file"], self.select_basemap_file)
        path_row(tab_proj, 7, "成果目录", self.vars["slice_dir"], self.select_output_dir)

        offset_frame = ctk.CTkFrame(tab_proj, fg_color="transparent")
        offset_frame.grid(row=8, column=1, columnspan=2, sticky="ew", pady=10)
        ctk.CTkLabel(tab_proj, text="CAD 坐标偏移", text_color=COLORS["text_main"], font=ctk.CTkFont(size=13, weight="bold")).grid(row=8, column=0, sticky="w", pady=10, padx=(20, 15))
        ctk.CTkLabel(offset_frame, text="X:", text_color=COLORS["text_muted"]).pack(side="left")
        ctk.CTkEntry(offset_frame, textvariable=self.vars["cad_x_offset"], width=90, height=36, corner_radius=6).pack(side="left", padx=(6, 24))
        ctk.CTkLabel(offset_frame, text="Y:", text_color=COLORS["text_muted"]).pack(side="left")
        ctk.CTkEntry(offset_frame, textvariable=self.vars["cad_y_offset"], width=90, height=36, corner_radius=6).pack(side="left", padx=6)

        # 底部操作栏
        btns = ctk.CTkFrame(tab_proj, fg_color="transparent")
        btns.grid(row=9, column=0, columnspan=3, pady=(20, 20), padx=20, sticky="ew")
        
        ctk.CTkButton(btns, text="打开历史项目", command=self.select_project_file, height=40, fg_color="transparent", border_width=1, border_color=COLORS["border"], text_color=COLORS["text_main"], corner_radius=8).pack(side="left")
        ctk.CTkButton(btns, text="自动识别资产", command=self.detect_current_project_assets, height=40, fg_color="transparent", border_width=1, border_color=COLORS["border"], text_color=COLORS["text_main"], corner_radius=8).pack(side="left", padx=12)
        ctk.CTkButton(btns, text="保存项目配置", command=self.save_current_project, font=ctk.CTkFont(weight="bold"), height=40, corner_radius=8, fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"]).pack(side="right")
        
        ctk.CTkLabel(tab_proj, textvariable=self.vars["project_file"], text_color=COLORS["text_muted"], font=ctk.CTkFont("Consolas", 11), wraplength=700).grid(row=10, column=0, columnspan=3, sticky="w", padx=20)

        # ---------------- 验证输入面板 ----------------
        tab_validation.columnconfigure(1, weight=1)
        path_row(
            tab_validation, 0, "走时详情 CSV", self.vars["detail_csv"],
            lambda: self._select_file_for_var("detail_csv", [("CSV文件", "*.csv"), ("所有文件", "*.*")])
        )
        path_row(
            tab_validation, 1, "拾取审计 CSV", self.vars["pick_audit_csv"],
            lambda: self._select_file_for_var("pick_audit_csv", [("CSV文件", "*.csv"), ("所有文件", "*.*")])
        )
        path_row(
            tab_validation, 2, "外部证据 CSV", self.vars["evidence_csv"],
            lambda: self._select_file_for_var("evidence_csv", [("CSV文件", "*.csv"), ("所有文件", "*.*")])
        )
        info_box = ctk.CTkFrame(tab_validation, corner_radius=8, fg_color=COLORS["bg"], border_width=1, border_color=COLORS["border"])
        info_box.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(20, 0), padx=20)
        ctk.CTkLabel(
            info_box,
            text=("💡 提示：详情 CSV 用于拾取质量统计；审计 CSV 保留导入/剔除依据；\n"
                  "证据 CSV 用于钻孔、矿压、瓦斯等独立验证。缺失项在报告中记为 SKIPPED。"),
            text_color=COLORS["text_muted"], justify="left", font=ctk.CTkFont(size=12)
        ).pack(padx=16, pady=16, anchor="w")

        # ---------------- 软件环境面板 ----------------
        tab_soft.columnconfigure(1, weight=1)
        soft_vars = {
            key: tk.StringVar(value=str(self.app_settings.get(key, "")))
            for key in ("autocad_core_console", "cad_cache_dir", "default_output_root")
        }
        
        path_row(tab_soft, 0, "AutoCAD 路径", soft_vars["autocad_core_console"], 
                 lambda: self._select_file_for_external_var(soft_vars["autocad_core_console"], [("accoreconsole", "accoreconsole.exe"), ("程序", "*.exe")]))
        path_row(tab_soft, 1, "CAD 缓存目录", soft_vars["cad_cache_dir"], 
                 lambda: self._select_dir_for_external_var(soft_vars["cad_cache_dir"]))
        path_row(tab_soft, 2, "默认成果目录", soft_vars["default_output_root"], 
                 lambda: self._select_dir_for_external_var(soft_vars["default_output_root"]))
        
        status_var = tk.StringVar(value="")
        ctk.CTkLabel(tab_soft, textvariable=status_var, text_color=COLORS["success"], font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, columnspan=3, sticky="w", padx=20, pady=(15, 5))
        
        btns2 = ctk.CTkFrame(tab_soft, fg_color="transparent")
        btns2.grid(row=4, column=0, columnspan=3, pady=10, padx=20, sticky="ew")

        def detect_autocad():
            detected = discover_accoreconsole()
            soft_vars["autocad_core_console"].set(detected)
            status_var.set("已发现 AutoCAD。" if detected else "未发现 AutoCAD；DXF 和无底图流程仍可使用。")

        def save_software():
            core = soft_vars["autocad_core_console"].get().strip()
            cache = soft_vars["cad_cache_dir"].get().strip()
            if core and not Path(core).is_file():
                messagebox.showerror("AutoCAD 路径错误", "accoreconsole.exe 不存在。")
                return
            try:
                cache.encode("ascii")
            except UnicodeEncodeError:
                messagebox.showerror("缓存路径错误", "CAD 缓存目录必须使用英文路径。")
                return
            try:
                if cache: Path(cache).mkdir(parents=True, exist_ok=True)
                self.app_settings = {
                    "autocad_core_console": core,
                    "cad_cache_dir": cache,
                    "default_output_root": soft_vars["default_output_root"].get().strip(),
                }
                saved = save_app_settings(self.app_settings)
                self._apply_app_settings_environment()
            except OSError as exc:
                messagebox.showerror("软件设置保存失败", str(exc))
                return
            status_var.set(f"设置已保存！")
            self.write_console(f">>> 软件设置已保存: {saved}\n", color="success")

        ctk.CTkButton(btns2, text="自动查找 AutoCAD", command=detect_autocad, height=40, fg_color="transparent", border_width=1, border_color=COLORS["border"], text_color=COLORS["text_main"], corner_radius=8).pack(side="left")
        ctk.CTkButton(btns2, text="保存环境设置", command=save_software, font=ctk.CTkFont(weight="bold"), height=40, corner_radius=8, fg_color=COLORS["primary"]).pack(side="right")
        
        ctk.CTkLabel(tab_soft, text=f"配置文件路径：{app_settings_path()}", text_color=COLORS["text_muted"], font=ctk.CTkFont("Consolas", 11)).grid(row=5, column=0, columnspan=3, sticky="w", padx=20, pady=(15, 0))

    def _select_file_for_var(self, key: str, filetypes) -> None:
        selected = filedialog.askopenfilename(initialdir=str(ROOT), filetypes=filetypes)
        if selected:
            self.vars[key].set(selected)

    @staticmethod
    def _select_file_for_external_var(variable: tk.StringVar, filetypes) -> None:
        selected = filedialog.askopenfilename(initialdir=str(ROOT), filetypes=filetypes)
        if selected:
            variable.set(selected)

    @staticmethod
    def _select_dir_for_external_var(variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or str(ROOT))
        if selected:
            variable.set(selected)

    def run_optional_boundary_report(self):
        boundary_text = self.vars["boundary_file"].get().strip()
        boundary_path = Path(boundary_text) if boundary_text else None
        if boundary_path is None or not boundary_path.is_file():
            self.select_boundary_file()
            boundary_text = self.vars["boundary_file"].get().strip()
            boundary_path = Path(boundary_text) if boundary_text else None
        if boundary_path is not None and boundary_path.is_file():
            self.run_workface_report()

    def select_waveform_root(self):
        selected = filedialog.askdirectory(initialdir=self.vars["waveform_root"].get())
        if selected:
            self.vars["waveform_root"].set(selected)
            self.save_current_project(silent=True)
            self.write_console(f">>> 已更换波形目录: {selected}\n", color="success")

    def select_input_csv(self):
        selected = filedialog.askopenfilename(
            initialdir=str((ROOT / "标波输出数据集").resolve()),
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if selected:
            selected_path = Path(selected)
            project_path = project_path_for_csv(selected_path)
            if project_path.is_file() and self.load_project(project_path, show_message=False):
                self.vars["output_csv"].set(selected)
            else:
                self.dataset_config = {}
                self.project_model_extras = {}
                self.project_file = project_path
                self.vars["project_file"].set(str(project_path))
                self.vars["project_name"].set(selected_path.parent.name or selected_path.stem)
                self.vars["output_csv"].set(selected)
                self.vars["detail_csv"].set(
                    str(selected_path.with_name(f"{selected_path.stem}_detail.csv"))
                )
                audit_candidate = selected_path.parent / "pick_audit.csv"
                evidence_candidate = selected_path.parent / "evidence.csv"
                station_candidate = selected_path.parent / "stations_for_picker.txt"
                self.vars["pick_audit_csv"].set(
                    str(audit_candidate) if audit_candidate.is_file() else ""
                )
                self.vars["evidence_csv"].set(
                    str(evidence_candidate) if evidence_candidate.is_file() else ""
                )
                self.vars["waveform_root"].set("")
                self.vars["station_file"].set(
                    str(station_candidate) if station_candidate.is_file() else ""
                )
                self.vars["boundary_file"].set("")
                self.vars["mapa_file"].set("")
                self.vars["dwg_file"].set("")
                self.vars["basemap_file"].set("")
                default_root = str(self.app_settings.get("default_output_root", "")).strip()
                output_dir = Path(default_root) / self.vars["project_name"].get() if default_root else selected_path.parent / "inversion"
                self.vars["slice_dir"].set(str(output_dir))
                self.vars["verify_dir"].set(str(output_dir))
                self._configure_picker_paths_for_csv(selected_path)
                self.save_current_project(silent=True)
            self.write_console(f">>> 已导入人工走时CSV: {selected}\n", color="success")
            self.refresh_all()

    def _configure_picker_paths_for_csv(self, csv_path: Path) -> None:
        dataset2 = ROOT / "新数据集处理结果"
        is_dataset2 = any("新数据集" in part for part in csv_path.parts)
        try:
            csv_path.resolve().relative_to(dataset2.resolve())
        except ValueError:
            pass
        else:
            is_dataset2 = True
        if is_dataset2:
            waveform_root = dataset2 / "observed_waveforms"
            station_file = dataset2 / "metadata" / "stations_for_picker.txt"
            if waveform_root.is_dir() and station_file.is_file():
                self.vars["waveform_root"].set(str(waveform_root))
                self.vars["station_file"].set(str(station_file))
                self.write_console(
                    ">>> 已自动关联新数据集的原始波形和台站坐标，可在人工标注界面复核CSV红线。\n",
                    color="success",
                )

        summary = self._summary_metadata(csv_path)
        inversion_dir = csv_path.parent / "inversion"
        if (inversion_dir / "velocity_model.npz").is_file():
            self.vars["slice_dir"].set(str(inversion_dir))
            self.vars["verify_dir"].set(str(inversion_dir))
        slices = summary.get("slice_targets_m")
        if isinstance(slices, list) and slices:
            self.vars["slice_z"].set(",".join(str(value) for value in slices))
        background = summary.get("background_velocity_m_s")
        if isinstance(background, (int, float)) and background > 0:
            self.vars["background_velocity"].set(f"{background:g}")
        for summary_key, var_key in (
            ("cad_x_offset", "cad_x_offset"),
            ("cad_y_offset", "cad_y_offset"),
        ):
            value = summary.get(summary_key)
            if isinstance(value, (int, float)):
                self.vars[var_key].set(f"{value:g}")
        if summary.get("report_template") in {"auto", "generic", "yanbei"}:
            self.vars["report_template"].set(str(summary["report_template"]))
        self._auto_detect_workface_assets(csv_path)

    def _summary_metadata(self, csv_path: Path) -> dict:
        for candidate in (csv_path.parent / "summary.json", csv_path.with_name("summary.json")):
            if not candidate.is_file():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
        return {}

    def _auto_detect_workface_assets(
        self, csv_path: Path, *, force: bool = False,
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        if str(self.dataset_config.get("data_type", "")).strip().lower() == "synthetic":
            self.vars["boundary_file"].set("")
            self.vars["basemap_file"].set("")
            self.vars["mapa_file"].set("")
            self.vars["dwg_file"].set("")
            return None, None, None
        configured = any(
            self.vars[key].get().strip()
            for key in ("boundary_file", "basemap_file", "mapa_file", "dwg_file")
        )
        if not configured and not force:
            return None, None, None
        boundary_text = self.vars["boundary_file"].get().strip()
        boundary = Path(boundary_text) if boundary_text else None
        if boundary is None or not boundary.is_file():
            candidates = []
            for parent in (csv_path.parent, *list(csv_path.parents)[:5]):
                candidates.extend((parent / "workface_boundary.csv", parent / "工作面边界.csv"))
            if "砚北" in str(csv_path):
                candidates.append(ROOT / "砚北处理结果" / "砚北工作面边界.csv")
            boundary = next((path for path in candidates if path.is_file()), None)
            if boundary is not None:
                self.vars["boundary_file"].set(str(boundary))

        if boundary is not None and boundary.is_file():
            model_bounds = coordinate_bounds_from_csv(csv_path)
            face_bounds = boundary_bounds_from_csv(boundary)
            if model_bounds is not None and face_bounds is not None:
                mx0, mx1, my0, my1 = model_bounds[:4]
                fx0, fx1, fy0, fy1 = face_bounds
                overlap_x = max(0.0, min(mx1, fx1) - max(mx0, fx0))
                overlap_y = max(0.0, min(my1, fy1) - max(my0, fy0))
                if overlap_x <= 0.0 or overlap_y <= 0.0:
                    self.write_console(
                        ">>> 工作面底图与当前数据坐标范围不重叠，已自动改用无底图通用切片。\n",
                        color="error",
                    )
                    for key in ("boundary_file", "basemap_file", "mapa_file", "dwg_file"):
                        self.vars[key].set("")
                    return None, None, None

        mapa_text = self.vars["mapa_file"].get().strip()
        mapa = Path(mapa_text) if mapa_text else None
        if mapa is None or not mapa.is_file():
            candidates = []
            for parent in (csv_path.parent, *list(csv_path.parents)[:5]):
                candidates.extend((parent / "Mapa.dat", parent / "SOS" / "Mapa.dat"))
            summary = self._summary_metadata(csv_path)
            period = str(summary.get("period") or csv_path.parent.name)
            data_root = ROOT.parent / "砚北煤矿数据集" / "砚北煤矿完整数据"
            if data_root.is_dir() and len(period) >= 7:
                period_prefixes = {period[:7], period[:7].replace("-", ".")}
                for archive in data_root.iterdir():
                    if archive.is_dir() and any(archive.name.startswith(prefix) for prefix in period_prefixes):
                        candidates.append(archive / "SOS" / "Mapa.dat")
            mapa = next((path for path in candidates if path.is_file()), None)
            if mapa is not None:
                self.vars["mapa_file"].set(str(mapa))

        dwg_text = self.vars["dwg_file"].get().strip()
        dwg = Path(dwg_text) if dwg_text else None
        if dwg is None or not dwg.is_file():
            candidates = []
            for parent in (csv_path.parent, *list(csv_path.parents)[:5]):
                candidates.extend((
                    parent / "mine_map.dwg",
                    parent / "mine_map.dxf",
                    parent / "矿井底图.dwg",
                    parent / "矿井底图.dxf",
                    parent / "工作面底图.dwg",
                    parent / "工作面底图.dxf",
                    parent / "\u5fae\u9707\u5b9a\u4f4d\u5e95\u56fe.dwg",
                    parent / "\u5fae\u9707\u5b9a\u4f4d\u5e95\u56fe.dxf",
                ))
            summary = self._summary_metadata(csv_path)
            period = str(summary.get("period") or csv_path.parent.name)
            data_root = ROOT.parent / "\u781a\u5317\u7164\u77ff\u6570\u636e\u96c6" / "\u781a\u5317\u7164\u77ff\u5b8c\u6574\u6570\u636e"
            if data_root.is_dir() and len(period) >= 7:
                period_prefixes = {period[:7], period[:7].replace("-", ".")}
                for archive in data_root.iterdir():
                    if archive.is_dir() and any(archive.name.startswith(prefix) for prefix in period_prefixes):
                        candidates.append(
                            archive / "SOS" / "\u5fae\u9707\u5b9a\u4f4d\u5e95\u56fe.dwg"
                        )
            dwg = next((path for path in candidates if path.is_file()), None)
            if dwg is not None:
                self.vars["dwg_file"].set(str(dwg))

        if dwg is not None and dwg.is_file():
            self.vars["basemap_file"].set(str(dwg))
        elif mapa is not None and mapa.is_file():
            self.vars["basemap_file"].set(str(mapa))

        if boundary is not None and boundary.is_file():
            map_note = f"，矿图: {mapa.name}" if mapa is not None and mapa.is_file() else ""
            if dwg is not None and dwg.is_file():
                map_note = f", CAD: {dwg.name}"
            self.write_console(f">>> 已识别工作面边界: {boundary.name}{map_note}\n", color="success")
        return boundary, mapa, dwg

    def select_output_dir(self):
        selected = filedialog.askdirectory(initialdir=self.vars["slice_dir"].get())
        if selected:
            self.vars["slice_dir"].set(selected)
            self.vars["verify_dir"].set(selected)
            self.save_current_project(silent=True)
            self.write_console(f">>> 成果输出目录已设置为: {selected}\n", color="success")
            self.refresh_all()

    def select_boundary_file(self):
        selected = filedialog.askopenfilename(
            initialdir=str(ROOT),
            filetypes=[("工作面边界CSV", "*.csv"), ("所有文件", "*.*")],
        )
        if selected:
            self.vars["boundary_file"].set(selected)
            self.save_current_project(silent=True)
            self.write_console(f">>> 已导入工作面边界: {selected}\n", color="success")

    def _selected_report_template(self, input_csv: Path) -> str:
        selected = self.vars["report_template"].get().strip().lower()
        if selected in {"generic", "yanbei"}:
            return selected
        summary_value = str(self._summary_metadata(input_csv).get("report_template", "")).strip().lower()
        return summary_value if summary_value in {"generic", "yanbei"} else "generic"

    def _find_reference_dataset_root(self) -> Optional[Path]:
        candidates: List[Path] = []
        raw_root = str(self.dataset_config.get("raw_root", "")).strip()
        if raw_root:
            candidates.append(Path(raw_root))
        for value in (self.vars["waveform_root"].get(), self.vars["output_csv"].get()):
            text = value.strip()
            if not text:
                continue
            path = Path(text)
            start = path if path.is_dir() else path.parent
            candidates.extend([start, *list(start.parents)[:4]])

        seen = set()
        for candidate in candidates:
            key = str(candidate.resolve()).lower()
            if key in seen or not candidate.is_dir():
                continue
            seen.add(key)
            try:
                from wave_ct.reference_sirt import _load_reference_slices
                if _load_reference_slices(candidate):
                    return candidate
            except (OSError, ValueError):
                continue
        return None

    def _run_optional_reference_comparison(self, on_success=None) -> None:
        dataset_root = self._find_reference_dataset_root()
        if dataset_root is None:
            if on_success is not None:
                on_success()
            return

        input_csv = Path(self.vars["output_csv"].get())
        output_dir = Path(self.vars["slice_dir"].get())
        validation_ids = ""
        report_path = output_dir / "slice_report.txt"
        if report_path.is_file():
            try:
                for line in report_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("validation_source_ids="):
                        validation_ids = line.partition("=")[2].strip()
                        break
            except OSError:
                validation_ids = ""

        args = [
            sys.executable, "-m", "wave_ct.tools.reference_compare",
            "--dataset-root", str(dataset_root),
            "--result-dir", str(output_dir),
            "--output-dir", str(output_dir / "参考结果对比"),
            "--input-csv", str(input_csv),
        ]
        if validation_ids:
            args.extend(["--validation-source-ids", validation_ids])
        self.write_console(
            f">>> 检测到外部参考速度TXT: {dataset_root}\n"
            ">>> 参考值只用于反演后对比，不参与模型求解。\n",
            color="success",
        )
        self._run_command(args, "生成外部参考逐点对比", on_success=on_success)

    def generate_all_results(self):
        if not self.validate_manual_input_ready():
            return
        input_csv = Path(self.vars["output_csv"].get())
        output_dir = Path(self.vars["slice_dir"].get())
        if not (output_dir / "velocity_model.npz").is_file():
            messagebox.showwarning("缺少反演结果", "请先完成第3步反演，再一键生成成果。")
            return
        self._save_runtime_project_snapshot()
        boundary, _, cad_file = self._auto_detect_workface_assets(input_csv)
        if boundary is not None and boundary.is_file():
            report_callback = (
                self._generate_workface_docx
                if self._selected_report_template(input_csv) == "yanbei"
                else self._run_validation_for_report
            )
            self.run_workface_report(
                on_success=lambda: self._run_optional_reference_comparison(report_callback),
                prompt_for_boundary=False,
            )
        else:
            if cad_file is not None and cad_file.is_file():
                self.write_console(
                    ">>> 已导入矿井底图，但未提供工作面边界CSV；本次生成通用CT成果，不伪造工作面范围。\n",
                    color="error",
                )
            self._run_optional_reference_comparison(self._run_validation_for_report)

    def run_workface_report(self, on_success=None, prompt_for_boundary: bool = True) -> bool:
        input_csv = Path(self.vars["output_csv"].get())
        output_dir = Path(self.vars["slice_dir"].get())
        model_path = output_dir / "velocity_model.npz"
        boundary_auto, mapa_path, cad_path = self._auto_detect_workface_assets(input_csv)
        boundary_text = self.vars["boundary_file"].get().strip()
        boundary_path = Path(boundary_text) if boundary_text else boundary_auto
        if not input_csv.is_file() or not model_path.is_file():
            messagebox.showwarning("缺少反演结果", "请先导入走时CSV并完成三维反演。")
            return False
        if boundary_path is None or not boundary_path.is_file():
            if prompt_for_boundary:
                self.select_boundary_file()
                boundary_text = self.vars["boundary_file"].get().strip()
                boundary_path = Path(boundary_text) if boundary_text else None
            if boundary_path is None or not boundary_path.is_file():
                messagebox.showwarning("缺少工作面边界", "未识别到包含 name,x,y 三列的工作面边界CSV。")
                return False
        try:
            with boundary_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            xs = [float(row["x"]) for row in rows]
            ys = [float(row["y"]) for row in rows]
            if len(xs) < 3:
                raise ValueError("边界顶点少于3个")
            view_padding = float(self.vars["workface_view_padding"].get())
            if not 0.0 <= view_padding <= 0.25:
                raise ValueError("工作面图边缘留白比例必须在 0 到 0.25 之间")
            pad_x = (max(xs) - min(xs)) * view_padding
            pad_y = (max(ys) - min(ys)) * view_padding
            vmin = float(self.vars["vmin_model"].get())
            vmax = float(self.vars["vmax_model"].get())
            if vmin <= 0 or vmax <= vmin:
                vmin = vmax = 0.0
        except (KeyError, TypeError, ValueError, OSError) as exc:
            messagebox.showerror("工作面参数错误", str(exc))
            return False

        summary = self._summary_metadata(input_csv)
        requested_start = str(summary.get("requested_start", "")).split(" ")[0]
        requested_end = str(summary.get("requested_end", "")).split(" ")[0]
        period = "-".join(value for value in (requested_start, requested_end) if value)
        if not period:
            period = str(summary.get("period") or input_csv.parent.name)
        try:
            background = float(self.vars["background_velocity"].get())
        except ValueError:
            background = 0.0

        args = [
            sys.executable, "-m", "wave_ct.workface_plot",
            str(model_path), str(input_csv), str(output_dir),
            "--boundary-file", str(boundary_path),
            f"--slice-z={self.vars['slice_z'].get()}",
            "--vmin", f"{vmin:g}", "--vmax", f"{vmax:g}",
            "--background-velocity", f"{background:g}",
            "--presentation-vmin", self.vars["presentation_vmin"].get(),
            "--presentation-vmax", self.vars["presentation_vmax"].get(),
            "--presentation-sigma", self.vars["presentation_sigma"].get(),
            "--anomaly-limit", self.vars["anomaly_limit"].get(),
            "--period", period,
            "--x-min", f"{min(xs) - pad_x:.12g}", "--x-max", f"{max(xs) + pad_x:.12g}",
            "--y-min", f"{min(ys) - pad_y:.12g}", "--y-max", f"{max(ys) + pad_y:.12g}",
        ]
        self.write_console(
            "工作面成果图显示范围："
            f"边界留白={view_padding:.3f}，"
            f"X=[{min(xs) - pad_x:.2f}, {max(xs) + pad_x:.2f}]，"
            f"Y=[{min(ys) - pad_y:.2f}, {max(ys) + pad_y:.2f}]\n"
        )
        if mapa_path is not None and mapa_path.is_file():
            args.extend(["--mapa-file", str(mapa_path)])
        if cad_path is not None and cad_path.is_file():
            try:
                cad_x_offset = float(self.vars["cad_x_offset"].get())
                cad_y_offset = float(self.vars["cad_y_offset"].get())
            except ValueError:
                messagebox.showerror("CAD坐标偏移错误", "CAD X/Y坐标偏移必须是数字。")
                return False
            args.extend([
                "--cad-file", str(cad_path),
                "--cad-x-offset", f"{cad_x_offset:.12g}",
                "--cad-y-offset", f"{cad_y_offset:.12g}",
            ])
            core_console = str(self.app_settings.get("autocad_core_console", "")).strip()
            cache_dir = str(self.app_settings.get("cad_cache_dir", "")).strip()
            if core_console:
                args.extend(["--accoreconsole", core_console])
            if cache_dir:
                args.extend(["--cad-cache-dir", cache_dir])
        self._run_command(args, "生成工作面全部成果图", on_success=on_success or self.refresh_result_images)
        return True

    def _run_validation_for_report(self):
        self.run_verify(on_success=self._generate_generic_report)

    def _generate_generic_report(self):
        input_csv = Path(self.vars["output_csv"].get())
        output_dir = Path(self.vars["slice_dir"].get())
        report_path = output_dir / f"{input_csv.stem}_CT反演成果报表.docx"
        dataset_name = self.vars["project_name"].get().strip() or input_csv.parent.name or input_csv.stem
        args = [
            sys.executable, "-m", "wave_ct.report",
            "--dataset-name", dataset_name,
            "--input-csv", str(input_csv),
            "--inversion-dir", str(output_dir),
            "--validation-dir", str(output_dir),
            "--output-docx", str(report_path),
            "--picking-source", "输入CSV中的观测P波到时",
        ]
        self._run_command(args, "生成CT反演成果报表", on_success=self.refresh_result_images)

    def _generate_workface_docx(self):
        input_csv = Path(self.vars["output_csv"].get())
        output_dir = Path(self.vars["slice_dir"].get())
        summary_path = input_csv.parent / "summary.json"
        slice_report = output_dir / "slice_report.txt"
        if not summary_path.is_file() or not slice_report.is_file():
            self._run_validation_for_report()
            return
        summary = self._summary_metadata(input_csv)
        requested_start = str(summary.get("requested_start", "")).split(" ")[0]
        requested_end = str(summary.get("requested_end", "")).split(" ")[0]
        period_label = "-".join(value for value in (requested_start, requested_end) if value)
        if not period_label:
            period_label = str(summary.get("period") or input_csv.parent.name)
        report_path = output_dir / f"砚北煤矿{summary.get('period', input_csv.parent.name)}震动波CT反演报表.docx"
        args = [
            sys.executable, "-m", "wave_ct.workface_report",
            str(summary_path), str(slice_report), str(output_dir), str(report_path),
            "--period", period_label,
            f"--slice-z={self.vars['slice_z'].get()}",
        ]
        self._run_command(args, "生成工作面CT反演报表", on_success=self.refresh_result_images)

    def update_bounds_from_csv(self, show_message: bool = False) -> bool:
        csv_path = Path(self.vars["output_csv"].get())
        geometry = acquisition_geometry_from_csv(csv_path)
        bounds = coordinate_bounds_from_csv(csv_path)
        if bounds is None or geometry is None:
            if show_message:
                messagebox.showwarning("无法自动设置", "当前CSV不存在，或没有可读取的震源/台站坐标。")
            return False

        boundary_text = self.vars["boundary_file"].get().strip()
        boundary_bounds = boundary_bounds_from_csv(Path(boundary_text)) if boundary_text else None
        workface_mode = boundary_bounds is not None and self.vars["allow_outside_rays"].get() == "1"

        # Both the deterministic GUI SIRT path and the optional automatic
        # selector use the same fine acquisition-centred grid.  The previous
        # auto branch fell back to ``recommend_workface_grid`` (about 2,000
        # cells), which produced broad, low-resolution anomalies.  Keep the
        # 10 m / 20 m-padding contract used by the reference probe.
        exact_sirt_grid = (
            self.vars["deep_reparameterization"].get() != "1"
            and (
                self.vars["solver_method"].get() == "sirt"
                or self.vars["auto_algorithm"].get() == "1"
            )
        )
        if exact_sirt_grid:
            raw_axes = (
                geometry["source_x"] + geometry["station_x"],
                geometry["source_y"] + geometry["station_y"],
                geometry["source_z"] + geometry["station_z"],
            )
            rounded = []
            exact_nodes = []
            for values in raw_axes:
                lower = float(min(values)) - 20.0
                requested_upper = float(max(values)) + 20.0
                cells = max(1, int(np.ceil((requested_upper - lower) / 10.0)))
                rounded.extend((lower, lower + 10.0 * cells))
                exact_nodes.append(cells + 1)
            for key, value in zip(
                ("nx_nodes", "ny_nodes", "nz_nodes"), exact_nodes
            ):
                self.vars[key].set(str(value))
        elif workface_mode:
            x_min, x_max, y_min, y_max = boundary_bounds
            targets = []
            try:
                targets = sorted(
                    float(value.strip())
                    for value in self.vars["slice_z"].get().split(",")
                    if value.strip()
                )
            except ValueError:
                targets = []
            workface_bounds, workface_nodes = recommend_workface_grid(
                geometry["row_count"], boundary_bounds, targets
            )
            for key, value in zip(
                ("nx_nodes", "ny_nodes", "nz_nodes"), workface_nodes
            ):
                self.vars[key].set(str(value))
            rounded = list(workface_bounds)
        else:
            try:
                features = extract_dataset_features(csv_path)
                rounded = list(features["inferred_bounds"])
                for key, value in zip(
                    ("nx_nodes", "ny_nodes", "nz_nodes"),
                    features["recommended_grid_nodes"],
                ):
                    self.vars[key].set(str(value))
            except (OSError, KeyError, TypeError, ValueError):
                rounded = list(bounds)

        for key, value in zip(
            ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"), rounded
        ):
            self.vars[key].set(f"{value:.12g}")

        for axis, nodes_key, min_key, max_key, spacing_key in (
            ("X", "nx_nodes", "x_min", "x_max", "dx"),
            ("Y", "ny_nodes", "y_min", "y_max", "dy"),
            ("Z", "nz_nodes", "z_min", "z_max", "dz"),
        ):
            nodes = max(2, int(float(self.vars[nodes_key].get())))
            spacing = (float(self.vars[max_key].get()) - float(self.vars[min_key].get())) / (nodes - 1)
            self.vars[spacing_key].set(f"{spacing:.6g}")

        self.write_console(
            ">>> 已自动设置模型范围与稳定网格: "
            f"X {self.vars['x_min'].get()}-{self.vars['x_max'].get()}, "
            f"Y {self.vars['y_min'].get()}-{self.vars['y_max'].get()}, "
            f"Z {self.vars['z_min'].get()}-{self.vars['z_max'].get()} m; "
            f"节点 {self.vars['nx_nodes'].get()}×{self.vars['ny_nodes'].get()}×{self.vars['nz_nodes'].get()}\n",
            color="success",
        )
        self.refresh_summary()
        return True

    def open_manual_picker(self):
        args = [
            sys.executable,
            "-m", "wave_ct.picker",
            "--waveform-root", self.vars["waveform_root"].get(),
            "--station-file", self.vars["station_file"].get(),
            "--output-csv", self.vars["output_csv"].get(),
            "--import-picks-csv", self.vars["output_csv"].get(),
            "--source-pattern", self.vars["source_pattern"].get(),
            "--waveform-pattern", self.vars["waveform_pattern"].get(),
            "--source-coord-filename", self.vars["source_coord_filename"].get(),
            "--read-only",
        ]
        self._run_command(args, "波形标注情况查看器", on_success=self.refresh_all)

    def validate_manual_input_ready(self) -> bool:
        csv_path = Path(self.vars["output_csv"].get())
        if not csv_path.exists():
            messagebox.showwarning(
                "缺少走时数据",
                "当前反演输入CSV不存在。\n\n请在“项目配置与设置”中选择已有走时CSV；"
                "如需复核原始波形，请点击“查看波形标注情况”。"
            )
            return False
        row_count = count_csv_rows(csv_path)
        if row_count <= 0:
            messagebox.showwarning(
                "走时数据为空",
                f"当前CSV没有有效记录：\n{csv_path}\n\n请在“项目配置与设置”中选择有效走时CSV。"
            )
            return False
        return True

    def _run_command(self, args, title, on_success=None):
        if self.proc and self.proc.poll() is None:
            messagebox.showwarning("警告", "当前有任务正在运行，请等待完成后再试！")
            return

        self._task_sequence += 1
        task_id = self._task_sequence
        self.write_console(f"\n========================================\n", color="cmd")
        self.write_console(f">>> 启动任务: {title}\n", color="cmd")
        self.write_console(f"========================================\n", color="cmd")
        self.set_status(f"正在运行任务: {title}...", state="running")

        def worker():
            try:
                self.proc = subprocess.Popen(args, cwd=path_text(ROOT), stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                for line in self.proc.stdout:
                    if "Error" in line or "Exception" in line or "Traceback" in line:
                        self.log_queue.put((line, "error"))
                    else:
                        self.log_queue.put((line, "info"))

                code = self.proc.wait()
                self.proc = None
                if code == 0:
                    self.log_queue.put((f"\n>>> ✅ 任务 [{title}] 成功完成! (代码: {code})\n", "success"))
                    def complete_success():
                        if on_success is not None:
                            on_success()
                        # A chained callback may start another command.  Only
                        # the latest idle task may mark the application ready.
                        if task_id == self._task_sequence and self.proc is None:
                            self.set_status("就绪", state="idle")
                    self.after(0, complete_success)
                else:
                    self.log_queue.put((f"\n>>> ❌ 任务 [{title}] 发生错误! (代码: {code})\n", "error"))
                    self.after(0, lambda: self.set_status(f"任务失败：{title}", state="error"))
            except Exception as e:
                self.log_queue.put((f"\n系统调用异常: {e}\n", "error"))
                self.proc = None
                self.after(0, lambda: self.set_status(f"启动失败：{title}", state="error"))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            msg, color = self.log_queue.get()
            self.write_console(msg, color)
        self.after(100, self._poll_log_queue)

    def run_one_click_inversion(self):
        """Run the deterministic probe-compatible inversion end to end.

        The callback after inversion already chains workface rendering,
        validation and report generation.  This entry point makes that path
        explicit and prevents an old project setting (auto pilot/deep model)
        from silently selecting the coarse-grid research workflow.
        """
        if not self.validate_manual_input_ready():
            return
        # Production processing is deterministic: it does not silently switch
        # to a research candidate selected on the same data.
        self.vars["auto_algorithm"].set("0")
        self.vars["deep_reparameterization"].set("0")
        self.vars["solver_method"].set("sirt")
        # Keep the explicitly verified 728 probe baseline.  This is a
        # reproducible presentation/inversion profile, not a claim that it is
        # independently optimal for every acquisition cohort.
        self.vars["reference_sirt_profile"].set("probe_728")
        self.vars["auto_bounds"].set("1")
        # Older saved projects retained the former 1.10-cell blur.  That
        # merges neighbouring SIRT cells into visually oversized red blocks
        # and no longer matches the production SIRT presentation contract.
        # Preserve deliberately smaller user values, but return stale broad
        # settings to the 0.65-cell default for a one-click production run.
        try:
            saved_sigma = float(self.vars["presentation_sigma"].get())
        except (TypeError, ValueError):
            saved_sigma = 0.65
        if saved_sigma > 0.75:
            self.vars["presentation_sigma"].set("0.65")
            self.write_console(
                ">>> 已将旧项目的展示平滑强度从过大的设置恢复为 0.65 单元；"
                "仅影响成果显示，不改变定量速度模型。\n",
                color="info",
            )
        # The two supplied 728 reference reports use different engineering
        # horizons.  Supply a cohort default only when the user left the
        # advanced slice field empty.  A one-click button must not silently
        # overwrite an explicit -750,-780,-810 (or any other) user choice.
        # ``output_csv`` is the GUI's authoritative travel-time input field.
        # Do not use the project-JSON key name (``travel_time_csv``) or the
        # non-existent legacy ``input_csv`` variable here: this callback is
        # invoked before a command line is constructed and must never crash.
        requested_slices = self.vars["slice_z"].get().strip()
        if not requested_slices:
            input_name = self.vars["output_csv"].get().replace("/", "\\")
            if "20260601-20260607" in input_name:
                requested_slices = "-740,-770,-800"
            elif "20260609-20260616" in input_name:
                requested_slices = "-750,-780,-810"
            if requested_slices:
                self.vars["slice_z"].set(requested_slices)
                self.write_console(
                    f">>> 未设置切片标高，已采用数据集默认层位：{requested_slices} m。\n",
                    color="info",
                )
        else:
            self.write_console(
                f">>> 保留高级求解参数中的切片标高：{requested_slices} m。\n",
                color="success",
            )
        self.write_console(
            ">>> 一键处理：兼容 SIRT 自动调参 + 成果图、验证和报告。"
            "参考结果若存在，仅作为复现实验的候选评分，不代表独立验证。\n",
            color="cmd",
        )
        self.run_inversion()

    def run_inversion(self):
        if not self.validate_manual_input_ready():
            return
        requested_slices = self.vars["slice_z"].get().strip()
        if requested_slices:
            self.write_console(
                f">>> 本次反演/成果绘图请求切片标高：{requested_slices} m。\n",
                color="info",
            )
        else:
            self.write_console(
                ">>> 切片标高为空：将输出模型全部网格层，非默认三层。\n",
                color="info",
            )
        auto_algorithm = self.vars["auto_algorithm"].get() == "1"
        deep_algorithm = self.vars["deep_reparameterization"].get() == "1"
        if auto_algorithm:
            deep_algorithm = False
        # The production GUI entry point is the source-compatible SIRT
        # implementation adapted from CT_shi_SIRT_Automatic_tuning_global_optimum.py.
        # Older project files may still contain solver_method=lsqr; do not let
        # that stale value silently bypass the requested production backend.
        use_reference_sirt = not auto_algorithm and not deep_algorithm
        if use_reference_sirt:
            self.vars["solver_method"].set("sirt")
            selected_profile = self.vars["reference_sirt_profile"].get().strip() or "probe_728"
            self.vars["reference_sirt_profile"].set(selected_profile)
            # Fixed probe runs must remain fixed.  AUTO/DE still use the
            # data-adaptive search when the user explicitly selects them.
            self.vars["sirt_auto_tune"].set("0" if selected_profile == "probe_728" else "1")
            if selected_profile == "probe_728":
                self.write_console(
                    ">>> 使用固定 728 探针 SIRT 档案：用于复现已确认的工作面异常展示风格。"
                    "该档案不替代后续事件级验证。\n",
                    color="info",
                )
            reference_root = self._find_reference_dataset_root()
            if reference_root is not None:
                self.write_console(
                    f">>> 检测到参考 Surfer TXT 网格，将用于候选参数的次级评分：{reference_root}\n"
                    ">>> 参考网格不进入反演方程；事件级留出预测仍是参数选择的安全门槛。\n",
                    color="info",
                )
            else:
                self.write_console(
                    ">>> 未检测到外部参考网格：仅按事件级留出走时误差调参。\n",
                    color="info",
                )
        if (
            (
                auto_algorithm
                or deep_algorithm
                or use_reference_sirt
            )
            and self.vars["auto_bounds"].get() == "1"
        ):
            if not self.update_bounds_from_csv(show_message=True):
                return
        if (
            not auto_algorithm
            and not deep_algorithm
            and
            self.vars["hierarchical_parameterization"].get() == "1"
            and self.vars["joint_sparsity"].get() == "1"
        ):
            messagebox.showerror(
                "算法选项冲突",
                "覆盖自适应分层网格与小波联合稀疏不能同时启用。\n"
                "请保留 TV，并关闭“小波+TV联合稀疏”。",
            )
            return
        self._save_runtime_project_snapshot()
        args = [
            sys.executable, "-m", "wave_ct.inversion",
            "--input-csv", self.vars["output_csv"].get(),
            "--output-dir", self.vars["slice_dir"].get(),
            "--mode", self.vars["mode"].get(),
            "--expected-sources", self.vars["expected_sources"].get(),
            "--expected-stations-per-source", self.vars["expected_stations_per_source"].get(),
            "--plot-style", self.vars["plot_style"].get(),
            f"--slice-z={self.vars['slice_z'].get()}",
            "--x-min", self.vars["x_min"].get(), "--x-max", self.vars["x_max"].get(),
            "--y-min", self.vars["y_min"].get(), "--y-max", self.vars["y_max"].get(),
            "--z-min", self.vars["z_min"].get(), "--z-max", self.vars["z_max"].get(),
            "--dx", self.vars["dx"].get(), "--dy", self.vars["dy"].get(), "--dz", self.vars["dz"].get(),
            "--nx-nodes", self.vars["nx_nodes"].get(),
            "--ny-nodes", self.vars["ny_nodes"].get(),
            "--nz-nodes", self.vars["nz_nodes"].get(),
            "--n-outer", self.vars["n_outer"].get(), "--n-lsqr", self.vars["n_lsqr"].get(),
            "--solver-method",
            "sirt" if use_reference_sirt else self.vars["solver_method"].get(),
            "--sirt-iterations", self.vars["sirt_iterations"].get(),
            "--sirt-omega", self.vars["sirt_omega"].get(),
            "--sirt-step-damp", self.vars["sirt_step_damp"].get(),
            "--sirt-tolerance", self.vars["sirt_tolerance"].get(),
            "--sirt-tune-maxiter", self.vars["sirt_tune_maxiter"].get(),
            "--sirt-tune-popsize", self.vars["sirt_tune_popsize"].get(),
            "--sirt-tune-iterations", self.vars["sirt_tune_iterations"].get(),
            "--reference-profile", self.vars["reference_sirt_profile"].get(),
            "--min-rays", self.vars["min_rays"].get(),
            "--alpha-reg", self.vars["alpha_reg"].get(), "--step-damp", self.vars["step_damp"].get(),
            "--vmin-qc", self.vars["vmin_qc"].get(), "--vmax-qc", self.vars["vmax_qc"].get(),
            "--vmin-model", self.vars["vmin_model"].get(), "--vmax-model", self.vars["vmax_model"].get(),
            "--background-velocity", self.vars["background_velocity"].get(),
            "--min-ray-coverage", self.vars["min_ray_coverage"].get(),
            "--coverage-weight-exponent", self.vars["coverage_weight_exponent"].get(),
            "--validation-fraction", self.vars["validation_fraction"].get(),
            "--huber-delta", self.vars["huber_delta"].get(),
            "--background-damping", self.vars["background_damping"].get(),
            "--model-damping", self.vars["model_damping"].get(),
            "--curvature-reg-factor", self.vars["curvature_reg_factor"].get(),
            "--curvature-z-factor", self.vars["curvature_z_factor"].get(),
            "--source-static-damping", self.vars["source_static_damping"].get(),
            "--global-time-damping", self.vars["global_time_damping"].get(),
            "--max-time-correction", self.vars["max_time_correction"].get(),
            "--wavelet-levels", self.vars["wavelet_levels"].get(),
            "--wavelet-threshold-factor", self.vars["wavelet_threshold_factor"].get(),
            "--hierarchical-split-rays", self.vars["hierarchical_split_rays"].get(),
            "--hierarchical-min-block-x", self.vars["hierarchical_min_block_x"].get(),
            "--hierarchical-min-block-y", self.vars["hierarchical_min_block_y"].get(),
            "--differential-weight", self.vars["differential_weight"].get(),
            "--deep-reparam-width", self.vars["deep_reparam_width"].get(),
            "--deep-reparam-depth", self.vars["deep_reparam_depth"].get(),
            "--deep-reparam-full-epochs", self.vars["deep_reparam_full_epochs"].get(),
            "--deep-reparam-starts", self.vars["deep_reparam_starts"].get(),
            "--deep-reparam-device", self.vars["deep_reparam_device"].get(),
        ]
        args.append("--edge-preserving-tv" if self.vars["edge_preserving_tv"].get() == "1"
                    else "--no-edge-preserving-tv")
        args.append("--joint-sparsity" if self.vars["joint_sparsity"].get() == "1"
                    else "--no-joint-sparsity")
        args.append(
            "--hierarchical-parameterization"
            if self.vars["hierarchical_parameterization"].get() == "1"
            else "--no-hierarchical-parameterization"
        )
        args.append("--differential-times" if self.vars["differential_times"].get() == "1"
                    else "--no-differential-times")
        args.append("--ray-length-normalization" if self.vars["ray_length_normalization"].get() == "1"
                    else "--no-ray-length-normalization")
        args.append(
            "--regularize-total-model"
            if self.vars["regularize_total_model"].get() == "1"
            else "--no-regularize-total-model"
        )
        args.append(
            "--sirt-auto-tune"
            if self.vars["sirt_auto_tune"].get() == "1"
            else "--no-sirt-auto-tune"
        )
        args.append("--allow-outside-rays" if self.vars["allow_outside_rays"].get() == "1"
                    else "--no-outside-rays")
        args.append("--event-centered-qc" if self.vars["event_centered_qc"].get() == "1"
                    else "--no-event-centered-qc")
        args.append(
            "--deep-reparameterization"
            if deep_algorithm
            else "--no-deep-reparameterization"
        )
        if use_reference_sirt:
            self.write_console(
                ">>> 反演后端：参考脚本兼容 SIRT（10 m 网格、DE 调参）。"
                "该兼容模式不使用原生静校正、TV 和模型约束参数；详见参数设置提示。\n",
                color="cmd",
            )
        if use_reference_sirt:
            # GUI SIRT uses the exact external-script-compatible backend.
            # Keep this explicit at the command boundary for old projects.
            args.append("--script-compatible-sirt")
            reference_root = self._find_reference_dataset_root()
            if reference_root is not None:
                args.extend(["--reference-dataset-root", str(reference_root)])
        task_title = "CT三维反演计算"
        if deep_algorithm:
            task_title = "2026 深度重参数化 CT 三维反演"
            self.write_console(
                ">>> 使用实验候选 DNR 后端：24×3 坐标网络、350轮、"
                "3次初始化，并输出初始化分散度；阶段9严格固定轮次"
                "泛化门槛未通过，结果须结合射线覆盖谨慎解释。\n",
                color="cmd",
            )
        if auto_algorithm:
            inversion_args = args[3:]
            args = [
                sys.executable, "-m", "wave_ct.auto_select",
                "--input-csv", self.vars["output_csv"].get(),
                "--output-dir", self.vars["slice_dir"].get(),
                "--cv-seeds", self.vars["auto_cv_seeds"].get(),
                "--pilot-outer", self.vars["auto_pilot_outer"].get(),
                "--pilot-lsqr", self.vars["auto_pilot_lsqr"].get(),
                "--", *inversion_args,
            ]
            task_title = "自动选模 CT 三维反演"
            self.write_console(
                ">>> 自动模式将按震源分组比较多种算法；耗时约为单次反演的 10–20 倍。\n",
                color="cmd",
            )
        self._run_command(args, task_title, on_success=self._after_inversion)

    def _after_inversion(self) -> None:
        self.refresh_result_images()
        self.write_console(
            ">>> 速度模型已更新，正在自动生成整套成果：速度模型TXT、带底图图件、"
            "外部参考对比、质量验证和报表。\n",
            color="success",
        )
        self.after(350, self.generate_all_results)

    def run_verify(self, on_success=None):
        if not self.validate_manual_input_ready():
            return
        self._save_runtime_project_snapshot()
        self._run_command(
            self._build_validation_command(),
            "人工走时结果验证",
            on_success=on_success or self.refresh_result_images,
        )

    def _build_validation_command(self) -> List[str]:
        args = [
            sys.executable, "-m", "wave_ct.validation_pipeline",
            "--input-csv", self.vars["output_csv"].get(),
            "--model-npz", str(Path(self.vars["slice_dir"].get()) / "velocity_model.npz"),
            "--slice-report", str(Path(self.vars["slice_dir"].get()) / "slice_report.txt"),
            "--out-dir", self.vars["verify_dir"].get(),
        ]
        for option, variable_name in (
            ("--detail-csv", "detail_csv"),
            ("--waveform-root", "waveform_root"),
            ("--evidence-csv", "evidence_csv"),
        ):
            value = self.vars[variable_name].get().strip()
            if value:
                args.extend((option, value))
        return args


def main() -> None:
    """Launch the WaveCT desktop application."""
    app = CtApp()
    app.mainloop()

if __name__ == "__main__":
    main()
