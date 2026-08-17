"""人工标记 P 波到时的 GUI。

功能：
- 选择 waveform root、station file 和输出 CSV。
- 逐个 source 加载，且每次只读取并显示一个 semv 文件。
- 鼠标点击波形即可标记 P 到时，支持上一条/下一条切换。
- 导出与自动脚本一致结构的反演数据集 CSV。

运行：
	python -m wave_ct.picker
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from wave_ct.w2_io import read_w2_trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID_RE = re.compile(r"(\d+)$", re.IGNORECASE)
STATION_RE = re.compile(r"^[^.]+\.([^.]+)\.", re.IGNORECASE)


try:
	import matplotlib.pyplot as plt

	plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
	plt.rcParams["axes.unicode_minus"] = False
except Exception:
	pass


@dataclass
class SourceMeta:
	source_name: str
	source_id: int
	source_xyz: Optional[Tuple[float, float, float]]


@dataclass
class WaveformEntry:
	source_name: str
	source_id: int
	source_xyz: Optional[Tuple[float, float, float]]
	station: str
	station_xyz: Optional[Tuple[float, float, float]]
	file_path: Path
	record_key: Optional[str] = None
	record_index: Optional[int] = None
	csv_only: bool = False
	time_origin_sec: float = 0.0
	source_delay_sec: Optional[float] = None


@dataclass
class PickState:
	pick_time_sec: Optional[float] = None
	pick_index: Optional[int] = None


def parse_source_id(source_name: str) -> int:
	match = SOURCE_ID_RE.search(source_name)
	if not match:
		return 10**9
	return int(match.group(1))


def parse_station_name(file_name: str) -> str:
	match = STATION_RE.search(file_name)
	if match:
		return match.group(1).upper()
	return Path(file_name).stem.upper()


def format_number(value: Optional[float], digits: int = 6) -> str:
	if value is None:
		return ""
	text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
	return text if text else "0"


def parse_float_or_none(value: object) -> Optional[float]:
	if value is None:
		return None
	raw = str(value).strip()
	if raw == "":
		return None
	if raw.lower() in {"none", "null", "nan", "na", "n/a"}:
		return None
	return float(raw)


def resolve_existing_path(candidates: List[Path], name: str) -> Path:
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(f"Cannot find {name}. Tried: {candidates}")


def load_station_xyz(station_file: Path) -> Dict[str, Tuple[float, float, float]]:
	station_xyz: Dict[str, Tuple[float, float, float]] = {}
	with station_file.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = line.split()
			if len(parts) < 6:
				continue
			station = parts[0].upper()
			y = float(parts[2])
			x = float(parts[3])
			z = float(parts[5])
			station_xyz[station] = (x, y, z)
	return station_xyz


def load_source_xyz(
	source_dir: Path,
	source_coord_filename: str = "output_list_sources.txt",
) -> Optional[Tuple[float, float, float]]:
	source_file = source_dir / source_coord_filename
	if not source_file.exists():
		return None
	with source_file.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			values = [v for v in line.strip().split() if v]
			if len(values) >= 3:
				return float(values[0]), float(values[1]), float(values[2])
	return None


def load_semv_waveform(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
	data = np.loadtxt(file_path, dtype=np.float64)
	if data.ndim != 2 or data.shape[1] < 2:
		raise ValueError(f"Invalid semv format: {file_path}")
	time = np.asarray(data[:, 0], dtype=np.float64)
	amplitude = np.asarray(data[:, 1], dtype=np.float64)
	return time, amplitude


def load_semv_time_origin(file_path: Path) -> float:
	with file_path.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) >= 1:
				return float(parts[0])
	return 0.0


def nearest_index(time_axis: np.ndarray, value: float) -> int:
	if time_axis.size == 0:
		return 0
	idx = int(np.searchsorted(time_axis, value))
	if idx <= 0:
		return 0
	if idx >= time_axis.size:
		return int(time_axis.size - 1)
	left = idx - 1
	if abs(time_axis[idx] - value) < abs(time_axis[left] - value):
		return idx
	return left


def entry_pick_key(entry: WaveformEntry) -> str:
	return entry.record_key or str(entry.file_path)


def write_csv(rows: List[Dict[str, object]], output_path: Path, fieldnames: List[str]) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8-sig", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


class ManualPickerApp:
	def __init__(
		self,
		root: tk.Tk,
		initial_waveform_root: Path,
		initial_station_file: Path,
		initial_output_csv: Path,
		initial_import_csv: Optional[Path] = None,
		source_pattern: str = "source_*",
		waveform_pattern: str = "*.FXZ.semv",
		source_coord_filename: str = "output_list_sources.txt",
		read_only: bool = False,
	) -> None:
		self.root = root
		self.read_only = bool(read_only)
		self.root.title(
			"Wave CT Studio - 波形标注情况（只读）"
			if self.read_only
			else "Wave CT Studio - P 波人工标注"
		)
		self.root.geometry("1480x920")

		self.waveform_root_var = tk.StringVar(value=str(initial_waveform_root))
		self.station_file_var = tk.StringVar(value=str(initial_station_file))
		self.output_csv_var = tk.StringVar(value=str(initial_output_csv))
		self.source_pattern = source_pattern or "source_*"
		self.waveform_pattern = waveform_pattern or "*.FXZ.semv"
		self.source_coord_filename = source_coord_filename or "output_list_sources.txt"
		self.source_var = tk.StringVar(value="")
		self.manual_time_var = tk.StringVar(value="")
		self.zoom_span_var = tk.StringVar(value="0.030")
		self.nudge_step_var = tk.StringVar(value="0.0003")
		self.source_delay_var = tk.StringVar(value="0.000")
		self.status_var = tk.StringVar(value="等待加载数据")
		self.info_var = tk.StringVar(value="")
		self.normalize_var = tk.BooleanVar(value=True)
		self.show_grid_var = tk.BooleanVar(value=True)
		self.vmin_var = tk.StringVar(value="1000")
		self.vmax_var = tk.StringVar(value="8000")
		self.session_file = Path(initial_output_csv).expanduser().with_name(f"{Path(initial_output_csv).expanduser().stem}_manual_picks.json")
		self.import_csv_path = Path(initial_import_csv).expanduser() if initial_import_csv else None
		self.pending_import_csv = self.import_csv_path
		self.imported_pick_count = 0
		self.unmatched_import_count = 0
		self.imported_pick_source = ""

		self.station_xyz: Dict[str, Tuple[float, float, float]] = {}
		self.source_meta: Dict[str, SourceMeta] = {}
		self.source_dirs: List[Path] = []
		self.all_entries: List[WaveformEntry] = []
		self.entries: List[WaveformEntry] = []
		self.current_index: int = -1
		self.current_time: Optional[np.ndarray] = None
		self.current_amplitude: Optional[np.ndarray] = None
		self.current_entry: Optional[WaveformEntry] = None
		self.current_pick_time: Optional[float] = None
		self.current_pick_index: Optional[int] = None
		self.current_xlim: Optional[Tuple[float, float]] = None
		self.current_time_origin: float = 0.0
		self.current_waveform_decoded = False
		self.picks: Dict[str, PickState] = {}
		self.csv_only_mode = False

		self._build_ui()
		self._bind_events()
		self.root.protocol("WM_DELETE_WINDOW", self.on_close)
		self.reload_all()

	def _build_ui(self) -> None:
		self.root.columnconfigure(1, weight=1)
		self.root.rowconfigure(1, weight=1)

		top = ttk.Frame(self.root, padding=8)
		top.grid(row=0, column=0, columnspan=2, sticky="ew")
		top.columnconfigure(1, weight=1)

		self._path_row(top, 0, "waveform root", self.waveform_root_var, self.browse_waveform_root)
		self._path_row(top, 1, "station file", self.station_file_var, self.browse_station_file)
		self._path_row(top, 2, "output csv", self.output_csv_var, self.browse_output_csv)

		left = ttk.Frame(self.root, padding=(8, 0, 8, 8))
		left.grid(row=1, column=0, sticky="nswe")
		left.columnconfigure(0, weight=1)
		left.rowconfigure(3, weight=1)

		source_bar = ttk.Frame(left)
		source_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
		source_bar.columnconfigure(1, weight=1)

		ttk.Label(source_bar, text="source").grid(row=0, column=0, sticky="w")
		self.source_combo = ttk.Combobox(source_bar, textvariable=self.source_var, state="readonly")
		self.source_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
		ttk.Button(source_bar, text="重扫", command=self.reload_all).grid(row=0, column=2, padx=(6, 0))

		file_bar = ttk.Frame(left)
		file_bar.grid(row=1, column=0, sticky="ew", pady=(0, 6))
		file_bar.columnconfigure(1, weight=1)

		ttk.Label(file_bar, text="semv").grid(row=0, column=0, sticky="w")
		ttk.Button(file_bar, text="上一条", command=self.prev_file).grid(row=0, column=1, sticky="w", padx=(6, 0))
		ttk.Button(file_bar, text="下一条", command=self.next_file).grid(row=0, column=2, sticky="w", padx=(6, 0))

		self.file_listbox = tk.Listbox(left, exportselection=False, activestyle="dotbox", height=18)
		self.file_listbox.grid(row=3, column=0, sticky="nswe")
		file_scroll = ttk.Scrollbar(left, orient="vertical", command=self.file_listbox.yview)
		file_scroll.grid(row=3, column=1, sticky="ns")
		self.file_listbox.configure(yscrollcommand=file_scroll.set)

		marker_frame = ttk.LabelFrame(left, text="手动标记", padding=8)
		marker_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
		marker_frame.columnconfigure(1, weight=1)

		ttk.Label(marker_frame, text="图上P时间(s)").grid(row=0, column=0, sticky="w")
		self.manual_time_entry = ttk.Entry(marker_frame, textvariable=self.manual_time_var, width=18)
		self.manual_time_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))
		edit_state = "disabled" if self.read_only else "normal"
		ttk.Button(
			marker_frame,
			text="应用",
			command=self.apply_manual_time,
			state=edit_state,
		).grid(row=0, column=2, padx=(6, 0))
		if self.read_only:
			self.manual_time_entry.configure(state="disabled")

		buttons = ttk.Frame(marker_frame)
		buttons.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
		for col in range(4):
			buttons.columnconfigure(col, weight=1)
		ttk.Button(
			buttons, text="标记/更新", command=self.apply_manual_time,
			state=edit_state,
		).grid(row=0, column=0, sticky="ew")
		ttk.Button(
			buttons, text="清除标记", command=self.clear_pick,
			state=edit_state,
		).grid(row=0, column=1, sticky="ew", padx=(6, 0))
		ttk.Button(
			buttons, text="保存并下一条", command=self.save_and_next,
			state=edit_state,
		).grid(row=0, column=2, sticky="ew", padx=(6, 0))
		ttk.Button(
			buttons, text="保存并上一条", command=self.save_and_prev,
			state=edit_state,
		).grid(row=0, column=3, sticky="ew", padx=(6, 0))

		self.vendor_viewer_button = ttk.Button(
			marker_frame,
			text="用原始 SeisWave 核验当前 .W 波形",
			command=self.open_current_in_vendor_viewer,
			state="disabled",
		)
		self.vendor_viewer_button.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))

		zoom_frame = ttk.LabelFrame(left, text="精细查看与微调", padding=8)
		zoom_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
		for col in range(4):
			zoom_frame.columnconfigure(col, weight=1)
		tk.Label(zoom_frame, text="窗口(s)").grid(row=0, column=0, sticky="w")
		ttk.Entry(zoom_frame, textvariable=self.zoom_span_var, width=8).grid(row=0, column=1, sticky="ew", padx=(4, 8))
		tk.Label(zoom_frame, text="微调(s)").grid(row=0, column=2, sticky="w")
		ttk.Entry(zoom_frame, textvariable=self.nudge_step_var, width=8).grid(row=0, column=3, sticky="ew", padx=(4, 0))
		ttk.Button(zoom_frame, text="放大到P点", command=self.zoom_to_pick).grid(row=1, column=0, sticky="ew", pady=(6, 0))
		ttk.Button(zoom_frame, text="放大到窗口", command=self.zoom_to_velocity_window).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))
		ttk.Button(zoom_frame, text="缩小", command=self.zoom_out).grid(row=1, column=2, sticky="ew", padx=(6, 0), pady=(6, 0))
		ttk.Button(zoom_frame, text="全图", command=self.reset_zoom).grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=(6, 0))
		ttk.Button(
			zoom_frame, text="← 微调", command=lambda: self.nudge_pick(-1),
			state=edit_state,
		).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
		ttk.Button(
			zoom_frame, text="微调 →", command=lambda: self.nudge_pick(1),
			state=edit_state,
		).grid(row=2, column=2, columnspan=2, sticky="ew", padx=(6, 0), pady=(6, 0))

		options = ttk.LabelFrame(left, text="显示选项", padding=8)
		options.grid(row=6, column=0, sticky="ew", pady=(8, 0))
		options.columnconfigure(5, weight=1)
		tk.Checkbutton(options, text="归一化显示", variable=self.normalize_var, command=self.redraw_current).grid(row=0, column=0, sticky="w")
		tk.Checkbutton(options, text="显示网格", variable=self.show_grid_var, command=self.redraw_current).grid(row=0, column=1, sticky="w", padx=(12, 0))
		tk.Label(options, text="速度下限").grid(row=0, column=2, sticky="e", padx=(12, 4))
		ttk.Entry(options, textvariable=self.vmin_var, width=8).grid(row=0, column=3, sticky="w")
		tk.Label(options, text="上限").grid(row=0, column=4, sticky="e", padx=(12, 4))
		ttk.Entry(options, textvariable=self.vmax_var, width=8).grid(row=0, column=5, sticky="w")
		tk.Label(options, text="震源延迟(s)").grid(row=1, column=0, sticky="w", pady=(6, 0))
		ttk.Entry(options, textvariable=self.source_delay_var, width=8).grid(row=1, column=1, sticky="w", pady=(6, 0))
		ttk.Button(options, text="刷新窗口", command=self.redraw_current).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(6, 0))
		tk.Button(
			options, text="导出 CSV", command=self.export_csv,
			state=edit_state,
		).grid(row=0, column=6, rowspan=2, sticky="e", padx=(12, 0))

		self.summary_label = ttk.Label(left, textvariable=self.info_var, justify="left")
		self.summary_label.grid(row=7, column=0, sticky="ew", pady=(8, 0))

		right = ttk.Frame(self.root, padding=(0, 0, 8, 8))
		right.grid(row=1, column=1, sticky="nswe")
		right.columnconfigure(0, weight=1)
		right.rowconfigure(1, weight=0)
		right.rowconfigure(2, weight=1)

		toolbar_frame = ttk.Frame(right)
		toolbar_frame.grid(row=1, column=0, sticky="ew")
		toolbar_frame.columnconfigure(0, weight=1)

		plot_frame = ttk.Frame(right)
		plot_frame.grid(row=2, column=0, sticky="nswe")
		plot_frame.columnconfigure(0, weight=1)
		plot_frame.rowconfigure(0, weight=1)

		self.fig = Figure(figsize=(10, 8), dpi=100)
		self.ax = self.fig.add_subplot(111)
		self.ax.set_title("选择一个 source 和 semv 文件")
		self.ax.set_xlabel("Time (s)")
		self.ax.set_ylabel("Amplitude")
		self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
		self.canvas_widget = self.canvas.get_tk_widget()
		self.canvas_widget.grid(row=0, column=0, sticky="nswe")
		self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
		self.toolbar.update()
		self.toolbar.pack(side=tk.TOP, fill=tk.X)

		bottom = ttk.Frame(self.root, padding=(8, 0, 8, 8))
		bottom.grid(row=2, column=0, columnspan=2, sticky="ew")
		bottom.columnconfigure(0, weight=1)
		ttk.Label(bottom, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

	def _path_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, browse_cmd) -> None:
		ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
		entry = ttk.Entry(
			parent,
			textvariable=var,
			state="readonly" if self.read_only else "normal",
		)
		entry.grid(row=row, column=1, columnspan=5, sticky="ew", padx=(6, 6), pady=2)
		ttk.Button(
			parent,
			text="浏览",
			command=browse_cmd,
			state="disabled" if self.read_only else "normal",
		).grid(row=row, column=6, sticky="e", pady=2)

	def _bind_events(self) -> None:
		self.source_combo.bind("<<ComboboxSelected>>", self.on_source_selected)
		self.file_listbox.bind("<<ListboxSelect>>", self.on_file_selected)
		self.canvas.mpl_connect("button_press_event", self.on_plot_click)
		self.canvas.mpl_connect("scroll_event", self.on_plot_scroll)
		self.root.bind("<Left>", lambda event: self.prev_file())
		self.root.bind("<Right>", lambda event: self.next_file())
		if not self.read_only:
			self.root.bind("<Shift-Left>", lambda event: self.nudge_pick(-1))
			self.root.bind("<Shift-Right>", lambda event: self.nudge_pick(1))
			self.root.bind("<Control-s>", lambda event: self.export_csv())
			self.root.bind("<Return>", lambda event: self.apply_manual_time())
			self.root.bind("<Escape>", lambda event: self.clear_pick())
		self.root.bind("<Control-0>", lambda event: self.reset_zoom())

	def browse_waveform_root(self) -> None:
		selected = filedialog.askdirectory(initialdir=self.waveform_root_var.get() or str(PROJECT_ROOT))
		if selected:
			self.waveform_root_var.set(selected)
			self.reload_all()

	def browse_station_file(self) -> None:
		selected = filedialog.askopenfilename(
			initialdir=str(PROJECT_ROOT),
			filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
		)
		if selected:
			self.station_file_var.set(selected)
			self.reload_all()

	def browse_output_csv(self) -> None:
		selected = filedialog.asksaveasfilename(
			initialdir=str(Path(self.output_csv_var.get()).parent) if self.output_csv_var.get() else str(PROJECT_ROOT),
			defaultextension=".csv",
			filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
		)
		if selected:
			self.output_csv_var.set(selected)

	def reload_all(self) -> None:
		try:
			self.save_current_pick(silent=True)
			self._scan_workspace()
			self._load_session_state()
			if self.pending_import_csv is not None:
				self._import_picks_from_csv(self.pending_import_csv)
				self.pending_import_csv = None
				self._save_session_state()
			self._populate_sources()
			if self.source_combo["values"]:
				self.source_var.set(self.source_combo["values"][0])
				self.on_source_changed(force_first=True)
			else:
				self.entries = []
				self._clear_plot("未找到 source_* 目录")
		except Exception as exc:
			messagebox.showerror("加载失败", str(exc))
			self.status_var.set(f"加载失败: {exc}")

	def _scan_workspace(self) -> None:
		waveform_root = Path(self.waveform_root_var.get()).expanduser()
		station_file = Path(self.station_file_var.get()).expanduser()
		if self.import_csv_path is not None:
			audit_path = self.import_csv_path.parent / "pick_audit.csv"
			if audit_path.is_file():
				self._scan_pick_audit(audit_path)
				return

		if not waveform_root.exists():
			raise FileNotFoundError(f"waveform root not found: {waveform_root}")
		if not station_file.exists():
			raise FileNotFoundError(f"station file not found: {station_file}")

		self.station_xyz = load_station_xyz(station_file)
		self.csv_only_mode = False
		self.source_dirs = [d for d in waveform_root.glob(self.source_pattern) if d.is_dir()]
		self.source_dirs.sort(key=lambda p: parse_source_id(p.name))
		if not self.source_dirs:
			raise RuntimeError(f"No source_* directories found under: {waveform_root}")

		self.source_meta = {}
		self.all_entries = []
		for source_index, source_dir in enumerate(self.source_dirs, start=1):
			source_name = source_dir.name
			source_xyz = load_source_xyz(source_dir, self.source_coord_filename)
			parsed_source_id = parse_source_id(source_name)
			source_id = source_index if parsed_source_id == 10**9 else parsed_source_id
			self.source_meta[source_name] = SourceMeta(
				source_name=source_name,
				source_id=source_id,
				source_xyz=source_xyz,
			)
			semv_files = sorted(source_dir.glob(self.waveform_pattern), key=lambda p: parse_station_name(p.name))
			for semv_file in semv_files:
				station = parse_station_name(semv_file.name)
				self.all_entries.append(
					WaveformEntry(
						source_name=source_name,
						source_id=self.source_meta[source_name].source_id,
						source_xyz=source_xyz,
						station=station,
						station_xyz=self.station_xyz.get(station),
						file_path=semv_file,
					)
				)

	def _scan_pick_audit(self, audit_path: Path) -> None:
		"""Build a review list from an auditable proprietary-waveform conversion."""
		self.csv_only_mode = True
		self.station_xyz = {}
		self.source_dirs = []
		self.source_meta = {}
		self.all_entries = []
		source_ids: Dict[str, int] = {}
		with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
			for row in csv.DictReader(handle):
				if str(row.get("status", "KEEP")).strip().upper() != "KEEP":
					continue
				source_name = str(
					row.get("event_name") or row.get("event_time") or row.get("event_id") or ""
				).strip()
				waveform_file = str(row.get("waveform_file") or "").strip()
				station_id = int(float(row.get("station_id") or 0))
				if not source_name or not waveform_file or station_id <= 0:
					continue
				if source_name not in source_ids:
					source_ids[source_name] = len(source_ids) + 1
				try:
					source_xyz = (float(row["source_x"]), float(row["source_y"]), float(row["source_z"]))
					station_xyz = (float(row["station_x"]), float(row["station_y"]), float(row["station_z"]))
					origin_sec = float(
						row.get("origin_ms") or row.get("estimated_origin_ms") or 0.0
					) / 1000.0
					record_index_text = str(
						row.get("record_index") or row.get("block_index") or ""
					).strip()
					record_index = int(float(record_index_text)) if record_index_text else None
				except (KeyError, TypeError, ValueError):
					continue
				source_id = source_ids[source_name]
				self.source_meta[source_name] = SourceMeta(source_name, source_id, source_xyz)
				station = f"STA{station_id:02d}"
				path = Path(waveform_file)
				self.all_entries.append(WaveformEntry(
					source_name=source_name,
					source_id=source_id,
					source_xyz=source_xyz,
					station=station,
					station_xyz=station_xyz,
					file_path=path,
					record_key=f"{path}::{station}",
					record_index=record_index,
					csv_only=True,
					time_origin_sec=origin_sec,
				))
		if not self.all_entries:
			raise RuntimeError(f"pick_audit.csv 中没有可审核的人工P标记: {audit_path}")

	def _session_payload(self) -> Dict[str, object]:
		items = []
		for file_path, pick in self.picks.items():
			items.append(
				{
					"file_path": file_path,
					"pick_time_sec": pick.pick_time_sec,
					"pick_index": pick.pick_index,
				}
			)
		return {
			"waveform_root": self.waveform_root_var.get(),
			"station_file": self.station_file_var.get(),
			"output_csv": self.output_csv_var.get(),
			"picks": items,
		}

	def _save_session_state(self) -> None:
		self.session_file.parent.mkdir(parents=True, exist_ok=True)
		with self.session_file.open("w", encoding="utf-8") as f:
			json.dump(self._session_payload(), f, ensure_ascii=False, indent=2)

	def _load_session_state(self) -> None:
		self.picks = {}
		if not self.session_file.exists():
			return
		try:
			with self.session_file.open("r", encoding="utf-8") as f:
				data = json.load(f)
			for item in data.get("picks", []):
				file_path = item.get("file_path")
				if not file_path:
					continue
				pick_time_sec = parse_float_or_none(item.get("pick_time_sec"))
				pick_index = item.get("pick_index")
				if pick_index is not None and str(pick_index).strip() != "":
					pick_index = int(pick_index)
				else:
					pick_index = None
				if pick_time_sec is None:
					continue
				self.picks[str(file_path)] = PickState(pick_time_sec=pick_time_sec, pick_index=pick_index)
		except Exception:
			self.picks = {}

	@staticmethod
	def _source_matches(entry: WaveformEntry, value: object) -> bool:
		text = str(value or "").strip().lower()
		if not text:
			return False
		if text == entry.source_name.lower():
			return True
		if not (text.isdigit() or re.fullmatch(r"source[_\- ]*\d+", text)):
			return False
		parsed = parse_source_id(text)
		return parsed != 10**9 and parsed == entry.source_id

	@staticmethod
	def _pick_time_from_row(row: Dict[str, str]) -> Optional[float]:
		for key in ("raw_pick_time_s", "p_time_s", "pick_time_sec"):
			value = parse_float_or_none(row.get(key))
			if value is not None:
				return value
		for key in ("raw_pick_time_ms", "raw_arrival_ms", "台站P波到时", "pick_ms", "P波到时_ms"):
			value = parse_float_or_none(row.get(key))
			if value is not None:
				return value / 1000.0
		return None

	@staticmethod
	def _pick_overlay_csv(csv_path: Path) -> Path:
		"""Prefer the detail table because inversion CSV stores travel, not plot-axis, time."""
		candidates = [
			csv_path,
			csv_path.with_name("pick_audit.csv"),
			csv_path.with_name("P波拾取明细.csv"),
			csv_path.with_name(f"{csv_path.stem}_detail.csv"),
			csv_path.with_name("反演数据集_detail.csv"),
		]
		for candidate in candidates:
			if not candidate.is_file():
				continue
			try:
				with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
					header = next(csv.reader(handle), [])
			except OSError:
				continue
			if any(key in header for key in ("raw_pick_time_s", "raw_pick_time_ms", "raw_arrival_ms", "pick_time_sec", "pick_ms")):
				return candidate
		return csv_path

	def _import_picks_from_csv(self, csv_path: Path) -> None:
		"""Overlay observed CSV picks on real waveform entries without changing them."""
		if not csv_path.is_file():
			raise FileNotFoundError(f"走时 CSV 不存在: {csv_path}")

		by_source: Dict[int, List[WaveformEntry]] = {}
		by_source_name: Dict[str, List[WaveformEntry]] = {}
		for entry in self.all_entries:
			by_source.setdefault(entry.source_id, []).append(entry)
			by_source_name.setdefault(entry.source_name.lower(), []).append(entry)

		overlay_csv = self._pick_overlay_csv(csv_path)
		matched = 0
		unmatched = 0
		with overlay_csv.open("r", encoding="utf-8-sig", newline="") as handle:
			for row in csv.DictReader(handle):
				pick_time = self._pick_time_from_row(row)
				if pick_time is None:
					continue
				source_value = (
					row.get("source")
					or row.get("source_name")
					or row.get("震源事件文件名")
					or row.get("event_name")
					or row.get("震源编号")
				)
				source_text = str(source_value or "").strip().lower()
				if source_text in by_source_name:
					candidates = list(by_source_name[source_text])
				elif source_text.isdigit() or re.fullmatch(r"source[_\- ]*\d+", source_text):
					source_id = parse_source_id(source_text)
					candidates = list(by_source.get(source_id, []))
				else:
					candidates = []
				if not candidates:
					unmatched += 1
					continue

				station_value = str(row.get("station") or row.get("station_id") or "").strip().upper()
				if station_value.isdigit():
					station_value = f"STA{int(station_value):02d}"
				if station_value:
					station_matches = [entry for entry in candidates if entry.station.upper() == station_value]
					if station_matches:
						candidates = station_matches

				coords = None
				try:
					coords = (
						float(row["台站坐标-x"]),
						float(row["台站坐标-y"]),
						float(row["台站坐标-z"]),
					)
				except (KeyError, TypeError, ValueError):
					pass
				if coords is not None:
					with_coords = [entry for entry in candidates if entry.station_xyz is not None]
					if with_coords:
						candidates = sorted(
							with_coords,
							key=lambda entry: sum((entry.station_xyz[i] - coords[i]) ** 2 for i in range(3)),
						)
						best_error = sum((candidates[0].station_xyz[i] - coords[i]) ** 2 for i in range(3)) ** 0.5
						if best_error > 1.0:
							unmatched += 1
							continue

				if len(candidates) != 1 and not station_value and coords is None:
					unmatched += 1
					continue
				entry = candidates[0]
				origin = parse_float_or_none(row.get("time_origin_s"))
				if origin is None:
					origin = parse_float_or_none(row.get("time_origin_sec"))
				if origin is not None:
					entry.time_origin_sec = origin
				delay = parse_float_or_none(row.get("source_delay_s"))
				if delay is None:
					delay = parse_float_or_none(row.get("source_delay_sec"))
				if delay is not None:
					entry.source_delay_sec = delay
				self.picks[entry_pick_key(entry)] = PickState(pick_time_sec=pick_time, pick_index=None)
				matched += 1

		self.imported_pick_count = matched
		self.unmatched_import_count = unmatched
		self.imported_pick_source = overlay_csv.name
		self.status_var.set(f"已从 {overlay_csv.name} 恢复 {matched} 条波形轴P标记，未匹配 {unmatched} 条")
		if matched == 0:
			messagebox.showwarning(
				"CSV 标记无法对应波形",
				"CSV 已导入反演，但没有记录能与当前波形目录对应。\n\n"
				"请确认波形目录、source 名称和台站坐标属于同一数据集。"
			)

	def _populate_sources(self) -> None:
		values = [meta.source_name for meta in self.source_meta.values()]
		self.source_combo["values"] = values

	def on_source_selected(self, event=None) -> None:
		self.on_source_changed(force_first=True)

	def on_source_changed(self, force_first: bool = False) -> None:
		self.save_current_pick(silent=True)
		source_name = self.source_var.get().strip()
		if not source_name or source_name not in self.source_meta:
			return

		# Keep the scanned entries instead of rebuilding them here. Imported CSV
		# metadata (raw-axis origin and source delay) is attached to these objects.
		self.entries = [entry for entry in self.all_entries if entry.source_name == source_name]

		self._refresh_file_list()
		if self.entries:
			index = 0 if force_first else max(0, min(self.current_index, len(self.entries) - 1))
			self.load_entry(index)
		else:
			self.current_index = -1
			self.current_entry = None
			self._clear_plot(f"{source_name} 下没有找到 {self.waveform_pattern}")

	def _refresh_file_list(self) -> None:
		self.file_listbox.delete(0, tk.END)
		for entry in self.entries:
			key = entry_pick_key(entry)
			pick = self.picks.get(key)
			if pick and pick.pick_time_sec is not None:
				label = f"{entry.station} | {entry.file_path.name} | {pick.pick_time_sec:.4f}s"
			else:
				label = f"{entry.station} | {entry.file_path.name}"
			self.file_listbox.insert(tk.END, label)

	def on_file_selected(self, event=None) -> None:
		if not self.file_listbox.curselection():
			return
		index = int(self.file_listbox.curselection()[0])
		self.load_entry(index)

	def prev_file(self) -> None:
		if not self.entries:
			return
		target = max(0, self.current_index - 1)
		self.load_entry(target)

	def next_file(self) -> None:
		if not self.entries:
			return
		target = min(len(self.entries) - 1, self.current_index + 1)
		self.load_entry(target)

	def load_entry(self, index: int) -> None:
		if not self.entries:
			return
		if index < 0 or index >= len(self.entries):
			return

		self.save_current_pick(silent=True)
		self.current_index = index
		self.current_entry = self.entries[index]
		self.file_listbox.selection_clear(0, tk.END)
		self.file_listbox.selection_set(index)
		self.file_listbox.see(index)
		self.current_pick_time = None
		self.current_pick_index = None
		self.manual_time_var.set("")
		self.current_xlim = None
		self.current_waveform_decoded = False
		if self.current_entry.source_delay_sec is not None:
			self.source_delay_var.set(format_number(self.current_entry.source_delay_sec))

		try:
			if self.current_entry.csv_only and self.current_entry.file_path.suffix.lower() == ".w2":
				station_digits = re.search(r"(\d+)$", self.current_entry.station)
				station_id = int(station_digits.group(1)) if station_digits else None
				self.current_time, self.current_amplitude, w2_header, w2_record = read_w2_trace(
					self.current_entry.file_path,
					station_id=station_id,
					record_index=self.current_entry.record_index,
				)
				self.current_time_origin = w2_header.origin_ms / 1000.0
				self.current_entry.time_origin_sec = self.current_time_origin
				self.current_waveform_decoded = True
			elif self.current_entry.csv_only:
				pick = self.picks.get(entry_pick_key(self.current_entry))
				center = pick.pick_time_sec if pick and pick.pick_time_sec is not None else self.current_entry.time_origin_sec
				self.current_time = np.linspace(max(0.0, center - 0.15), center + 0.15, 301)
				self.current_amplitude = np.zeros_like(self.current_time)
				self.current_time_origin = self.current_entry.time_origin_sec
			else:
				self.current_time, self.current_amplitude = load_semv_waveform(self.current_entry.file_path)
				# The first SEMV sample may be negative to contain the acausal tail of
				# the source wavelet. It is not the physical source origin.
				self.current_time_origin = self.current_entry.time_origin_sec
			self._apply_existing_pick()
			if self.current_pick_time is not None and self.imported_pick_count > 0:
				try:
					span = max(float(self.zoom_span_var.get().strip()), 0.002)
				except Exception:
					span = 0.03
				self.current_xlim = self.clamp_xlim(
					self.current_pick_time - span / 2.0,
					self.current_pick_time + span / 2.0,
				)
			self.vendor_viewer_button.configure(
				state="normal" if self.current_entry.csv_only and self.current_entry.file_path.is_file() else "disabled"
			)
			self.redraw_current()
			self._update_info()
			if self.current_entry.csv_only and self.current_waveform_decoded:
				self.status_var.set(
					f"已校验解码 W2 真实波形: {self.current_entry.source_name} / "
					f"{self.current_entry.station}"
				)
			elif self.current_entry.csv_only:
				self.status_var.set("砚北.W文件头人工P标记审核：未解码专有振幅，不显示伪造波形")
			else:
				self.status_var.set(f"已加载 {self.current_entry.source_name} / {self.current_entry.station} / {self.current_entry.file_path.name}")
		except Exception as exc:
			self.current_time = None
			self.current_amplitude = None
			self.current_time_origin = 0.0
			self.current_waveform_decoded = False
			self._clear_plot(f"读取失败: {exc}")
			self.status_var.set(f"读取失败: {self.current_entry.file_path} -> {exc}")

	def _apply_existing_pick(self) -> None:
		if self.current_entry is None:
			return
		key = entry_pick_key(self.current_entry)
		pick = self.picks.get(key)
		if not pick or pick.pick_time_sec is None:
			return
		self.current_pick_time = pick.pick_time_sec
		if self.current_time is not None:
			self.current_pick_index = nearest_index(self.current_time, pick.pick_time_sec)
		self.manual_time_var.set(format_number(pick.pick_time_sec))

	def pick_to_travel_time(self, pick_time_sec: Optional[float], origin_sec: Optional[float] = None) -> Optional[float]:
		if pick_time_sec is None:
			return None
		if origin_sec is None:
			origin_sec = self.current_time_origin
		return float(pick_time_sec) - float(origin_sec) - self.source_delay_sec()

	def source_delay_sec(self) -> float:
		try:
			return float(self.source_delay_var.get().strip())
		except Exception:
			return 0.0

	def current_distance_and_window(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
		if self.current_entry is None or self.current_entry.source_xyz is None or self.current_entry.station_xyz is None:
			return None, None, None
		try:
			vmin = float(self.vmin_var.get().strip())
			vmax = float(self.vmax_var.get().strip())
		except Exception:
			return None, None, None
		if vmin <= 0 or vmax <= 0 or vmin > vmax:
			return None, None, None
		sx, sy, sz = self.current_entry.source_xyz
		rx, ry, rz = self.current_entry.station_xyz
		distance_m = float(np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2 + (sz - rz) ** 2))
		delay = self.source_delay_sec()
		t_min = distance_m / vmax + self.current_time_origin + delay
		t_max = distance_m / vmin + self.current_time_origin + delay
		return distance_m, t_min, t_max

	def redraw_current(self) -> None:
		if self.current_time is None or self.current_amplitude is None or self.current_entry is None:
			return

		self.ax.clear()
		if self.normalize_var.get():
			amp = self.current_amplitude.astype(np.float64)
			amp = amp - np.mean(amp)
			scale = np.std(amp)
			if scale == 0:
				scale = 1.0
			plot_amp = amp / scale
			ylabel = "Normalized amplitude"
		else:
			plot_amp = self.current_amplitude
			ylabel = "Amplitude"

		has_real_waveform = not self.current_entry.csv_only or self.current_waveform_decoded
		if self.current_entry.csv_only and not self.current_waveform_decoded:
			self.ax.text(
				0.5, 0.72,
				"当前红线来自 .W 文件头的人工 P 标记\n"
				"专有振幅尚未通过解码校验，本图不能判断初至是否准确\n"
				"请点击左侧按钮，用原始 SeisWave 查看真实波形",
				transform=self.ax.transAxes,
				ha="center",
				va="center",
				color="#475569",
				fontsize=13,
			)
		if has_real_waveform:
			self.ax.plot(self.current_time, plot_amp, color="black", linewidth=0.8)
		distance_m, t_min, t_max = self.current_distance_and_window()
		if has_real_waveform and distance_m is not None and t_min is not None and t_max is not None:
			vmin = float(self.vmin_var.get().strip())
			vmax = float(self.vmax_var.get().strip())
			self.ax.axvspan(
				t_min,
				t_max,
				color="#22C55E",
				alpha=0.16,
				label=f"{vmin:g}-{vmax:g} m/s 参考窗口(已修正起点)",
			)
			self.ax.axvline(t_min, color="#16A34A", linestyle=":", linewidth=1.1)
			self.ax.axvline(t_max, color="#16A34A", linestyle=":", linewidth=1.1)
			y_top = float(np.nanmax(plot_amp)) if plot_amp.size else 0.0
			self.ax.text(t_min, y_top, f"{t_min:.4f}s", color="#166534", fontsize=9, va="top")
			self.ax.text(t_max, y_top, f"{t_max:.4f}s", color="#166534", fontsize=9, va="top", ha="right")
		if self.current_pick_time is not None:
			pick_value = self.current_pick_time
			self.ax.axvline(pick_value, color="red", linestyle="--", linewidth=1.2)
			if self.current_pick_index is not None and 0 <= self.current_pick_index < self.current_time.size:
				y_value = plot_amp[self.current_pick_index]
			else:
				y_value = np.interp(pick_value, self.current_time, plot_amp)
			self.ax.scatter([pick_value], [y_value], color="red", s=40, zorder=5)
			self.ax.annotate(f"{pick_value:.4f}s", (pick_value, y_value), xytext=(8, 8), textcoords="offset points", color="red")

		self.ax.set_title(f"{self.current_entry.source_name} | {self.current_entry.station} | {self.current_entry.file_path.name}")
		self.ax.set_xlabel("Time (s)")
		self.ax.set_ylabel(ylabel)
		if self.show_grid_var.get():
			self.ax.grid(True, linestyle="--", alpha=0.35)
		if has_real_waveform and distance_m is not None:
			self.ax.legend(loc="upper right")
		if self.current_xlim is not None:
			self.ax.set_xlim(*self.current_xlim)
			self.ax.relim()
			self.ax.autoscale_view(scalex=False, scaley=True)
		else:
			self.fig_autoscale()
		self.canvas.draw_idle()

	def _find_vendor_viewer(self, waveform_path: Path) -> Optional[Path]:
		for parent in (waveform_path.parent, *waveform_path.parents):
			for candidate in (parent / "SeisWave.exe", parent / "SOS" / "SeisWave.exe"):
				if candidate.is_file():
					return candidate
		return None

	def open_current_in_vendor_viewer(self) -> None:
		"""Open the original viewer instead of synthesizing proprietary amplitudes."""
		if self.current_entry is None or not self.current_entry.file_path.is_file():
			messagebox.showwarning("无法核验", "当前记录没有可访问的原始 .W 文件。")
			return
		viewer = self._find_vendor_viewer(self.current_entry.file_path)
		if viewer is None:
			messagebox.showwarning("缺少原始软件", "没有在该月份数据目录中找到 SOS/SeisWave.exe。")
			return
		result = ctypes.windll.shell32.ShellExecuteW(
			None,
			"runas",
			str(viewer),
			f'"{self.current_entry.file_path}"',
			str(viewer.parent),
			1,
		)
		if result <= 32:
			messagebox.showerror("启动失败", f"SeisWave 启动失败，系统返回码: {result}")
			return
		self.root.clipboard_clear()
		self.root.clipboard_append(str(self.current_entry.file_path))
		self.status_var.set("已启动原始 SeisWave；当前 .W 路径已复制，可在 File > Open AS file 中打开。")

	def fig_autoscale(self) -> None:
		if self.current_time is None or self.current_amplitude is None:
			return
		self.ax.relim()
		self.ax.autoscale_view()

	def _clear_plot(self, title: str) -> None:
		self.ax.clear()
		self.ax.set_title(title)
		self.ax.set_xlabel("Time (s)")
		self.ax.set_ylabel("Amplitude")
		self.canvas.draw_idle()
		self.info_var.set("")

	def clamp_xlim(self, left: float, right: float) -> Tuple[float, float]:
		if self.current_time is None or self.current_time.size == 0:
			return left, right
		full_left = float(np.nanmin(self.current_time))
		full_right = float(np.nanmax(self.current_time))
		if right <= left:
			return full_left, full_right
		width = min(right - left, full_right - full_left)
		if left < full_left:
			left = full_left
			right = left + width
		if right > full_right:
			right = full_right
			left = right - width
		return max(full_left, left), min(full_right, right)

	def set_zoom_center(self, center: float, span: float) -> None:
		if self.current_time is None or span <= 0:
			return
		left, right = self.clamp_xlim(center - span / 2.0, center + span / 2.0)
		self.current_xlim = (left, right)
		self.redraw_current()

	def zoom_to_pick(self) -> None:
		if self.current_pick_time is None:
			messagebox.showinfo("提示", "请先点击波形或输入 P 到时。")
			return
		try:
			span = float(self.zoom_span_var.get().strip())
		except Exception:
			span = 0.03
		self.set_zoom_center(self.current_pick_time, span)

	def zoom_to_velocity_window(self) -> None:
		_, t_min, t_max = self.current_distance_and_window()
		if t_min is None or t_max is None:
			messagebox.showinfo("提示", "当前记录缺少坐标或速度范围，无法计算参考窗口。")
			return
		padding = max((t_max - t_min) * 0.6, 0.006)
		self.current_xlim = self.clamp_xlim(t_min - padding, t_max + padding)
		self.redraw_current()

	def zoom_out(self) -> None:
		if self.current_time is None:
			return
		left, right = self.current_xlim if self.current_xlim is not None else self.ax.get_xlim()
		center = 0.5 * (left + right)
		span = (right - left) * 1.8
		self.set_zoom_center(center, span)

	def reset_zoom(self) -> None:
		self.current_xlim = None
		self.redraw_current()

	def nudge_pick(self, direction: int) -> None:
		if self.read_only:
			return
		if self.current_entry is not None and self.current_entry.csv_only and not self.current_waveform_decoded:
			self.status_var.set("文件头标记审核模式为只读，不能在未解码波形上微调")
			return
		if self.current_time is None:
			return
		if self.current_pick_time is None:
			try:
				self.current_pick_time = float(self.manual_time_var.get().strip())
			except Exception:
				messagebox.showinfo("提示", "请先点击波形或输入 P 到时。")
				return
		try:
			step = abs(float(self.nudge_step_var.get().strip()))
		except Exception:
			step = 0.0003
		left = float(np.nanmin(self.current_time))
		right = float(np.nanmax(self.current_time))
		self.current_pick_time = min(max(self.current_pick_time + direction * step, left), right)
		self.current_pick_index = nearest_index(self.current_time, self.current_pick_time)
		self.manual_time_var.set(format_number(self.current_pick_time))
		self.save_current_pick(silent=True)
		self.redraw_current()
		self._update_info()
		self.status_var.set(f"已微调到时: {self.current_pick_time:.6f} s")

	def on_plot_scroll(self, event) -> None:
		if event.inaxes != self.ax or event.xdata is None or self.current_time is None:
			return
		left, right = self.current_xlim if self.current_xlim is not None else self.ax.get_xlim()
		factor = 0.72 if event.button == "up" else 1.38
		new_width = (right - left) * factor
		center = float(event.xdata)
		ratio = (center - left) / (right - left) if right > left else 0.5
		new_left = center - new_width * ratio
		new_right = new_left + new_width
		self.current_xlim = self.clamp_xlim(new_left, new_right)
		self.redraw_current()

	def on_plot_click(self, event) -> None:
		if self.read_only:
			return
		if self.current_entry is not None and self.current_entry.csv_only and not self.current_waveform_decoded:
			self.status_var.set("文件头标记审核模式为只读，不能在未解码波形上重新标注")
			return
		if event.inaxes != self.ax or event.xdata is None or self.current_time is None:
			return
		if event.button != 1:
			return
		toolbar_mode = getattr(getattr(self, "toolbar", None), "mode", "")
		mode_text = str(toolbar_mode).strip().lower()
		if toolbar_mode and mode_text not in {"", "none", "_mode.none"}:
			return
		self.current_pick_time = float(event.xdata)
		self.current_pick_index = nearest_index(self.current_time, self.current_pick_time)
		self.manual_time_var.set(format_number(self.current_pick_time))
		self.redraw_current()
		self._update_info()
		self.status_var.set(f"已标记 {self.current_entry.station if self.current_entry else ''}: {self.current_pick_time:.6f} s")

	def apply_manual_time(self) -> None:
		if self.read_only:
			return
		if self.current_entry is not None and self.current_entry.csv_only and not self.current_waveform_decoded:
			self.status_var.set("文件头标记审核模式为只读")
			return
		if self.current_time is None or self.current_entry is None:
			return
		try:
			value = float(self.manual_time_var.get().strip())
		except Exception:
			messagebox.showwarning("输入错误", "请输入合法的时间数值，例如 0.1234")
			return
		self.current_pick_time = value
		self.current_pick_index = nearest_index(self.current_time, value)
		self.redraw_current()
		self.save_current_pick(silent=True)
		self._update_info()

	def clear_pick(self) -> None:
		if self.read_only:
			return
		if self.current_entry is not None and self.current_entry.csv_only and not self.current_waveform_decoded:
			self.status_var.set("文件头标记审核模式为只读")
			return
		self.current_pick_time = None
		self.current_pick_index = None
		self.manual_time_var.set("")
		self.redraw_current()
		self.save_current_pick(silent=True)
		self._update_info()
		self.status_var.set("已清除当前标记")

	def save_current_pick(self, silent: bool = False) -> None:
		if self.read_only:
			return
		if self.current_entry is None:
			return
		key = entry_pick_key(self.current_entry)
		if self.current_pick_time is None:
			self.picks.pop(key, None)
		else:
			self.picks[key] = PickState(pick_time_sec=self.current_pick_time, pick_index=self.current_pick_index)
		self._refresh_file_list()
		self._update_info()
		self._save_session_state()
		if not silent:
			self.status_var.set(f"已保存 {self.current_entry.station}")

	def save_and_next(self) -> None:
		self.save_current_pick()
		self.next_file()

	def save_and_prev(self) -> None:
		self.save_current_pick()
		self.prev_file()

	def _build_export_rows(self, vmin: float, vmax: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
		final_rows: List[Dict[str, object]] = []
		detail_rows: List[Dict[str, object]] = []
		counters = {
			"total": 0,
			"kept": 0,
			"drop_not_picked": 0,
			"drop_non_positive_time": 0,
			"drop_missing_coord": 0,
			"drop_invalid_velocity": 0,
			"velocity_out_of_range": 0,
		}
		source_delay_sec = self.source_delay_sec()

		for entry in self.all_entries:
			counters["total"] += 1
			key = entry_pick_key(entry)
			pick = self.picks.get(key)
			p_time_sec = pick.pick_time_sec if pick else None
			time_origin_sec = entry.time_origin_sec
			travel_time_sec = self.pick_to_travel_time(p_time_sec, time_origin_sec)
			if travel_time_sec is None or travel_time_sec <= 0:
				reason = "not_picked" if travel_time_sec is None else "non_positive_travel_time"
				if travel_time_sec is not None and travel_time_sec <= 0:
					counters["drop_non_positive_time"] += 1
				else:
					counters["drop_not_picked"] += 1
				detail_rows.append({
					"source_id": str(entry.source_id),
					"source_name": entry.source_name,
					"station": entry.station,
					"file_name": entry.file_path.name,
					"p_time_sec": "",
					"travel_time_sec": "" if travel_time_sec is None else format_number(travel_time_sec),
					"time_origin_sec": format_number(time_origin_sec),
					"source_delay_sec": format_number(source_delay_sec),
					"p_index": "",
					"keep": "0",
					"drop_reason": reason,
				})
				continue

			if entry.source_xyz is None or entry.station_xyz is None:
				counters["drop_missing_coord"] += 1
				detail_rows.append({
					"source_id": str(entry.source_id),
					"source_name": entry.source_name,
					"station": entry.station,
					"file_name": entry.file_path.name,
					"distance_m": "",
					"velocity_mps": "",
					"p_time_sec": format_number(p_time_sec),
					"travel_time_sec": format_number(travel_time_sec),
					"time_origin_sec": format_number(time_origin_sec),
					"source_delay_sec": format_number(source_delay_sec),
					"p_index": "" if not pick or pick.pick_index is None else str(int(pick.pick_index)),
					"keep": "0",
					"drop_reason": "missing_coordinates",
				})
				continue

			sx, sy, sz = entry.source_xyz
			rx, ry, rz = entry.station_xyz
			distance_m = float(np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2 + (sz - rz) ** 2))
			p_time_ms = travel_time_sec * 1000.0
			raw_pick_time_ms = p_time_sec * 1000.0
			velocity_mps = distance_m / travel_time_sec

			if not np.isfinite(velocity_mps):
				counters["drop_invalid_velocity"] += 1
				detail_rows.append({
					"source_id": str(entry.source_id),
					"source_name": entry.source_name,
					"station": entry.station,
					"file_name": entry.file_path.name,
					"distance_m": format_number(distance_m),
					"velocity_mps": format_number(velocity_mps),
					"source_x": format_number(sx),
					"source_y": format_number(sy),
					"source_z": format_number(sz),
					"station_x": format_number(rx),
					"station_y": format_number(ry),
					"station_z": format_number(rz),
					"p_time_sec": format_number(p_time_sec),
					"travel_time_sec": format_number(travel_time_sec),
					"time_origin_sec": format_number(time_origin_sec),
					"source_delay_sec": format_number(source_delay_sec),
					"p_time_ms": format_number(p_time_ms),
					"raw_pick_time_ms": format_number(raw_pick_time_ms),
					"p_index": "" if not pick or pick.pick_index is None else str(int(pick.pick_index)),
					"keep": "0",
					"drop_reason": "invalid_velocity",
				})
				continue

			velocity_warning = ""
			if not (vmin <= velocity_mps <= vmax):
				counters["velocity_out_of_range"] += 1
				velocity_warning = "velocity_out_of_reference_range"

			final_rows.append({
				"震源编号": str(entry.source_id),
				"震源坐标-x": format_number(sx),
				"震源坐标-y": format_number(sy),
				"震源坐标-z": format_number(sz),
				"发震时刻t": "0",
				"台站坐标-x": format_number(rx),
				"台站坐标-y": format_number(ry),
				"台站坐标-z": format_number(rz),
				"台站P波到时": format_number(p_time_ms),
				"震源-台站传播时间": format_number(p_time_ms),
				"震源事件文件名": entry.source_name,
			})
			detail_rows.append({
				"source_id": str(entry.source_id),
				"source_name": entry.source_name,
				"station": entry.station,
				"file_name": entry.file_path.name,
				"distance_m": format_number(distance_m),
				"velocity_mps": format_number(velocity_mps),
				"source_x": format_number(sx),
				"source_y": format_number(sy),
				"source_z": format_number(sz),
				"station_x": format_number(rx),
				"station_y": format_number(ry),
				"station_z": format_number(rz),
				"p_time_sec": format_number(p_time_sec),
				"travel_time_sec": format_number(travel_time_sec),
				"time_origin_sec": format_number(time_origin_sec),
				"source_delay_sec": format_number(source_delay_sec),
				"p_time_ms": format_number(p_time_ms),
				"raw_pick_time_ms": format_number(raw_pick_time_ms),
				"p_index": "" if not pick or pick.pick_index is None else str(int(pick.pick_index)),
				"keep": "1",
				"drop_reason": velocity_warning,
			})
			counters["kept"] += 1

		return final_rows, detail_rows, counters

	def export_csv(self) -> None:
		if self.read_only:
			return
		self.save_current_pick(silent=True)
		try:
			vmin = float(self.vmin_var.get().strip())
			vmax = float(self.vmax_var.get().strip())
		except Exception:
			messagebox.showwarning("参数错误", "速度范围必须是数字，例如 1500 和 6000")
			return
		if vmin > vmax:
			messagebox.showwarning("参数错误", "速度下限不能大于上限")
			return

		final_rows, detail_rows, counters = self._build_export_rows(vmin, vmax)

		output_csv = Path(self.output_csv_var.get()).expanduser()
		output_csv.parent.mkdir(parents=True, exist_ok=True)

		final_fields = [
			"震源编号",
			"震源坐标-x",
			"震源坐标-y",
			"震源坐标-z",
			"发震时刻t",
			"台站坐标-x",
			"台站坐标-y",
			"台站坐标-z",
			"台站P波到时",
			"震源-台站传播时间",
			"震源事件文件名",
		]
		write_csv(final_rows, output_csv, final_fields)

		detail_csv = output_csv.with_name(f"{output_csv.stem}_detail.csv")
		detail_fields = [
			"source_id",
			"source_name",
			"station",
			"file_name",
			"distance_m",
			"velocity_mps",
			"source_x",
			"source_y",
			"source_z",
			"station_x",
			"station_y",
			"station_z",
			"p_time_sec",
			"travel_time_sec",
			"time_origin_sec",
			"source_delay_sec",
			"p_time_ms",
			"raw_pick_time_ms",
			"p_index",
			"keep",
			"drop_reason",
		]
		write_csv(detail_rows, detail_csv, detail_fields)

		report = output_csv.with_name(f"{output_csv.stem}_report.txt")
		with report.open("w", encoding="utf-8") as f:
			f.write("P 波人工标记导出报告\n")
			f.write("=" * 60 + "\n")
			f.write(f"waveform_root={self.waveform_root_var.get()}\n")
			f.write(f"station_file={self.station_file_var.get()}\n")
			f.write(f"output_csv={output_csv}\n")
			f.write(f"detail_csv={detail_csv}\n\n")
			f.write("time_origin_mode=physical_source_time_zero\n")
			f.write(f"source_delay_sec={self.source_delay_sec():.6f}\n")
			f.write("travel_time_sec=pick_time_sec-time_origin_sec-source_delay_sec\n\n")
			f.write(f"velocity_range=[{vmin:.1f}, {vmax:.1f}]\n\n")
			for key in ("total", "kept", "drop_not_picked", "drop_non_positive_time", "drop_missing_coord", "drop_invalid_velocity", "velocity_out_of_range"):
				f.write(f"  {key}: {counters[key]}\n")

		self.status_var.set(f"已导出 {output_csv}，有效记录 {counters['kept']} 条")
		self._save_session_state()
		messagebox.showinfo(
			"导出完成",
			f"已输出：\n{output_csv}\n{detail_csv}\n{report}\n\n有效记录：{counters['kept']} / {counters['total']}",
		)

	def on_close(self) -> None:
		if not self.read_only:
			try:
				self.save_current_pick(silent=True)
				self._save_session_state()
			except Exception:
				pass
		self.root.destroy()

	def _update_info(self) -> None:
		if self.current_entry is None:
			self.info_var.set("")
			return
		source_meta = self.source_meta.get(self.current_entry.source_name)
		source_text = "未找到源坐标"
		if source_meta and source_meta.source_xyz is not None:
			sx, sy, sz = source_meta.source_xyz
			source_text = f"源: ({sx:.3f}, {sy:.3f}, {sz:.3f})"
		station_text = "未找到台站坐标"
		if self.current_entry.station_xyz is not None:
			rx, ry, rz = self.current_entry.station_xyz
			station_text = f"台站: ({rx:.3f}, {ry:.3f}, {rz:.3f})"
		pick_text = "未标记"
		if self.current_pick_time is not None:
			travel_time = self.pick_to_travel_time(self.current_pick_time)
			pick_text = (
				f"图上P时间: {self.current_pick_time:.6f} s\n"
				f"波形起始: {self.current_time_origin:.6f} s\n"
				f"震源延迟: {self.source_delay_sec():.6f} s\n"
				f"传播时间: {travel_time:.6f} s"
			)
		distance_m, t_min, t_max = self.current_distance_and_window()
		window_text = "速度窗口: 坐标或速度范围不足，无法计算"
		if distance_m is not None and t_min is not None and t_max is not None:
			window_text = f"距离: {distance_m:.3f} m；图上参考窗口: {t_min:.6f}-{t_max:.6f} s"
			if self.current_pick_time is not None:
				travel_time = self.pick_to_travel_time(self.current_pick_time)
				velocity_mps = distance_m / travel_time if travel_time and travel_time > 0 else float("inf")
				ok_text = "通过" if t_min <= self.current_pick_time <= t_max else "不通过"
				window_text += f"\n当前表观速度: {velocity_mps:.2f} m/s；质控: {ok_text}"
		marked_count = sum(1 for p in self.picks.values() if p.pick_time_sec is not None)
		self.info_var.set(
			f"source={self.current_entry.source_name}\n"
			f"station={self.current_entry.station}\n"
			f"{source_text}\n"
			f"{station_text}\n"
			f"{pick_text}\n"
			f"{window_text}\n"
			f"已标记: {marked_count} / {len(self.entries)}\n"
			f"本次CSV恢复: {self.imported_pick_count}；未匹配: {self.unmatched_import_count}\n"
			f"红线时刻来源: {self.imported_pick_source or '会话记录'}"
		)


def build_default_paths() -> Tuple[Path, Path, Path]:
	waveform_root = resolve_existing_path(
		[PROJECT_ROOT / "正演波形数据集", PROJECT_ROOT / "正演波形"],
		"waveform root",
	)
	station_file = resolve_existing_path(
		[
			PROJECT_ROOT / "反演输入数据集" / "台站（台站编号-台网代码-Y-X-0-Z）.txt",
			PROJECT_ROOT / "台站（台站编号-台网代码-Y-X-0-Z）.txt",
		],
		"station file",
	)
	output_csv = PROJECT_ROOT / "标波输出数据集" / "反演数据集.csv"
	return waveform_root, station_file, output_csv


def main() -> None:
	parser = argparse.ArgumentParser(description="P 波手动标记 UI")
	parser.add_argument("--waveform-root", type=Path, default=None)
	parser.add_argument("--station-file", type=Path, default=None)
	parser.add_argument("--output-csv", type=Path, default=None)
	parser.add_argument("--import-picks-csv", type=Path, default=None)
	parser.add_argument("--source-pattern", type=str, default="source_*")
	parser.add_argument("--waveform-pattern", type=str, default="*.FXZ.semv")
	parser.add_argument("--source-coord-filename", type=str, default="output_list_sources.txt")
	parser.add_argument("--read-only", action="store_true")
	args = parser.parse_args()

	default_waveform_root, default_station_file, default_output_csv = build_default_paths()

	waveform_root = args.waveform_root or default_waveform_root
	station_file = args.station_file or default_station_file
	output_csv = args.output_csv or default_output_csv

	root = tk.Tk()
	try:
		ManualPickerApp(
			root,
			waveform_root,
			station_file,
			output_csv,
			initial_import_csv=args.import_picks_csv,
			source_pattern=args.source_pattern,
			waveform_pattern=args.waveform_pattern,
			source_coord_filename=args.source_coord_filename,
			read_only=args.read_only,
		)
		root.mainloop()
	except Exception as exc:
		messagebox.showerror("启动失败", str(exc))
		raise


if __name__ == "__main__":
	main()
