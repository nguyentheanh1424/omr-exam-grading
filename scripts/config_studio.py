from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2 as cv
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orm_engine.roi_editor import LayoutParams, build_circle_grid

DEFAULT_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"
DEFAULT_ROI_PRESET = REPO_ROOT / "config" / "circle_grid_preset.json"
DEFAULT_ROIS = REPO_ROOT / "config" / "circle_rois.json"
DEFAULT_MARKERS = REPO_ROOT / "config" / "template_marker_layout.json"
DEFAULT_REGION_WINDOWS = REPO_ROOT / "config" / "region_windows.json"
DEFAULT_BUBBLE_FIELDS = REPO_ROOT / "config" / "id_bubble_fields.json"
DEFAULT_HANDWRITTEN = REPO_ROOT / "config" / "handwritten_regions.json"
DEFAULT_OUTPUTS = REPO_ROOT / "config" / "pipeline_outputs.json"


def read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def infer_preset_from_rois(rois: list[dict[str, Any]]) -> LayoutParams:
    if not rois:
        return LayoutParams()

    by_question: dict[int, list[dict[str, Any]]] = {}
    for roi in rois:
        by_question.setdefault(int(roi["question"]), []).append(roi)

    questions = sorted(by_question)
    first_question = sorted(by_question[questions[0]], key=lambda item: int(item["option"]))
    options_per_question = len(first_question)
    circle_diameter = int(first_question[0]["r"]) * 2
    option_gap = 0
    if len(first_question) >= 2:
        option_gap = int(first_question[1]["cx"]) - int(first_question[0]["cx"]) - circle_diameter

    question_starts = []
    for q in questions:
        items = sorted(by_question[q], key=lambda item: int(item["option"]))
        question_starts.append((q, int(items[0]["cx"]), int(items[0]["cy"])))

    grouped: list[list[tuple[int, int, int]]] = []
    for item in sorted(question_starts, key=lambda value: (value[2], value[1])):
        placed = False
        for group in grouped:
            if abs(group[0][1] - item[1]) < circle_diameter * 3:
                group.append(item)
                placed = True
                break
        if not placed:
            grouped.append([item])

    grouped.sort(key=lambda group: min(question for question, _, _ in group))
    num_cols = len(grouped)
    questions_per_col = max(len(group) for group in grouped)
    margin_left = min(x - int(circle_diameter / 2) for _, x, _ in question_starts)
    margin_top = min(y - int(circle_diameter / 2) for _, _, y in question_starts)

    col_dx: list[int] = []
    col_dy: list[int] = []
    column_width = circle_diameter * options_per_question + option_gap * max(0, options_per_question - 1) + 20
    column_gap = 40
    row_height = circle_diameter + 20

    for col_index, group in enumerate(grouped):
        group.sort(key=lambda value: value[0])
        base_x = margin_left + col_index * (column_width + column_gap) + int(circle_diameter / 2)
        base_y = margin_top + int(circle_diameter / 2)
        col_dx.append(group[0][1] - base_x)
        col_dy.append(group[0][2] - base_y)
        if len(group) >= 2:
            row_height = max(row_height, group[1][2] - group[0][2])
        if col_index + 1 < len(grouped):
            next_group = grouped[col_index + 1]
            column_gap = max(column_gap, next_group[0][1] - group[0][1] - column_width)

    return LayoutParams(
        num_questions=len(questions),
        num_cols=max(1, num_cols),
        questions_per_col=max(1, questions_per_col),
        options_per_question=max(1, options_per_question),
        margin_left=max(0, margin_left),
        margin_top=max(0, margin_top),
        column_width=max(20, column_width),
        column_gap=max(0, column_gap),
        row_height=max(8, row_height),
        circle_diameter=max(6, circle_diameter),
        option_gap=max(0, option_gap),
        option_left_padding=0,
        col_dx=col_dx,
        col_dy=col_dy,
    )


class ConfigStudioApp:
    HANDLE_THICK = 12
    OG_HANDLE_HEIGHT = 16
    REGION_HANDLE_SIZE = 16
    MARKER_RADIUS = 16

    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.image_bgr = cv.imread(str(image_path))
        if self.image_bgr is None:
            raise FileNotFoundError(image_path)

        self.img_h, self.img_w = self.image_bgr.shape[:2]
        self.base_rgb = cv.cvtColor(self.image_bgr, cv.COLOR_BGR2RGB)

        self.roi_preset_path = DEFAULT_ROI_PRESET
        self.roi_export_path = DEFAULT_ROIS
        self.marker_path = DEFAULT_MARKERS
        self.region_windows_path = DEFAULT_REGION_WINDOWS
        self.bubble_field_path = DEFAULT_BUBBLE_FIELDS
        self.handwritten_path = DEFAULT_HANDWRITTEN
        self.output_path = DEFAULT_OUTPUTS

        self.selected_col = 0
        self.selected_marker_id: str | None = None
        self.selected_window_index = 0
        self.window_pick_slot = 0
        self.selected_field_index = 0
        self.selected_region_index = 0

        self.drag_mode: str | None = None
        self.drag_start_img_xy = (0, 0)
        self.drag_col = -1
        self.drag_snapshot: dict[str, Any] | None = None

        self.roi_params = self._load_roi_params()
        self.markers: dict[str, list[float]] = {}
        self.region_windows: list[list[int]] = []
        self.bubble_fields: list[dict[str, Any]] = []
        self.handwritten_regions: list[dict[str, Any]] = []
        self.output_config: dict[str, Any] = {}
        self._load_all_configs()

        self.root = tk.Tk()
        self.root.title("Config Studio")
        self.root.geometry("1600x980")

        self.status = tk.StringVar(value="Ready")
        self.active_tab = tk.StringVar(value="roi")
        self.image_var = tk.StringVar(value=str(self.image_path))

        self._build_ui()
        self._refresh_all_panels()
        self._redraw()

    def _load_roi_params(self) -> LayoutParams:
        if self.roi_preset_path.exists():
            data = read_json(self.roi_preset_path, default={})
            params = LayoutParams(**data)
            params.ensure_offsets()
            return params
        rois = read_json(self.roi_export_path, default=[])
        params = infer_preset_from_rois(rois)
        params.ensure_offsets()
        return params

    def _load_all_configs(self) -> None:
        self.markers = {
            str(marker_id): [float(coords[0]), float(coords[1])]
            for marker_id, coords in read_json(self.marker_path, default={}).items()
        }
        self.region_windows = [
            [int(marker_id) for marker_id in window]
            for window in read_json(self.region_windows_path, default=[])
        ]
        self.bubble_fields = read_json(self.bubble_field_path, default=[])
        self.handwritten_regions = read_json(self.handwritten_path, default=[])
        self.output_config = read_json(
            self.output_path,
            default={
                "debug_intermediate": True,
                "summary_json": True,
                "scored_image": True,
                "bubble_fields": {
                    "enabled": True,
                    "overlay_image": True,
                    "values_json": True,
                },
                "handwritten_review": {
                    "enabled": False,
                    "save_patches": True,
                    "save_merged_template": True,
                    "save_ink_mask": True,
                    "save_aligned_source_img": True,
                    "save_aligned_source_regions": True,
                    "save_template_merged_img": True,
                    "save_template_merged_regions": True,
                    "save_scored_img": True,
                },
            },
        )
        if self.markers and self.selected_marker_id is None:
            self.selected_marker_id = sorted(self.markers, key=lambda value: int(value))[0]

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Image").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.image_var, width=90).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Open Image", command=self._choose_image).pack(side=tk.LEFT)
        ttk.Button(top, text="Reload Configs", command=self._reload_configs).pack(side=tk.LEFT, padx=6)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=4)
        body.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.tab_roi = ttk.Frame(self.notebook, padding=8)
        self.tab_markers = ttk.Frame(self.notebook, padding=8)
        self.tab_windows = ttk.Frame(self.notebook, padding=8)
        self.tab_fields = ttk.Frame(self.notebook, padding=8)
        self.tab_regions = ttk.Frame(self.notebook, padding=8)
        self.tab_outputs = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.tab_roi, text="ROI Grid")
        self.notebook.add(self.tab_markers, text="Markers")
        self.notebook.add(self.tab_windows, text="Region Windows")
        self.notebook.add(self.tab_fields, text="Bubble Fields")
        self.notebook.add(self.tab_regions, text="Handwritten")
        self.notebook.add(self.tab_outputs, text="Outputs")

        self._build_roi_tab()
        self._build_markers_tab()
        self._build_windows_tab()
        self._build_fields_tab()
        self._build_regions_tab()
        self._build_outputs_tab()

        ttk.Label(self.root, textvariable=self.status, padding=6).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_roi_tab(self) -> None:
        row = 0
        ttk.Label(self.tab_roi, text="Edit preset and export runtime circle ROIs").grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        self.roi_controls: list[tuple[str, tk.Spinbox]] = []
        for key, label, low, high in [
            ("num_questions", "Questions", 1, 1000),
            ("num_cols", "Columns", 1, 10),
            ("questions_per_col", "Questions/col", 1, 1000),
            ("options_per_question", "Options", 1, 10),
            ("margin_left", "Margin left", 0, 5000),
            ("margin_top", "Margin top", 0, 5000),
            ("column_width", "Column width", 20, 5000),
            ("column_gap", "Column gap", 0, 2000),
            ("row_height", "Row height", 8, 500),
            ("circle_diameter", "Circle diameter", 6, 400),
            ("option_gap", "Option gap", 0, 400),
            ("option_left_padding", "Option left pad", 0, 400),
        ]:
            ttk.Label(self.tab_roi, text=label).grid(row=row, column=0, sticky="w", pady=2)
            spin = tk.Spinbox(self.tab_roi, from_=low, to=high, width=10, command=self._on_roi_spin_change)
            spin.grid(row=row, column=1, sticky="w", pady=2)
            self.roi_controls.append((key, spin))
            row += 1

        ttk.Label(self.tab_roi, text="Selected col").grid(row=row, column=0, sticky="w", pady=2)
        self.sp_selected_col = tk.Spinbox(self.tab_roi, from_=0, to=max(0, self.roi_params.num_cols - 1), width=10, command=self._on_selected_col_change)
        self.sp_selected_col.grid(row=row, column=1, sticky="w", pady=2)
        row += 1

        ttk.Button(self.tab_roi, text="Save Preset", command=self._save_roi_preset).grid(row=row, column=0, sticky="ew", pady=6)
        ttk.Button(self.tab_roi, text="Export ROI JSON", command=self._save_roi_runtime).grid(row=row, column=1, sticky="ew", pady=6)

    def _build_markers_tab(self) -> None:
        self.marker_list = tk.Listbox(self.tab_markers, height=12, exportselection=False)
        self.marker_list.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.marker_list.bind("<<ListboxSelect>>", lambda _e: self._on_marker_select())
        self.tab_markers.rowconfigure(0, weight=1)
        self.tab_markers.columnconfigure(1, weight=1)

        ttk.Label(self.tab_markers, text="Marker ID").grid(row=1, column=0, sticky="w", pady=2)
        self.marker_id_var = tk.StringVar()
        ttk.Entry(self.tab_markers, textvariable=self.marker_id_var).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(self.tab_markers, text="X").grid(row=2, column=0, sticky="w", pady=2)
        self.marker_x_var = tk.StringVar()
        ttk.Entry(self.tab_markers, textvariable=self.marker_x_var).grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(self.tab_markers, text="Y").grid(row=3, column=0, sticky="w", pady=2)
        self.marker_y_var = tk.StringVar()
        ttk.Entry(self.tab_markers, textvariable=self.marker_y_var).grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Button(self.tab_markers, text="Apply", command=self._apply_marker_fields).grid(row=4, column=0, sticky="ew", pady=6)
        ttk.Button(self.tab_markers, text="Add", command=self._add_marker).grid(row=4, column=1, sticky="ew", pady=6)
        ttk.Button(self.tab_markers, text="Delete", command=self._delete_marker).grid(row=5, column=0, sticky="ew", pady=6)
        ttk.Button(self.tab_markers, text="Save JSON", command=self._save_markers).grid(row=5, column=1, sticky="ew", pady=6)

    def _build_windows_tab(self) -> None:
        self.windows_list = tk.Listbox(self.tab_windows, height=10, exportselection=False)
        self.windows_list.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.windows_list.bind("<<ListboxSelect>>", lambda _e: self._on_window_select())
        self.tab_windows.rowconfigure(0, weight=1)
        self.tab_windows.columnconfigure(1, weight=1)

        ttk.Label(self.tab_windows, text="Marker IDs (4, comma-separated)").grid(row=1, column=0, sticky="w", pady=2)
        self.window_ids_var = tk.StringVar()
        ttk.Entry(self.tab_windows, textvariable=self.window_ids_var).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(self.tab_windows, text="Click slot").grid(row=2, column=0, sticky="w", pady=2)
        self.sp_window_slot = tk.Spinbox(self.tab_windows, from_=0, to=3, width=10, command=self._on_window_slot_change)
        self.sp_window_slot.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Button(self.tab_windows, text="Apply", command=self._apply_window_editor).grid(row=3, column=0, sticky="ew", pady=6)
        ttk.Button(self.tab_windows, text="Add", command=self._add_window).grid(row=3, column=1, sticky="ew", pady=6)
        ttk.Button(self.tab_windows, text="Delete", command=self._delete_window).grid(row=4, column=0, sticky="ew", pady=6)
        ttk.Button(self.tab_windows, text="Save JSON", command=self._save_windows).grid(row=4, column=1, sticky="ew", pady=6)

    def _build_fields_tab(self) -> None:
        self.field_list = tk.Listbox(self.tab_fields, height=12, exportselection=False)
        self.field_list.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.field_list.bind("<<ListboxSelect>>", lambda _e: self._on_field_select())
        self.tab_fields.rowconfigure(0, weight=1)
        self.tab_fields.columnconfigure(1, weight=1)

        self.field_vars: dict[str, tk.StringVar] = {}
        specs = [
            ("id", "ID"),
            ("label", "Label"),
            ("origin_x", "Origin X"),
            ("origin_y", "Origin Y"),
            ("dx", "DX"),
            ("dy", "DY"),
            ("n_cols", "Columns"),
            ("n_rows", "Rows"),
            ("radius", "Radius"),
            ("row_values", "Row values"),
            ("abs_th", "Abs th"),
            ("rel_th", "Rel th"),
        ]
        row = 1
        for key, label in specs:
            ttk.Label(self.tab_fields, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar()
            ttk.Entry(self.tab_fields, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
            self.field_vars[key] = var
            row += 1

        ttk.Button(self.tab_fields, text="Apply", command=self._apply_field_editor).grid(row=row, column=0, sticky="ew", pady=6)
        ttk.Button(self.tab_fields, text="Add", command=self._add_field).grid(row=row, column=1, sticky="ew", pady=6)
        row += 1
        ttk.Button(self.tab_fields, text="Delete", command=self._delete_field).grid(row=row, column=0, sticky="ew", pady=6)
        ttk.Button(self.tab_fields, text="Save JSON", command=self._save_bubble_fields).grid(row=row, column=1, sticky="ew", pady=6)

    def _build_regions_tab(self) -> None:
        self.region_list = tk.Listbox(self.tab_regions, height=12, exportselection=False)
        self.region_list.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.region_list.bind("<<ListboxSelect>>", lambda _e: self._on_region_select())
        self.tab_regions.rowconfigure(0, weight=1)
        self.tab_regions.columnconfigure(1, weight=1)

        self.region_vars: dict[str, tk.StringVar] = {}
        specs = [
            ("id", "ID"),
            ("label", "Label"),
            ("x0", "X0"),
            ("y0", "Y0"),
            ("x1", "X1"),
            ("y1", "Y1"),
            ("padding_px", "Padding"),
            ("merge_mode", "Merge mode"),
            ("save_patch", "Save patch"),
        ]
        row = 1
        for key, label in specs:
            ttk.Label(self.tab_regions, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar()
            ttk.Entry(self.tab_regions, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
            self.region_vars[key] = var
            row += 1

        ttk.Button(self.tab_regions, text="Apply", command=self._apply_region_editor).grid(row=row, column=0, sticky="ew", pady=6)
        ttk.Button(self.tab_regions, text="Add", command=self._add_region).grid(row=row, column=1, sticky="ew", pady=6)
        row += 1
        ttk.Button(self.tab_regions, text="Delete", command=self._delete_region).grid(row=row, column=0, sticky="ew", pady=6)
        ttk.Button(self.tab_regions, text="Save JSON", command=self._save_regions).grid(row=row, column=1, sticky="ew", pady=6)

    def _build_outputs_tab(self) -> None:
        self.output_vars: dict[str, tk.BooleanVar] = {}
        checks = [
            ("debug_intermediate", "Debug intermediate"),
            ("summary_json", "Summary JSON"),
            ("scored_image", "Scored image"),
            ("bubble_fields.enabled", "Bubble fields enabled"),
            ("bubble_fields.overlay_image", "Bubble overlay image"),
            ("bubble_fields.values_json", "Bubble values JSON"),
            ("handwritten_review.enabled", "Handwritten enabled"),
            ("handwritten_review.save_patches", "Save handwritten patches"),
            ("handwritten_review.save_merged_template", "Save merged template"),
            ("handwritten_review.save_ink_mask", "Save ink mask"),
            ("handwritten_review.save_aligned_source_img", "Save aligned source image"),
            ("handwritten_review.save_aligned_source_regions", "Save aligned source regions"),
            ("handwritten_review.save_template_merged_img", "Save template merged image"),
            ("handwritten_review.save_template_merged_regions", "Save template merged regions"),
            ("handwritten_review.save_scored_img", "Save scored image in review flow"),
        ]
        for row, (key, label) in enumerate(checks):
            var = tk.BooleanVar(value=False)
            self.output_vars[key] = var
            ttk.Checkbutton(self.tab_outputs, text=label, variable=var).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Button(self.tab_outputs, text="Save JSON", command=self._save_outputs).grid(row=len(checks), column=0, sticky="ew", pady=10)

    def _refresh_all_panels(self) -> None:
        self._sync_roi_controls_from_params()
        self._refresh_marker_list()
        self._refresh_windows_list()
        self._refresh_field_list()
        self._refresh_region_list()
        self._refresh_outputs()

    def _sync_roi_controls_from_params(self) -> None:
        self.roi_params.ensure_offsets()
        for key, spin in self.roi_controls:
            spin.delete(0, "end")
            spin.insert(0, str(getattr(self.roi_params, key)))
        self.sp_selected_col.config(to=max(0, self.roi_params.num_cols - 1))
        self.sp_selected_col.delete(0, "end")
        self.sp_selected_col.insert(0, str(self.selected_col))

    def _refresh_marker_list(self) -> None:
        self.marker_list.delete(0, tk.END)
        marker_ids = sorted(self.markers, key=lambda value: int(value))
        for marker_id in marker_ids:
            coords = self.markers[marker_id]
            self.marker_list.insert(tk.END, f"{marker_id}: ({int(coords[0])}, {int(coords[1])})")
        if marker_ids:
            if self.selected_marker_id not in self.markers:
                self.selected_marker_id = marker_ids[0]
            idx = marker_ids.index(self.selected_marker_id)
            self.marker_list.selection_clear(0, tk.END)
            self.marker_list.selection_set(idx)
            self.marker_list.activate(idx)
            self._fill_marker_editor(self.selected_marker_id)
        else:
            self.selected_marker_id = None
            self.marker_id_var.set("")
            self.marker_x_var.set("")
            self.marker_y_var.set("")

    def _refresh_field_list(self) -> None:
        self.field_list.delete(0, tk.END)
        for index, field in enumerate(self.bubble_fields):
            self.field_list.insert(tk.END, f"{index}: {field.get('id', 'field')} ({field.get('label', '')})")
        if self.bubble_fields:
            self.selected_field_index = clamp(self.selected_field_index, 0, len(self.bubble_fields) - 1)
            self.field_list.selection_clear(0, tk.END)
            self.field_list.selection_set(self.selected_field_index)
            self.field_list.activate(self.selected_field_index)
            self._fill_field_editor()

    def _refresh_windows_list(self) -> None:
        self.windows_list.delete(0, tk.END)
        for index, window in enumerate(self.region_windows):
            self.windows_list.insert(tk.END, f"{index}: {window}")
        if self.region_windows:
            self.selected_window_index = clamp(self.selected_window_index, 0, len(self.region_windows) - 1)
            self.windows_list.selection_clear(0, tk.END)
            self.windows_list.selection_set(self.selected_window_index)
            self.windows_list.activate(self.selected_window_index)
            self._fill_window_editor()
        else:
            self.window_ids_var.set("")
        self.sp_window_slot.delete(0, "end")
        self.sp_window_slot.insert(0, str(self.window_pick_slot))

    def _refresh_region_list(self) -> None:
        self.region_list.delete(0, tk.END)
        for index, region in enumerate(self.handwritten_regions):
            self.region_list.insert(tk.END, f"{index}: {region.get('id', 'region')} ({region.get('label', '')})")
        if self.handwritten_regions:
            self.selected_region_index = clamp(self.selected_region_index, 0, len(self.handwritten_regions) - 1)
            self.region_list.selection_clear(0, tk.END)
            self.region_list.selection_set(self.selected_region_index)
            self.region_list.activate(self.selected_region_index)
            self._fill_region_editor()

    def _refresh_outputs(self) -> None:
        for key, var in self.output_vars.items():
            var.set(bool(self._get_nested_output(key)))

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open background image",
            initialdir=str(self.image_path.parent),
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        self._set_image(Path(path))

    def _set_image(self, path: Path) -> None:
        image = cv.imread(str(path))
        if image is None:
            messagebox.showerror("Error", f"Cannot read image: {path}")
            return
        self.image_path = path
        self.image_var.set(str(path))
        self.image_bgr = image
        self.img_h, self.img_w = image.shape[:2]
        self.base_rgb = cv.cvtColor(self.image_bgr, cv.COLOR_BGR2RGB)
        self.status.set(f"Loaded image: {path.name}")
        self._redraw()

    def _reload_configs(self) -> None:
        self.roi_params = self._load_roi_params()
        self._load_all_configs()
        self.selected_col = clamp(self.selected_col, 0, max(0, self.roi_params.num_cols - 1))
        self.selected_window_index = 0
        self.selected_field_index = 0
        self.selected_region_index = 0
        self._refresh_all_panels()
        self.status.set("Reloaded configs from disk")
        self._redraw()

    def _on_tab_changed(self, _event) -> None:
        current_text = self.notebook.tab(self.notebook.select(), "text")
        mapping = {
            "ROI Grid": "roi",
            "Markers": "markers",
            "Region Windows": "windows",
            "Bubble Fields": "fields",
            "Handwritten": "regions",
            "Outputs": "outputs",
        }
        self.active_tab.set(mapping.get(current_text, "roi"))
        self._redraw()

    def _canvas_size(self) -> tuple[int, int]:
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _canvas_to_img(self, x: int, y: int) -> tuple[int, int]:
        cw, ch = self._canvas_size()
        return int(x * self.img_w / cw), int(y * self.img_h / ch)

    def _on_roi_spin_change(self) -> None:
        try:
            for key, spin in self.roi_controls:
                setattr(self.roi_params, key, int(spin.get()))
            self.roi_params.num_cols = max(1, int(self.roi_params.num_cols))
            self.roi_params.ensure_offsets()
            current_dx = list(self.roi_params.col_dx or [])
            current_dy = list(self.roi_params.col_dy or [])
            self.roi_params.col_dx = current_dx[: self.roi_params.num_cols] + [0] * max(0, self.roi_params.num_cols - len(current_dx))
            self.roi_params.col_dy = current_dy[: self.roi_params.num_cols] + [0] * max(0, self.roi_params.num_cols - len(current_dy))
            self.selected_col = clamp(self.selected_col, 0, self.roi_params.num_cols - 1)
            self._sync_roi_controls_from_params()
            self.status.set("Updated ROI preset parameters")
            self._redraw()
        except ValueError:
            pass

    def _on_selected_col_change(self) -> None:
        try:
            self.selected_col = clamp(int(self.sp_selected_col.get()), 0, max(0, self.roi_params.num_cols - 1))
            self._redraw()
        except ValueError:
            pass

    def _fill_marker_editor(self, marker_id: str) -> None:
        coords = self.markers[marker_id]
        self.marker_id_var.set(marker_id)
        self.marker_x_var.set(str(int(coords[0])))
        self.marker_y_var.set(str(int(coords[1])))

    def _on_marker_select(self) -> None:
        selection = self.marker_list.curselection()
        if not selection:
            return
        marker_ids = sorted(self.markers, key=lambda value: int(value))
        self.selected_marker_id = marker_ids[selection[0]]
        self._fill_marker_editor(self.selected_marker_id)
        self._redraw()

    def _apply_marker_fields(self) -> None:
        marker_id = self.marker_id_var.get().strip()
        if not marker_id:
            messagebox.showerror("Error", "Marker ID is required")
            return
        try:
            x = clamp(int(float(self.marker_x_var.get())), 0, self.img_w - 1)
            y = clamp(int(float(self.marker_y_var.get())), 0, self.img_h - 1)
            int(marker_id)
        except ValueError:
            messagebox.showerror("Error", "Marker values must be valid integers")
            return
        if self.selected_marker_id and self.selected_marker_id != marker_id:
            self.markers.pop(self.selected_marker_id, None)
        self.markers[marker_id] = [float(x), float(y)]
        self.selected_marker_id = marker_id
        self._refresh_marker_list()
        self.status.set(f"Updated marker {marker_id}")
        self._redraw()

    def _add_marker(self) -> None:
        next_id = 1
        if self.markers:
            next_id = max(int(value) for value in self.markers) + 1
        self.markers[str(next_id)] = [float(self.img_w // 2), float(self.img_h // 2)]
        self.selected_marker_id = str(next_id)
        self._refresh_marker_list()
        self.status.set(f"Added marker {next_id}")
        self._redraw()

    def _delete_marker(self) -> None:
        if not self.selected_marker_id:
            return
        self.markers.pop(self.selected_marker_id, None)
        self.selected_marker_id = None
        self._refresh_marker_list()
        self.status.set("Deleted marker")
        self._redraw()

    def _save_markers(self) -> None:
        payload = {
            marker_id: [float(coords[0]), float(coords[1])]
            for marker_id, coords in sorted(self.markers.items(), key=lambda item: int(item[0]))
        }
        write_json(self.marker_path, payload)
        self.status.set(f"Saved markers -> {self.marker_path.name}")

    def _on_window_select(self) -> None:
        selection = self.windows_list.curselection()
        if not selection:
            return
        self.selected_window_index = selection[0]
        self._fill_window_editor()
        self._redraw()

    def _on_window_slot_change(self) -> None:
        try:
            self.window_pick_slot = clamp(int(self.sp_window_slot.get()), 0, 3)
        except ValueError:
            self.window_pick_slot = 0
        self.sp_window_slot.delete(0, "end")
        self.sp_window_slot.insert(0, str(self.window_pick_slot))
        self._redraw()

    def _fill_window_editor(self) -> None:
        if not self.region_windows:
            self.window_ids_var.set("")
            return
        self.window_ids_var.set(",".join(str(marker_id) for marker_id in self.region_windows[self.selected_window_index]))

    def _apply_window_editor(self) -> None:
        if not self.region_windows:
            return
        try:
            marker_ids = [int(value.strip()) for value in self.window_ids_var.get().split(",") if value.strip()]
        except ValueError as exc:
            messagebox.showerror("Error", f"Invalid window marker ids: {exc}")
            return
        if len(marker_ids) != 4 or len(set(marker_ids)) != 4:
            messagebox.showerror("Error", "Each window must contain exactly 4 unique marker ids")
            return
        self.region_windows[self.selected_window_index] = marker_ids
        self._refresh_windows_list()
        self.status.set(f"Updated region window {self.selected_window_index}")
        self._redraw()

    def _add_window(self) -> None:
        next_window = [1, 2, 3, 4]
        if self.markers and len(self.markers) >= 4:
            marker_ids = sorted(int(marker_id) for marker_id in self.markers)[:4]
            next_window = marker_ids
        self.region_windows.append(list(next_window))
        self.selected_window_index = len(self.region_windows) - 1
        self._refresh_windows_list()
        self.status.set("Added region window")
        self._redraw()

    def _delete_window(self) -> None:
        if not self.region_windows:
            return
        self.region_windows.pop(self.selected_window_index)
        self.selected_window_index = clamp(self.selected_window_index, 0, max(0, len(self.region_windows) - 1))
        self._refresh_windows_list()
        self.status.set("Deleted region window")
        self._redraw()

    def _save_windows(self) -> None:
        write_json(self.region_windows_path, self.region_windows)
        self.status.set(f"Saved region windows -> {self.region_windows_path.name}")

    def _find_marker_at(self, ix: int, iy: int) -> str | None:
        nearest_id = None
        nearest_dist = None
        for marker_id, coords in self.markers.items():
            dist = (coords[0] - ix) ** 2 + (coords[1] - iy) ** 2
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_id = marker_id
        if nearest_id is None or nearest_dist is None:
            return None
        if nearest_dist > (self.MARKER_RADIUS * 2) ** 2:
            return None
        return nearest_id

    def _on_field_select(self) -> None:
        selection = self.field_list.curselection()
        if not selection:
            return
        self.selected_field_index = selection[0]
        self._fill_field_editor()
        self._redraw()

    def _fill_field_editor(self) -> None:
        if not self.bubble_fields:
            return
        field = self.bubble_fields[self.selected_field_index]
        self.field_vars["id"].set(str(field.get("id", "")))
        self.field_vars["label"].set(str(field.get("label", "")))
        origin = field.get("origin", [0, 0])
        self.field_vars["origin_x"].set(str(origin[0]))
        self.field_vars["origin_y"].set(str(origin[1]))
        for key in ["dx", "dy", "n_cols", "n_rows", "radius"]:
            self.field_vars[key].set(str(field.get(key, "")))
        self.field_vars["row_values"].set(",".join(str(value) for value in field.get("row_values", [])))
        self.field_vars["abs_th"].set("" if "abs_th" not in field else str(field.get("abs_th")))
        self.field_vars["rel_th"].set("" if "rel_th" not in field else str(field.get("rel_th")))

    def _apply_field_editor(self) -> None:
        if not self.bubble_fields:
            return
        try:
            row_values = [value.strip() for value in self.field_vars["row_values"].get().split(",") if value.strip()]
            entry = {
                "id": self.field_vars["id"].get().strip(),
                "label": self.field_vars["label"].get().strip(),
                "origin": [int(self.field_vars["origin_x"].get()), int(self.field_vars["origin_y"].get())],
                "dx": int(self.field_vars["dx"].get()),
                "dy": int(self.field_vars["dy"].get()),
                "n_cols": int(self.field_vars["n_cols"].get()),
                "n_rows": int(self.field_vars["n_rows"].get()),
                "radius": int(self.field_vars["radius"].get()),
                "row_values": row_values,
            }
            abs_th_raw = self.field_vars["abs_th"].get().strip()
            rel_th_raw = self.field_vars["rel_th"].get().strip()
            if abs_th_raw:
                entry["abs_th"] = float(abs_th_raw)
            if rel_th_raw:
                entry["rel_th"] = float(rel_th_raw)
            self.bubble_fields[self.selected_field_index] = entry
        except ValueError as exc:
            messagebox.showerror("Error", f"Invalid field values: {exc}")
            return
        self._refresh_field_list()
        self.status.set(f"Updated field {entry['id']}")
        self._redraw()

    def _add_field(self) -> None:
        self.bubble_fields.append(
            {
                "id": f"field_{len(self.bubble_fields) + 1}",
                "label": "New Field",
                "origin": [100, 100],
                "dx": 40,
                "dy": 60,
                "n_cols": 5,
                "n_rows": 8,
                "radius": 15,
                "row_values": [str(i) for i in range(8)],
            }
        )
        self.selected_field_index = len(self.bubble_fields) - 1
        self._refresh_field_list()
        self.status.set("Added bubble field")
        self._redraw()

    def _delete_field(self) -> None:
        if not self.bubble_fields:
            return
        self.bubble_fields.pop(self.selected_field_index)
        self.selected_field_index = clamp(self.selected_field_index, 0, max(0, len(self.bubble_fields) - 1))
        self._refresh_field_list()
        self.status.set("Deleted bubble field")
        self._redraw()

    def _save_bubble_fields(self) -> None:
        write_json(self.bubble_field_path, self.bubble_fields)
        self.status.set(f"Saved bubble fields -> {self.bubble_field_path.name}")

    def _on_region_select(self) -> None:
        selection = self.region_list.curselection()
        if not selection:
            return
        self.selected_region_index = selection[0]
        self._fill_region_editor()
        self._redraw()

    def _fill_region_editor(self) -> None:
        if not self.handwritten_regions:
            return
        region = self.handwritten_regions[self.selected_region_index]
        self.region_vars["id"].set(str(region.get("id", "")))
        self.region_vars["label"].set(str(region.get("label", "")))
        rect = region.get("rect", [0, 0, 100, 100])
        self.region_vars["x0"].set(str(rect[0]))
        self.region_vars["y0"].set(str(rect[1]))
        self.region_vars["x1"].set(str(rect[2]))
        self.region_vars["y1"].set(str(rect[3]))
        self.region_vars["padding_px"].set(str(region.get("padding_px", 0)))
        self.region_vars["merge_mode"].set(str(region.get("merge_mode", "replace_rect")))
        self.region_vars["save_patch"].set("true" if region.get("save_patch", True) else "false")

    def _apply_region_editor(self) -> None:
        if not self.handwritten_regions:
            return
        try:
            rect = [
                int(self.region_vars["x0"].get()),
                int(self.region_vars["y0"].get()),
                int(self.region_vars["x1"].get()),
                int(self.region_vars["y1"].get()),
            ]
            entry = {
                "id": self.region_vars["id"].get().strip(),
                "label": self.region_vars["label"].get().strip(),
                "rect": rect,
                "padding_px": int(self.region_vars["padding_px"].get()),
                "merge_mode": self.region_vars["merge_mode"].get().strip() or "replace_rect",
                "save_patch": self.region_vars["save_patch"].get().strip().lower() not in {"false", "0", "no"},
            }
            self.handwritten_regions[self.selected_region_index] = entry
        except ValueError as exc:
            messagebox.showerror("Error", f"Invalid region values: {exc}")
            return
        self._refresh_region_list()
        self.status.set(f"Updated region {entry['id']}")
        self._redraw()

    def _add_region(self) -> None:
        self.handwritten_regions.append(
            {
                "id": f"region_{len(self.handwritten_regions) + 1}",
                "label": "New Region",
                "rect": [100, 100, 300, 180],
                "padding_px": 10,
                "merge_mode": "replace_rect",
                "save_patch": True,
            }
        )
        self.selected_region_index = len(self.handwritten_regions) - 1
        self._refresh_region_list()
        self.status.set("Added handwritten region")
        self._redraw()

    def _delete_region(self) -> None:
        if not self.handwritten_regions:
            return
        self.handwritten_regions.pop(self.selected_region_index)
        self.selected_region_index = clamp(self.selected_region_index, 0, max(0, len(self.handwritten_regions) - 1))
        self._refresh_region_list()
        self.status.set("Deleted handwritten region")
        self._redraw()

    def _save_regions(self) -> None:
        write_json(self.handwritten_path, self.handwritten_regions)
        self.status.set(f"Saved regions -> {self.handwritten_path.name}")

    def _get_nested_output(self, key: str) -> Any:
        node: Any = self.output_config
        for part in key.split("."):
            if not isinstance(node, dict):
                return False
            node = node.get(part, False)
        return node

    def _set_nested_output(self, key: str, value: bool) -> None:
        node = self.output_config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = bool(value)

    def _save_outputs(self) -> None:
        for key, var in self.output_vars.items():
            self._set_nested_output(key, var.get())
        write_json(self.output_path, self.output_config)
        self.status.set(f"Saved outputs -> {self.output_path.name}")

    def _save_roi_preset(self) -> None:
        self._on_roi_spin_change()
        write_json(self.roi_preset_path, asdict(self.roi_params))
        self.status.set(f"Saved ROI preset -> {self.roi_preset_path.name}")

    def _save_roi_runtime(self) -> None:
        self._on_roi_spin_change()
        rois = build_circle_grid(self.img_w, self.img_h, self.roi_params)
        write_json(self.roi_export_path, [roi.to_dict() for roi in rois])
        self.status.set(f"Exported {len(rois)} circle ROIs -> {self.roi_export_path.name}")

    def _active_tab_name(self) -> str:
        return self.active_tab.get()

    def _on_mouse_down(self, event) -> None:
        ix, iy = self._canvas_to_img(event.x, event.y)
        self.drag_start_img_xy = (ix, iy)
        self.drag_snapshot = {
            "roi": asdict(self.roi_params),
            "markers": json.loads(json.dumps(self.markers)),
            "fields": json.loads(json.dumps(self.bubble_fields)),
            "regions": json.loads(json.dumps(self.handwritten_regions)),
        }

        active = self._active_tab_name()
        if active == "roi":
            self._start_roi_drag(ix, iy)
        elif active == "markers":
            self._start_marker_drag(ix, iy)
        elif active == "windows":
            self._handle_window_click(ix, iy)
        elif active == "fields":
            self._start_field_drag(ix, iy)
        elif active == "regions":
            self._start_region_drag(ix, iy)
        else:
            self.drag_mode = None

    def _start_roi_drag(self, ix: int, iy: int) -> None:
        self.drag_mode = None
        self.drag_col = -1
        for col in range(self.roi_params.num_cols):
            x0, y0, x1, y1 = self._roi_column_rect(col)
            if x0 <= ix <= x1 and y0 - self.OG_HANDLE_HEIGHT <= iy <= y1:
                self.drag_col = col
                self.selected_col = col
                if y0 - self.OG_HANDLE_HEIGHT <= iy <= y0:
                    self.drag_mode = "roi_option_gap"
                elif abs(ix - x1) <= self.HANDLE_THICK:
                    self.drag_mode = "roi_col_width"
                elif abs(iy - y1) <= self.HANDLE_THICK:
                    self.drag_mode = "roi_row_height"
                else:
                    self.drag_mode = "roi_move_col"
                self._sync_roi_controls_from_params()
                self.status.set(f"ROI drag: col {col}, mode={self.drag_mode}")
                self._redraw()
                return

    def _start_marker_drag(self, ix: int, iy: int) -> None:
        self.drag_mode = None
        nearest_id = self._find_marker_at(ix, iy)
        if nearest_id is not None:
            self.selected_marker_id = nearest_id
            self.drag_mode = "marker_move"
            self._refresh_marker_list()
            self.status.set(f"Moving marker {nearest_id}")
            self._redraw()

    def _handle_window_click(self, ix: int, iy: int) -> None:
        self.drag_mode = None
        if not self.region_windows:
            return
        marker_id = self._find_marker_at(ix, iy)
        if marker_id is None:
            return
        window = list(self.region_windows[self.selected_window_index])
        window[self.window_pick_slot] = int(marker_id)
        self.region_windows[self.selected_window_index] = window
        self.window_pick_slot = (self.window_pick_slot + 1) % 4
        self._refresh_windows_list()
        self.selected_marker_id = str(marker_id)
        self.status.set(
            f"Assigned marker {marker_id} to window {self.selected_window_index}, slot {(self.window_pick_slot - 1) % 4}"
        )
        self._redraw()

    def _start_field_drag(self, ix: int, iy: int) -> None:
        self.drag_mode = None
        for index, field in enumerate(self.bubble_fields):
            x0, y0, x1, y1 = self._bubble_field_bounds(field)
            if x0 <= ix <= x1 and y0 <= iy <= y1:
                self.selected_field_index = index
                self.drag_mode = "field_move"
                self._refresh_field_list()
                self.status.set(f"Moving bubble field {field.get('id', index)}")
                self._redraw()
                return

    def _start_region_drag(self, ix: int, iy: int) -> None:
        self.drag_mode = None
        for index, region in enumerate(self.handwritten_regions):
            x0, y0, x1, y1 = [int(value) for value in region.get("rect", [0, 0, 0, 0])]
            if x0 <= ix <= x1 and y0 <= iy <= y1:
                self.selected_region_index = index
                if abs(ix - x1) <= self.REGION_HANDLE_SIZE and abs(iy - y1) <= self.REGION_HANDLE_SIZE:
                    self.drag_mode = "region_resize"
                else:
                    self.drag_mode = "region_move"
                self._refresh_region_list()
                self.status.set(f"Editing region {region.get('id', index)}")
                self._redraw()
                return

    def _on_mouse_drag(self, event) -> None:
        if not self.drag_mode or self.drag_snapshot is None:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        dx = ix - self.drag_start_img_xy[0]
        dy = iy - self.drag_start_img_xy[1]

        if self.drag_mode.startswith("roi_"):
            self._perform_roi_drag(dx, dy)
        elif self.drag_mode == "marker_move":
            self._perform_marker_drag(dx, dy)
        elif self.drag_mode == "field_move":
            self._perform_field_drag(dx, dy)
        elif self.drag_mode in {"region_move", "region_resize"}:
            self._perform_region_drag(dx, dy)
        self._redraw()

    def _perform_roi_drag(self, dx: int, dy: int) -> None:
        snapshot = LayoutParams(**self.drag_snapshot["roi"])
        snapshot.ensure_offsets()
        self.roi_params = snapshot
        col = self.drag_col
        if col < 0:
            return
        if self.drag_mode == "roi_move_col":
            self.roi_params.col_dx[col] += int(dx)
            self.roi_params.col_dy[col] += int(dy)
        elif self.drag_mode == "roi_col_width":
            self.roi_params.column_width = max(20, self.roi_params.column_width + int(dx))
        elif self.drag_mode == "roi_row_height":
            delta = int(dy / max(1, self.roi_params.questions_per_col))
            self.roi_params.row_height = max(8, self.roi_params.row_height + delta)
        elif self.drag_mode == "roi_option_gap":
            delta = int(dx / max(1, self.roi_params.options_per_question))
            self.roi_params.option_gap = max(0, self.roi_params.option_gap + delta)
        self._sync_roi_controls_from_params()

    def _perform_marker_drag(self, dx: int, dy: int) -> None:
        if not self.selected_marker_id:
            return
        snapshot_markers = self.drag_snapshot["markers"]
        coords = snapshot_markers[self.selected_marker_id]
        self.markers[self.selected_marker_id] = [
            float(clamp(int(coords[0] + dx), 0, self.img_w - 1)),
            float(clamp(int(coords[1] + dy), 0, self.img_h - 1)),
        ]
        self._fill_marker_editor(self.selected_marker_id)
        self._refresh_marker_list()

    def _perform_field_drag(self, dx: int, dy: int) -> None:
        if not self.bubble_fields:
            return
        snapshot_fields = self.drag_snapshot["fields"]
        field = snapshot_fields[self.selected_field_index]
        origin = field.get("origin", [0, 0])
        new_x = clamp(int(origin[0] + dx), 0, self.img_w - 1)
        new_y = clamp(int(origin[1] + dy), 0, self.img_h - 1)
        self.bubble_fields[self.selected_field_index]["origin"] = [new_x, new_y]
        self._fill_field_editor()
        self._refresh_field_list()

    def _perform_region_drag(self, dx: int, dy: int) -> None:
        if not self.handwritten_regions:
            return
        snapshot_regions = self.drag_snapshot["regions"]
        region = snapshot_regions[self.selected_region_index]
        x0, y0, x1, y1 = [int(value) for value in region.get("rect", [0, 0, 0, 0])]
        if self.drag_mode == "region_move":
            width = x1 - x0
            height = y1 - y0
            x0 = clamp(x0 + dx, 0, self.img_w - width)
            y0 = clamp(y0 + dy, 0, self.img_h - height)
            x1 = x0 + width
            y1 = y0 + height
        else:
            x1 = clamp(x1 + dx, x0 + 10, self.img_w)
            y1 = clamp(y1 + dy, y0 + 10, self.img_h)
        self.handwritten_regions[self.selected_region_index]["rect"] = [x0, y0, x1, y1]
        self._fill_region_editor()
        self._refresh_region_list()

    def _on_mouse_up(self, _event) -> None:
        if self.drag_mode:
            self.status.set("Ready")
        self.drag_mode = None
        self.drag_col = -1
        self.drag_snapshot = None

    def _roi_column_rect(self, col: int) -> tuple[int, int, int, int]:
        self.roi_params.ensure_offsets()
        x0 = self.roi_params.margin_left + col * (self.roi_params.column_width + self.roi_params.column_gap) + self.roi_params.col_dx[col]
        y0 = self.roi_params.margin_top + self.roi_params.col_dy[col]
        x1 = x0 + self.roi_params.column_width
        y1 = y0 + self.roi_params.questions_per_col * self.roi_params.row_height
        return x0, y0, x1, y1

    def _bubble_field_bounds(self, field: dict[str, Any]) -> tuple[int, int, int, int]:
        origin = field.get("origin", [0, 0])
        radius = int(field.get("radius", 0))
        x0 = int(origin[0] - radius)
        y0 = int(origin[1] - radius)
        x1 = int(origin[0] + (int(field.get("n_cols", 1)) - 1) * int(field.get("dx", 0)) + radius)
        y1 = int(origin[1] + (int(field.get("n_rows", 1)) - 1) * int(field.get("dy", 0)) + radius)
        return x0, y0, x1, y1

    def _compose_overlay(self):
        vis = self.base_rgb.copy()
        active = self._active_tab_name()
        if active == "roi":
            self._draw_roi_overlay(vis)
        elif active == "markers":
            self._draw_marker_overlay(vis)
        elif active == "windows":
            self._draw_windows_overlay(vis)
        elif active == "fields":
            self._draw_field_overlay(vis)
        elif active == "regions":
            self._draw_region_overlay(vis)
        else:
            self._draw_outputs_overlay(vis)
        return vis

    def _draw_roi_overlay(self, vis) -> None:
        circles = build_circle_grid(self.img_w, self.img_h, self.roi_params)
        for col in range(self.roi_params.num_cols):
            x0, y0, x1, y1 = self._roi_column_rect(col)
            color = (255, 255, 0) if col == self.selected_col else (0, 255, 0)
            cv.rectangle(vis, (x0, y0), (x1, y1), color, 2 if col == self.selected_col else 1)
            cv.rectangle(vis, (x0, max(0, y0 - self.OG_HANDLE_HEIGHT)), (x1, y0), (0, 128, 255), -1)
            cv.line(vis, (x1, y0), (x1, y1), (0, 165, 255), 3)
            cv.line(vis, (x0, y1), (x1, y1), (0, 165, 255), 3)
            cv.putText(vis, f"col {col}", (x0 + 8, max(24, y0 - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for roi in circles:
            cv.circle(vis, (roi.cx, roi.cy), roi.r, (0, 255, 0), 2)

    def _draw_marker_overlay(self, vis) -> None:
        for marker_id, coords in sorted(self.markers.items(), key=lambda item: int(item[0])):
            x, y = int(coords[0]), int(coords[1])
            color = (255, 255, 0) if marker_id == self.selected_marker_id else (255, 0, 0)
            cv.circle(vis, (x, y), self.MARKER_RADIUS, color, 3)
            cv.putText(vis, str(marker_id), (x + 12, y - 12), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_windows_overlay(self, vis) -> None:
        self._draw_marker_overlay(vis)
        for index, window in enumerate(self.region_windows):
            points = []
            for slot_index, marker_id in enumerate(window):
                coords = self.markers.get(str(marker_id))
                if coords is None:
                    continue
                point = (int(coords[0]), int(coords[1]))
                points.append(point)
                slot_color = (255, 255, 0) if index == self.selected_window_index and slot_index == self.window_pick_slot else (255, 255, 255)
                cv.putText(
                    vis,
                    str(slot_index),
                    (point[0] - 6, point[1] + 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    slot_color,
                    2,
                )
            if len(points) < 2:
                continue
            color = (0, 255, 0) if index == self.selected_window_index else (0, 165, 255)
            for point_index in range(len(points)):
                cv.line(vis, points[point_index], points[(point_index + 1) % len(points)], color, 2)
            label_pos = points[0]
            cv.putText(
                vis,
                f"window {index} slot {self.window_pick_slot}" if index == self.selected_window_index else f"window {index}",
                (label_pos[0] + 8, label_pos[1] + 24),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    def _draw_field_overlay(self, vis) -> None:
        for index, field in enumerate(self.bubble_fields):
            selected = index == self.selected_field_index
            color = (0, 255, 0) if selected else (255, 180, 0)
            x0, y0, x1, y1 = self._bubble_field_bounds(field)
            cv.rectangle(vis, (x0, y0), (x1, y1), color, 2 if selected else 1)
            origin = field.get("origin", [0, 0])
            for col in range(int(field.get("n_cols", 0))):
                for row in range(int(field.get("n_rows", 0))):
                    cx = int(origin[0]) + col * int(field.get("dx", 0))
                    cy = int(origin[1]) + row * int(field.get("dy", 0))
                    cv.circle(vis, (cx, cy), int(field.get("radius", 0)), color, 2 if selected else 1)
            label = f"{field.get('label', field.get('id', index))}"
            cv.putText(vis, label, (x0 + 6, max(24, y0 - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_region_overlay(self, vis) -> None:
        for index, region in enumerate(self.handwritten_regions):
            selected = index == self.selected_region_index
            color = (0, 255, 0) if selected else (0, 140, 255)
            x0, y0, x1, y1 = [int(value) for value in region.get("rect", [0, 0, 0, 0])]
            cv.rectangle(vis, (x0, y0), (x1, y1), color, 2 if selected else 1)
            cv.rectangle(vis, (x1 - self.REGION_HANDLE_SIZE, y1 - self.REGION_HANDLE_SIZE), (x1, y1), color, -1)
            label = f"{region.get('label', region.get('id', index))} [{region.get('merge_mode', 'replace_rect')}]"
            cv.putText(vis, label, (x0 + 6, max(24, y0 - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_outputs_overlay(self, vis) -> None:
        lines = [
            "Outputs config is edited from the right panel.",
            f"debug_intermediate={self.output_config.get('debug_intermediate', False)}",
            f"summary_json={self.output_config.get('summary_json', False)}",
            f"scored_image={self.output_config.get('scored_image', False)}",
            f"bubble_fields.enabled={self._get_nested_output('bubble_fields.enabled')}",
            f"handwritten_review.enabled={self._get_nested_output('handwritten_review.enabled')}",
        ]
        for idx, text in enumerate(lines):
            y = 60 + idx * 34
            cv.putText(vis, text, (40, y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    def _redraw(self) -> None:
        vis = self._compose_overlay()
        cw, ch = self._canvas_size()
        disp = cv.resize(vis, (cw, ch), interpolation=cv.INTER_LINEAR)
        image = Image.fromarray(disp)
        self._photo = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Config Studio for OMR repo configs")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Background image for visual editing")
    args = parser.parse_args()
    app = ConfigStudioApp(Path(args.image))
    app.run()


if __name__ == "__main__":
    main()
