# path: roi_grid_editor_circle.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ====== Data structures ======
@dataclass
class CircleROI:
    cx: int
    cy: int
    r: int
    question: int
    option: int

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "r": self.r, "question": self.question, "option": self.option}

    def to_rect_tuple(self) -> Tuple[int, int, int, int, int, int]:
        # compat: (yT, yB, xL, xR, q, o)
        xL = self.cx - self.r
        xR = self.cx + self.r
        yT = self.cy - self.r
        yB = self.cy + self.r
        return yT, yB, xL, xR, self.question, self.option


@dataclass
class LayoutParams:
    num_questions: int = 50
    num_cols: int = 2
    questions_per_col: int = 25
    options_per_question: int = 5

    # Global geometry
    margin_left: int = 60
    margin_top: int = 60
    column_width: int = 260
    column_gap: int = 40

    row_height: int = 28
    circle_diameter: int = 22
    option_gap: int = 18
    option_left_padding: int = 8  # khoảng từ mép trái cột đến tâm A trừ r

    # Per-column offsets
    col_dx: List[int] | None = None
    col_dy: List[int] | None = None

    def ensure_offsets(self):
        if self.col_dx is None or len(self.col_dx) != self.num_cols:
            self.col_dx = [0 for _ in range(self.num_cols)]
        if self.col_dy is None or len(self.col_dy) != self.num_cols:
            self.col_dy = [0 for _ in range(self.num_cols)]


# ====== Layout generator (circles) ======
def build_circle_grid(img_w: int, img_h: int, p: LayoutParams) -> List[CircleROI]:
    p.ensure_offsets()
    rois: List[CircleROI] = []
    q_total = p.num_questions
    q_per_col = max(1, p.questions_per_col)
    cols = max(p.num_cols, int(np.ceil(q_total / q_per_col)))
    opts = p.options_per_question
    r = max(1, int(p.circle_diameter // 2))

    q_idx = 1
    for c in range(cols):
        col_x0 = p.margin_left + c * (p.column_width + p.column_gap) + p.col_dx[c]
        col_y0 = p.margin_top + p.col_dy[c]
        for r_row in range(q_per_col):
            if q_idx > q_total:
                break
            cy = int(col_y0 + r_row * p.row_height + r)
            # Tâm option A: mép trái cột + padding + r
            base_cx = int(col_x0 + p.option_left_padding + r)
            for o in range(opts):
                cx = int(base_cx + o * (p.circle_diameter + p.option_gap))
                # Clamp tâm để hình tròn nằm trong ảnh
                cx = int(np.clip(cx, r, max(r, img_w - r)))
                cy = int(np.clip(cy, r, max(r, img_h - r)))
                rois.append(CircleROI(cx, cy, r, q_idx, o))
            q_idx += 1
    return rois


# ====== App (no zoom, drag & drop by column) ======
class CircleGridEditorApp:
    """
    Lưới ROI tròn + kéo–thả theo cột (move/resize width/row height/option gap).
    """

    HANDLE_THICK = 10
    OG_HANDLE_HEIGHT = 14

    def __init__(self, image_bgr: np.ndarray):
        self.img_bgr = image_bgr
        self.img_h, self.img_w = self.img_bgr.shape[:2]
        self.base_rgb = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2RGB)

        self.params = LayoutParams()
        self.params.ensure_offsets()
        self.selected_col = 0

        self.root = tk.Tk()
        self.root.title("Circle ROI Grid Editor")

        self.canvas = tk.Canvas(self.root, bg="black", width=1000, height=1200, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        control = ttk.Frame(self.root, padding=8)
        control.grid(row=0, column=1, sticky="ns")
        self._build_controls(control)

        self.drag_mode: str | None = None
        self.drag_col: int = -1
        self.drag_start_img_xy: Tuple[int, int] = (0, 0)
        self.params_snapshot: LayoutParams | None = None

        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self._redraw()

    # ----- UI helpers -----
    def _add_spin(self, parent, label, from_, to, val, row, col, cb):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w")
        sp = tk.Spinbox(parent, from_=from_, to=to, width=8, command=cb)
        sp.delete(0, "end"); sp.insert(0, str(val))
        sp.grid(row=row, column=col+1, sticky="w")
        return sp

    def _build_controls(self, panel: ttk.Frame):
        r = 0
        ttk.Label(panel, text="Cấu hình lưới (tròn)").grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.sp_q = self._add_spin(panel, "Số câu", 1, 1000, self.params.num_questions, r, 0, self._on_param_change); r += 1
        self.sp_cols = self._add_spin(panel, "Số cột", 1, 10, self.params.num_cols, r, 0, self._on_change_cols); r += 1
        self.sp_qpc = self._add_spin(panel, "Câu mỗi cột", 1, 1000, self.params.questions_per_col, r, 0, self._on_param_change); r += 1
        self.sp_opts = self._add_spin(panel, "Số đáp án", 1, 10, self.params.options_per_question, r, 0, self._on_param_change); r += 1

        ttk.Separator(panel).grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        ttk.Label(panel, text="Kích thước/Giãn (toàn cục)").grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.sp_margin_l = self._add_spin(panel, "Margin trái", 0, 1000, self.params.margin_left, r, 0, self._on_param_change); r += 1
        self.sp_margin_t = self._add_spin(panel, "Margin trên", 0, 1000, self.params.margin_top, r, 0, self._on_param_change); r += 1
        self.sp_col_w   = self._add_spin(panel, "Rộng cột", 20, 3000, self.params.column_width, r, 0, self._on_param_change); r += 1
        self.sp_col_gap = self._add_spin(panel, "Khoảng cột", 0, 3000, self.params.column_gap, r, 0, self._on_param_change); r += 1
        self.sp_row_h   = self._add_spin(panel, "Cao hàng", 8, 500, self.params.row_height, r, 0, self._on_param_change); r += 1
        self.sp_cdia    = self._add_spin(panel, "Đường kính ô", 6, 400, self.params.circle_diameter, r, 0, self._on_param_change); r += 1
        self.sp_opt_gap = self._add_spin(panel, "Khoảng ô", 0, 400, self.params.option_gap, r, 0, self._on_param_change); r += 1
        self.sp_opt_pad = self._add_spin(panel, "Padding trái ô", 0, 400, self.params.option_left_padding, r, 0, self._on_param_change); r += 1

        ttk.Separator(panel).grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        ttk.Label(panel, text="Cột chọn").grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.sp_sel_col = self._add_spin(panel, "Chọn cột", 0, self.params.num_cols-1, 0, r, 0, self._on_select_col); r += 1

        fo = ttk.Frame(panel); fo.grid(row=r, column=0, columnspan=2, pady=6, sticky="ew"); r += 1
        ttk.Button(fo, text="Load Preset", command=self._load_preset).grid(row=0, column=0, padx=2)
        ttk.Button(fo, text="Save Preset", command=self._save_preset).grid(row=0, column=1, padx=2)
        ttk.Button(fo, text="Export Circles JSON", command=self._export_circles).grid(row=1, column=0, columnspan=2, padx=2, pady=4)
        ttk.Button(fo, text="Export Rect (compat)", command=self._export_rects).grid(row=2, column=0, columnspan=2, padx=2, pady=4)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(panel, textvariable=self.status, foreground="#0a7").grid(row=r, column=0, columnspan=2, sticky="w", pady=4)

    # ----- Param handlers -----
    def _on_param_change(self):
        try:
            self.params.num_questions = int(self.sp_q.get())
            self.params.questions_per_col = int(self.sp_qpc.get())
            self.params.options_per_question = int(self.sp_opts.get())
            self.params.margin_left = int(self.sp_margin_l.get())
            self.params.margin_top = int(self.sp_margin_t.get())
            self.params.column_width = max(20, int(self.sp_col_w.get()))
            self.params.column_gap = max(0, int(self.sp_col_gap.get()))
            self.params.row_height = max(8, int(self.sp_row_h.get()))
            self.params.circle_diameter = max(6, int(self.sp_cdia.get()))
            self.params.option_gap = max(0, int(self.sp_opt_gap.get()))
            self.params.option_left_padding = max(0, int(self.sp_opt_pad.get()))
        except Exception:
            pass
        self._redraw()

    def _on_change_cols(self):
        try:
            new_cols = max(1, int(self.sp_cols.get()))
        except Exception:
            return
        old_cols = self.params.num_cols
        self.params.num_cols = new_cols
        self.params.ensure_offsets()
        if new_cols > old_cols:
            for _ in range(new_cols - old_cols):
                self.params.col_dx.append(0); self.params.col_dy.append(0)
        else:
            self.params.col_dx = self.params.col_dx[:new_cols]
            self.params.col_dy = self.params.col_dy[:new_cols]
        self.sp_sel_col.config(to=new_cols - 1)
        self.selected_col = min(self.selected_col, new_cols - 1)
        self._redraw()

    def _on_select_col(self):
        try:
            self.selected_col = max(0, min(int(self.sp_sel_col.get()), self.params.num_cols - 1))
        except Exception:
            pass
        self._redraw()

    # ----- Canvas mapping -----
    def _canvas_size(self) -> Tuple[int, int]:
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _canvas_to_img(self, cx: int, cy: int) -> Tuple[int, int]:
        cw, ch = self._canvas_size()
        sx = self.img_w / cw
        sy = self.img_h / ch
        return int(cx * sx), int(cy * sy)

    # ----- Column geometry -----
    def _column_rect_img(self, c: int) -> Tuple[int, int, int, int]:
        p = self.params
        x0 = p.margin_left + c * (p.column_width + p.column_gap) + p.col_dx[c]
        y0 = p.margin_top + p.col_dy[c]
        x1 = x0 + p.column_width
        y1 = y0 + p.questions_per_col * p.row_height
        return int(x0), int(y0), int(x1), int(y1)

    # ----- Mouse / Drag -----
    def _on_mouse_down(self, event):
        ix, iy = self._canvas_to_img(event.x, event.y)
        hit_col = -1
        edge_right = False
        edge_bottom = False
        og_handle = False

        for c in range(self.params.num_cols):
            x0, y0, x1, y1 = self._column_rect_img(c)
            if (x0 <= ix <= x1) and (y0 <= iy <= y1):
                hit_col = c
                if abs(ix - x1) <= self.HANDLE_THICK:
                    edge_right = True
                if abs(iy - y1) <= self.HANDLE_THICK:
                    edge_bottom = True
                if (y0 - self.OG_HANDLE_HEIGHT <= iy <= y0) and (x0 <= ix <= x1):
                    og_handle = True
                break
            if (x0 <= ix <= x1) and (y0 - self.OG_HANDLE_HEIGHT <= iy <= y0):
                hit_col = c
                og_handle = True
                break

        if hit_col == -1:
            self.drag_mode = None
            return

        self.selected_col = hit_col
        self.sp_sel_col.delete(0, "end"); self.sp_sel_col.insert(0, str(self.selected_col))

        if og_handle:
            self.drag_mode = "adjust_option_gap"
        elif edge_right:
            self.drag_mode = "resize_col_width"
        elif edge_bottom:
            self.drag_mode = "resize_row_height"
        else:
            self.drag_mode = "move_col"

        self.drag_col = hit_col
        self.drag_start_img_xy = (ix, iy)
        self.params_snapshot = LayoutParams(**asdict(self.params))
        self.params_snapshot.ensure_offsets()
        self.status.set(f"Drag {self.drag_mode} @ col {hit_col}")
        self._redraw()

    def _on_mouse_drag(self, event):
        if not self.drag_mode or self.drag_col < 0:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        sx, sy = ix - self.drag_start_img_xy[0], iy - self.drag_start_img_xy[1]
        p0 = self.params_snapshot
        p = self.params

        if self.drag_mode == "move_col":
            p.col_dx[self.drag_col] = int(p0.col_dx[self.drag_col] + sx)
            p.col_dy[self.drag_col] = int(p0.col_dy[self.drag_col] + sy)
        elif self.drag_mode == "resize_col_width":
            p.column_width = max(20, int(p0.column_width + sx))
        elif self.drag_mode == "resize_row_height":
            p.row_height = max(8, int(p0.row_height + sy / max(1, p.questions_per_col)))
        elif self.drag_mode == "adjust_option_gap":
            p.option_gap = max(0, int(p0.option_gap + sx / max(1, p.options_per_question)))

        self._redraw()

    def _on_mouse_up(self, _event):
        self.drag_mode = None
        self.drag_col = -1
        self.params_snapshot = None
        self.status.set("Ready")

    # ----- Preset / Export -----
    def _load_preset(self):
        path = filedialog.askopenfilename(title="Load Preset (JSON)", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.params = LayoutParams(**data)
            self.params.ensure_offsets()
            self.sp_sel_col.config(to=self.params.num_cols - 1)
            self.selected_col = 0
            self._sync_spins_from_params()
            self.status.set(f"Loaded preset: {os.path.basename(path)}")
            self._redraw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_preset(self):
        path = filedialog.asksaveasfilename(title="Save Preset (JSON)", defaultextension=".json",
                                            filetypes=[("JSON", "*.json")], initialfile="omr_bubble_grid_template.json")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.params), f, indent=2, ensure_ascii=False)
            self.status.set(f"Saved preset -> {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export_circles(self):
        rois = build_circle_grid(self.img_w, self.img_h, self.params)
        path = filedialog.asksaveasfilename(title="Export Circles JSON", defaultextension=".json",
                                            initialfile="omr_bubble_layout.json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in rois], f, indent=2, ensure_ascii=False)
            self.status.set(f"Exported {len(rois)} circles -> {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export_rects(self):
        rois = build_circle_grid(self.img_w, self.img_h, self.params)
        path = filedialog.asksaveasfilename(title="Export Rect (compat)", defaultextension=".json",
                                            initialfile="roi_calibrated_rect.json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([r.to_rect_tuple() for r in rois], f, indent=2, ensure_ascii=False)
            self.status.set(f"Exported {len(rois)} rects -> {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ----- Drawing -----
    def _compose_overlay(self) -> np.ndarray:
        vis = self.base_rgb.copy()
        p = self.params
        circles = build_circle_grid(self.img_w, self.img_h, p)

        # Draw columns & handles
        for c in range(p.num_cols):
            x0, y0, x1, y1 = self._column_rect_img(c)
            if c == self.selected_col:
                cv.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)
            else:
                cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
            og_y0 = max(0, y0 - self.OG_HANDLE_HEIGHT)
            cv.rectangle(vis, (x0, og_y0), (x1, y0), (0, 128, 255), -1)
            cv.putText(vis, "opt-gap", (x0 + 4, og_y0 + self.OG_HANDLE_HEIGHT - 4),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv.line(vis, (x1, y0), (x1, y1), (0, 165, 255), 3)  # right edge
            cv.line(vis, (x0, y1), (x1, y1), (0, 165, 255), 3)  # bottom edge

        # Draw circle ROIs
        for r in circles:
            cv.circle(vis, (r.cx, r.cy), r.r, (0, 255, 0), 3)

        return vis

    def _redraw(self):
        vis = self._compose_overlay()
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        disp = cv.resize(vis, (cw, ch), interpolation=cv.INTER_LINEAR)
        img = Image.fromarray(disp)
        self._photo = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _sync_spins_from_params(self):
        p = self.params
        pairs = [
            (self.sp_q, p.num_questions),
            (self.sp_cols, p.num_cols),
            (self.sp_qpc, p.questions_per_col),
            (self.sp_opts, p.options_per_question),
            (self.sp_margin_l, p.margin_left),
            (self.sp_margin_t, p.margin_top),
            (self.sp_col_w, p.column_width),
            (self.sp_col_gap, p.column_gap),
            (self.sp_row_h, p.row_height),
            (self.sp_cdia, p.circle_diameter),
            (self.sp_opt_gap, p.option_gap),
            (self.sp_opt_pad, p.option_left_padding),
        ]
        for sp, val in pairs:
            sp.delete(0, "end"); sp.insert(0, str(val))

    # ----- Run -----
    @staticmethod
    def run(image_path: str):
        img = cv.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        app = CircleGridEditorApp(img)
        app.root.mainloop()


# ====== CLI ======
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Circle ROI Grid Editor (drag & drop, no zoom)")
    parser.add_argument("image", nargs="?", default=os.path.join("../samples", "template_scan1.png"))
    args = parser.parse_args()
    CircleGridEditorApp.run(args.image)


if __name__ == "__main__":
    main()
