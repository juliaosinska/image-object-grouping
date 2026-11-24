# GENAI: Gemini and Claude for UI implementation (pretty much whole file)
# Gemini - base code for GUI layout, widgets, and event handling (from previous tasks)
# Claude - adaptation for task 7: loading the whole dataset folder (images + labels.csv), preview of the
#          images with their masks, example objects with masks, descriptor comparison per category,
#          scatter plot of two chosen descriptors colored by category, K-means view with centroids and
#          the cluster x category table, feature table (Treeview) with sorting and CSV export
#
# The GUI only calls the analysis functions and draws the results. All analysis is done in:
#   dataset.py       - load_labels, assign_grid_positions, load_dataset
#   mask_preparation - compute_mask          objects_analysis - connected_component_labeling
#   features.py      - build_feature_table   clustering.py    - run_kmeans

import os
import traceback
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mask_preparation import compute_mask
from objects_analysis import connected_component_labeling
from dataset import load_dataset
from features import build_feature_table
from clustering import run_kmeans

VIEW_IMAGES = "1. Images and masks"
VIEW_EXAMPLES = "2. Example objects"
VIEW_DESCRIPTORS = "3. Descriptors per category"
VIEW_SCATTER = "4. Scatter plot (categories)"
VIEW_KMEANS = "5. K-means clustering"
VIEWS = [VIEW_IMAGES, VIEW_EXAMPLES, VIEW_DESCRIPTORS, VIEW_SCATTER, VIEW_KMEANS]

OBJECTS_DARKER = "darker"
OBJECTS_BRIGHTER = "brighter"
THRESHOLD_OTSU = "Otsu"
THRESHOLD_FIXED = "Fixed"

# columns of the feature table that are not descriptors
INFO_COLUMNS = ["id", "file", "category", "cluster"]

CATEGORY_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
CLUSTER_COLORS = ["#e6550d", "#3182bd", "#31a354", "#756bb1", "#e7ba52", "#636363", "#d6616b", "#17becf"]
CATEGORY_MARKERS = ["o", "^", "s", "D", "v", "P"]
TABLE_TAG_COLORS = ["#e8f1fa", "#fbe8e8", "#e8f6e8", "#f1ebf7", "#fff1e0", "#f3ece8"]
EXAMPLES_PER_CATEGORY = 6


class ObjectGroupingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Grouping by Geometric and Texture Descriptors")
        self.root.geometry("1400x950")

        # --- Application state ---
        self.folder = None          # dataset folder (images + labels.csv)
        self.images = []            # [(file name, rgb array, mask, number of objects)] for the preview
        self.objects = []           # result of load_dataset
        self.df = None              # feature table (pandas DataFrame)
        self.categories = []        # sorted list of the real categories
        self.kmeans = None          # (feature_x, feature_y, k, clusters, centroids, comparison)
        self.sort_state = (None, False)

        # --- 1. Top Control Panel ---
        control_frame = tk.LabelFrame(root, text="Control Panel", padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

        tk.Button(control_frame, text="Load dataset folder", command=self.choose_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Process dataset", command=self.process_dataset,
                  bg="#e0e0e0").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Export table (CSV)", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        self.folder_var = tk.StringVar(value="Dataset: -")
        tk.Label(control_frame, textvariable=self.folder_var, fg="#555").pack(side=tk.LEFT, padx=(10, 15))

        tk.Label(control_frame, text="View:").pack(side=tk.LEFT, padx=(15, 2))
        self.view_var = tk.StringVar(value=VIEW_IMAGES)
        view_dropdown = ttk.Combobox(control_frame, textvariable=self.view_var, state="readonly",
                                     width=30, values=VIEWS)
        view_dropdown.pack(side=tk.LEFT, padx=5)
        view_dropdown.bind("<<ComboboxSelected>>", lambda event: self.draw_view())

        # --- 2. Parameters ---
        params_frame = tk.LabelFrame(root, text="Parameters", padx=10, pady=5)
        params_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # mask (task 4-6 segmentation); defaults for the buttons dataset
        mask_frame = tk.LabelFrame(params_frame, text="Mask (click 'Process dataset' after a change)", padx=5, pady=3)
        mask_frame.pack(side=tk.LEFT, padx=5)
        self.channel_var = tk.StringVar(value="Gray")
        self.objects_var = tk.StringVar(value=OBJECTS_DARKER)
        self.add_combobox(mask_frame, "Channel:", self.channel_var, ["Gray", "Blue"])
        self.add_combobox(mask_frame, "Objects:", self.objects_var, [OBJECTS_DARKER, OBJECTS_BRIGHTER])
        self.threshold_method_var = tk.StringVar(value=THRESHOLD_FIXED)
        tk.Label(mask_frame, text="Threshold:").pack(side=tk.LEFT, padx=(5, 2))
        self.threshold_dropdown = ttk.Combobox(mask_frame, textvariable=self.threshold_method_var,
                                               state="readonly", width=6, values=[THRESHOLD_OTSU, THRESHOLD_FIXED])
        self.threshold_dropdown.pack(side=tk.LEFT, padx=(0, 8))
        self.threshold_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_threshold_params())
        self.threshold_params = tk.Frame(mask_frame)
        self.threshold_value_var = tk.IntVar(value=250)
        self.add_spinbox(self.threshold_params, "t:", self.threshold_value_var, 0, 255, 1)
        self.update_threshold_params()
        self.se_size_var = tk.IntVar(value=3)
        self.add_spinbox(mask_frame, "Cleaning SE (0 = off):", self.se_size_var, 0, 51, 1)

        # analysis: two descriptors for the scatter plot and K-means, number of clusters
        analysis_frame = tk.LabelFrame(params_frame, text="Scatter plot / K-means", padx=5, pady=3)
        analysis_frame.pack(side=tk.LEFT, padx=5)
        self.feature_x_var = tk.StringVar(value="contrast")
        self.feature_y_var = tk.StringVar(value="homogeneity")
        tk.Label(analysis_frame, text="X:").pack(side=tk.LEFT, padx=(5, 2))
        self.feature_x_box = ttk.Combobox(analysis_frame, textvariable=self.feature_x_var, state="readonly", width=14)
        self.feature_x_box.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(analysis_frame, text="Y:").pack(side=tk.LEFT, padx=(5, 2))
        self.feature_y_box = ttk.Combobox(analysis_frame, textvariable=self.feature_y_var, state="readonly", width=14)
        self.feature_y_box.pack(side=tk.LEFT, padx=(0, 8))
        for box in (self.feature_x_box, self.feature_y_box):
            box.bind("<<ComboboxSelected>>", lambda event: self.on_analysis_changed())
        self.k_var = tk.IntVar(value=2)
        tk.Label(analysis_frame, text="K:").pack(side=tk.LEFT, padx=(5, 2))
        tk.Spinbox(analysis_frame, from_=1, to=10, increment=1, textvariable=self.k_var, width=4,
                   command=self.on_analysis_changed).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(analysis_frame, text="Run K-means", command=self.on_analysis_changed).pack(side=tk.LEFT, padx=5)

        # --- 3. Main area: matplotlib figure (top) and feature table (bottom) ---
        main = tk.Frame(root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=3, uniform="row")
        main.grid_rowconfigure(1, weight=2, uniform="row")

        figure_frame = tk.LabelFrame(main, text="Results")
        figure_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        self.figure = Figure(figsize=(4, 3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        table_frame = tk.LabelFrame(main, text="Feature table (click a header to sort)")
        table_frame.grid(row=1, column=0, sticky="nsew")
        self.table = ttk.Treeview(table_frame, show="headings", height=8)
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.table.xview)
        self.table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.table.pack(fill=tk.BOTH, expand=True)

        # --- 4. Status Bar ---
        self.status_var = tk.StringVar(value="Ready. Load a dataset folder (images + labels.csv).")
        tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                 padx=5, pady=3).pack(side=tk.BOTTOM, fill=tk.X)

        # default: the "dataset" folder next to this file, if it exists
        default_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        if os.path.isfile(os.path.join(default_folder, "labels.csv")):
            self.set_folder(default_folder)

    # ------------------------------------------------------------------
    # Small widget helpers
    # ------------------------------------------------------------------
    def add_combobox(self, parent, label, variable, values):
        tk.Label(parent, text=label).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly",
                     width=7).pack(side=tk.LEFT, padx=(0, 8))

    def add_spinbox(self, parent, label, variable, from_, to, increment):
        tk.Label(parent, text=label).pack(side=tk.LEFT, padx=(5, 2))
        tk.Spinbox(parent, from_=from_, to=to, increment=increment, textvariable=variable,
                   width=5).pack(side=tk.LEFT, padx=(0, 8))

    def update_threshold_params(self):
        self.threshold_params.pack_forget()
        if self.threshold_method_var.get() == THRESHOLD_FIXED:
            self.threshold_params.pack(side=tk.LEFT, after=self.threshold_dropdown)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def choose_folder(self):
        folder = filedialog.askdirectory(title="Dataset folder (images + labels.csv)")
        if folder:
            self.set_folder(folder)

    def set_folder(self, folder):
        self.folder = folder
        self.folder_var.set(f"Dataset: {os.path.basename(folder)}")
        if not os.path.isfile(os.path.join(folder, "labels.csv")):
            self.status_var.set("This folder has no labels.csv - the real categories are needed for the comparison.")
            return
        self.status_var.set(f"Dataset folder: {folder} - click 'Process dataset'")

    def read_mask_params(self):
        se_size = int(self.se_size_var.get())
        if se_size != 0 and (se_size < 3 or se_size % 2 == 0):
            raise ValueError("Cleaning SE size must be 0 (no morphology) or an odd integer >= 3.")
        threshold = None
        if self.threshold_method_var.get() == THRESHOLD_FIXED:
            threshold = int(self.threshold_value_var.get())
            if not 0 <= threshold <= 255:
                raise ValueError("Threshold t must be between 0 and 255.")
        # same keys as the parameters of compute_mask (load_dataset calls compute_mask(rgb, **mask_params))
        return {"channel": self.channel_var.get(), "dark_objects": self.objects_var.get() == OBJECTS_DARKER,
                "background": "None", "background_param": None, "se_size": se_size, "threshold": threshold}

    def process_dataset(self):
        if self.folder is None:
            self.status_var.set("Choose the dataset folder first.")
            return
        try:
            mask_params = self.read_mask_params()
        except (ValueError, tk.TclError) as e:
            self.status_var.set(f"Invalid parameter: {e}")
            return

        self.status_var.set("Processing the dataset...")
        self.root.update_idletasks()
        try:
            self.images = self.preview_images(mask_params)
            self.objects = load_dataset(self.folder, os.path.join(self.folder, "labels.csv"), mask_params)
            self.df = build_feature_table(self.objects)
        except Exception as e:
            traceback.print_exc()
            self.objects, self.df = [], None
            self.status_var.set(f"Error {type(e).__name__}: {e} (traceback in the console)")
            self.fill_table()
            self.draw_view()
            return

        self.categories = sorted(self.df["category"].unique())
        features = self.descriptor_columns()
        self.feature_x_box.configure(values=features)
        self.feature_y_box.configure(values=features)
        if self.feature_x_var.get() not in features:
            self.feature_x_var.set(features[0])
        if self.feature_y_var.get() not in features:
            self.feature_y_var.set(features[1] if len(features) > 1 else features[0])

        self.run_clustering()
        self.fill_table()
        self.draw_view()
        counts = ", ".join(f"{c}: {(self.df['category'] == c).sum()}" for c in self.categories)
        self.status_var.set(f"{len(self.images)} images, {len(self.df)} objects ({counts})")

    def preview_images(self, mask_params):
        """Whole images with their masks (only for the preview of the segmentation)."""
        images = []
        for name in sorted(os.listdir(self.folder)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                rgb = np.array(Image.open(os.path.join(self.folder, name)).convert("RGB"))
                mask, t = compute_mask(rgb, **mask_params)
                _, n = connected_component_labeling(mask, 8)
                images.append((name, rgb, mask, n, t))
        return images

    def descriptor_columns(self):
        return [c for c in self.df.columns if c not in INFO_COLUMNS]

    # ------------------------------------------------------------------
    # K-means
    # ------------------------------------------------------------------
    def on_analysis_changed(self):
        if self.df is None:
            return
        self.run_clustering()
        self.fill_table()
        if self.view_var.get() in (VIEW_SCATTER, VIEW_KMEANS):
            self.draw_view()

    def run_clustering(self):
        self.kmeans = None
        try:
            k = int(self.k_var.get())
            if not 1 <= k <= len(self.df):
                raise ValueError(f"K must be between 1 and the number of objects ({len(self.df)}).")
            fx, fy = self.feature_x_var.get(), self.feature_y_var.get()
            clusters, centroids, comparison = run_kmeans(self.df, fx, fy, k)
        except Exception as e:
            traceback.print_exc()
            self.status_var.set(f"K-means error {type(e).__name__}: {e}")
            if "cluster" in self.df.columns:
                self.df = self.df.drop(columns="cluster")
            return
        self.df["cluster"] = np.asarray(clusters)
        self.kmeans = (fx, fy, k, np.asarray(clusters), np.asarray(centroids), comparison)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_view(self):
        self.figure.clear()
        try:
            if not self.images:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "Load a dataset folder and click 'Process dataset'", ha="center", va="center")
                ax.axis("off")
            elif self.df is None and self.view_var.get() != VIEW_IMAGES:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No objects - see the status bar", ha="center", va="center")
                ax.axis("off")
            else:
                {
                    VIEW_IMAGES: self.view_images,
                    VIEW_EXAMPLES: self.view_examples,
                    VIEW_DESCRIPTORS: self.view_descriptors,
                    VIEW_SCATTER: self.view_scatter,
                    VIEW_KMEANS: self.view_kmeans,
                }[self.view_var.get()]()
        except Exception as e:
            traceback.print_exc()
            self.status_var.set(f"Error while drawing: {type(e).__name__}: {e} (traceback in the console)")
        finally:
            self.canvas.draw_idle()

    def category_style(self, category):
        index = self.categories.index(category)
        return CATEGORY_COLORS[index % len(CATEGORY_COLORS)], CATEGORY_MARKERS[index % len(CATEGORY_MARKERS)]

    # View 1: every image with its mask (check of the segmentation)
    def view_images(self):
        axes = self.figure.subplots(2, len(self.images), squeeze=False)
        for col, (name, rgb, mask, n, t) in enumerate(self.images):
            axes[0][col].imshow(rgb)
            axes[0][col].set_title(name, fontsize=9)
            axes[1][col].imshow(mask, cmap="gray")
            axes[1][col].set_title(f"mask: t = {t}, {n} objects", fontsize=9)
            for ax in axes[:, col]:
                ax.axis("off")

    # View 2: a few objects of every category, image above its mask
    def view_examples(self):
        rows = 2 * len(self.categories)
        axes = self.figure.subplots(rows, EXAMPLES_PER_CATEGORY, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for c_index, category in enumerate(self.categories):
            examples = [o for o in self.objects if o["category"] == category][:EXAMPLES_PER_CATEGORY]
            for col, obj in enumerate(examples):
                ax_img, ax_mask = axes[2 * c_index][col], axes[2 * c_index + 1][col]
                # pixels outside the object's mask shown white (neighbours inside the margin are not analysed)
                ax_img.imshow(np.where(obj["mask"][..., None], obj["image"], 255).astype(np.uint8))
                ax_img.set_title(f"#{obj['id']} {category}", fontsize=8, color=self.category_style(category)[0])
                ax_mask.imshow(obj["mask"], cmap="gray")

    # View 3: distribution of every descriptor in every category
    def view_descriptors(self):
        features = self.descriptor_columns()
        cols = 4
        rows = int(np.ceil(len(features) / cols))
        axes = self.figure.subplots(rows, cols, squeeze=False)
        for ax in axes.ravel()[len(features):]:
            ax.axis("off")
        for ax, feature in zip(axes.ravel(), features):
            data = [self.df.loc[self.df["category"] == c, feature].to_numpy() for c in self.categories]
            parts = ax.boxplot(data, patch_artist=True, widths=0.6)
            ax.set_xticks(range(1, len(self.categories) + 1), self.categories, fontsize=8)
            for patch, c in zip(parts["boxes"], self.categories):
                patch.set_facecolor(self.category_style(c)[0])
                patch.set_alpha(0.45)
            ax.set_title(feature, fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
        self.status_var.set("Distribution of every descriptor per category - descriptors whose boxes "
                            "do not overlap separate the categories")

    # View 4: two chosen descriptors, colors = real categories
    def view_scatter(self):
        fx, fy = self.feature_x_var.get(), self.feature_y_var.get()
        ax = self.figure.add_subplot(111)
        for c in self.categories:
            part = self.df[self.df["category"] == c]
            color, marker = self.category_style(c)
            ax.scatter(part[fx], part[fy], color=color, marker=marker, s=40, alpha=0.8,
                       edgecolor="black", linewidth=0.4, label=f"{c} ({len(part)})")
        self.annotate_ids(ax, fx, fy)
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(f"{fy} vs {fx} - real categories", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        self.status_var.set(f"Scatter plot: {fx} (X) vs {fy} (Y), colors = real categories")

    # View 5: same two descriptors, colors = K-means clusters, markers = real categories
    def view_kmeans(self):
        if self.kmeans is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "K-means not computed - see the status bar", ha="center", va="center")
            ax.axis("off")
            return
        fx, fy, k, clusters, centroids, comparison = self.kmeans
        ax, ax_table = self.figure.subplots(1, 2, width_ratios=[3, 1])

        for cluster in range(k):
            color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
            for c in self.categories:
                part = self.df[(clusters == cluster) & (self.df["category"] == c).to_numpy()]
                ax.scatter(part[fx], part[fy], color=color, marker=self.category_style(c)[1], s=40,
                           alpha=0.8, edgecolor="black", linewidth=0.4)
            ax.scatter(centroids[cluster, 0], centroids[cluster, 1], color=color, marker="X", s=260,
                       edgecolor="black", linewidth=1.2, zorder=5)
        self.annotate_ids(ax, fx, fy)

        # legend: clusters (colors), real categories (markers), centroids
        handles = [Line2D([], [], marker="o", linestyle="", color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                          markeredgecolor="black", label=f"cluster {i}") for i in range(k)]
        handles += [Line2D([], [], marker=self.category_style(c)[1], linestyle="", color="white",
                           markeredgecolor="black", label=f"category: {c}") for c in self.categories]
        handles.append(Line2D([], [], marker="X", linestyle="", color="gray", markeredgecolor="black",
                              markersize=12, label="centroid"))
        ax.legend(handles=handles, fontsize=8)
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(f"K-means, K = {k} (standardized {fx} and {fy}); colors = clusters, markers = categories",
                     fontsize=10)
        ax.grid(alpha=0.3)

        # cluster x category table (counts)
        ax_table.axis("off")
        cells = [[str(v) for v in row] for row in comparison.to_numpy()]
        table = ax_table.table(cellText=cells, rowLabels=[f"cluster {i}" for i in comparison.index],
                               colLabels=[str(c) for c in comparison.columns], loc="center")
        table.scale(1, 1.6)
        ax_table.set_title("objects: cluster x category", fontsize=9)

        # agreement: every cluster is assigned to its most frequent category; baseline = one group for
        # everything (the share of the largest category) - with unequal categories this is already high
        n = len(self.df)
        matched = int(comparison.max(axis=1).sum())
        baseline = int(self.df["category"].value_counts().max())
        self.status_var.set(f"K-means K = {k} on {fx} / {fy}: {matched} of {n} objects ({100 * matched / n:.0f}%) "
                            f"are in a cluster dominated by their own category "
                            f"(baseline, all in one group: {100 * baseline / n:.0f}%)")

    def annotate_ids(self, ax, fx, fy):
        for _, row in self.df.iterrows():
            ax.annotate(str(row["id"]), (row[fx], row[fy]), fontsize=6, xytext=(3, 3), textcoords="offset points")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    @staticmethod
    def format_value(value):
        if isinstance(value, str):
            return value
        value = float(value)
        return f"{value:.0f}" if value.is_integer() else f"{value:.3f}"

    def fill_table(self):
        self.table.delete(*self.table.get_children())
        self.sort_state = (None, False)
        if self.df is None:
            self.table.configure(columns=())
            return
        columns = list(self.df.columns)
        self.table.configure(columns=columns)
        for col in columns:
            width = 110 if col in ("file", "category") else 85
            self.table.heading(col, text=col, command=lambda c=col: self.sort_table(c))
            self.table.column(col, width=width, anchor=tk.CENTER, stretch=False)
        for i, c in enumerate(self.categories):
            self.table.tag_configure(c, background=TABLE_TAG_COLORS[i % len(TABLE_TAG_COLORS)])
        for _, row in self.df.iterrows():
            self.table.insert("", tk.END, values=[self.format_value(row[c]) for c in columns],
                              tags=(row["category"],))

    def sort_table(self, column):
        last_column, descending = self.sort_state
        descending = not descending if column == last_column else False
        self.sort_state = (column, descending)

        def key(item):
            value = self.table.set(item, column)
            try:
                return (0, float(value))
            except ValueError:
                return (1, value)

        items = sorted(self.table.get_children(""), key=key, reverse=descending)
        for index, item in enumerate(items):
            self.table.move(item, "", index)

    def export_csv(self):
        if self.df is None:
            self.status_var.set("Nothing to export - process the dataset first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="feature_table.csv")
        if path:
            self.df.to_csv(path, index=False)
            self.status_var.set(f"Feature table saved: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectGroupingGUI(root)
    root.mainloop()
