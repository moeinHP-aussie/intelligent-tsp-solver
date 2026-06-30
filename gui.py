# =============================================================================
# فایل: src/gui.py
# قاز 4
#
#
#   ۱. ConfigTab  — پیکربندی داده: انتخاب کشور/شهر و نمایش اولیه گراف
#   ۲. AIAdvisorTab — مشاور هوشمند ML: پیشنهاد الگوریتم + اجرا
#   ۳. LiveSolverTab — انیمیشن زنده: نمایش گام‌به‌گام بهبود مسیر
#   ۴. BenchmarkTab — بنچمارک نهایی: مقایسه‌ی هر سه الگوریتم
#
# اصل کلیدی معماری این فایل:
#   هیچ عملیات سنگین‌ی (شبکه، محاسبه، Prolog) روی thread اصلی اجرا نمیشه.
#   همه از طریق الگوی QThread + pyqtSignal مدیریت میشن.
#

# =============================================================================

import os
import sys
import logging
import traceback
import math      # برای math.isnan — تشخیص نتایج NaN پرولاگ (Skip/Unavailable) در نمودارهای بنچمارک
from typing import Optional

# --- کتابخانه‌ی PyQt6 ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QProgressBar, QTextEdit, QGroupBox, QSplitter,
    QRadioButton, QButtonGroup, QFrame, QScrollArea,
    QSizePolicy, QStatusBar
)
from PyQt6.QtCore import (
    Qt, QThread, QObject, pyqtSignal, pyqtSlot, QTimer
)
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon

# --- Matplotlib با بکند PyQt6 ---
import matplotlib
matplotlib.use("QtAgg")  # بکند مناسب برای PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- ماژول‌های داخلی پروژه ---
# از sys.path مطمئن میشیم که src/ پیدا میشه
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import City, get_tsp_data, list_cached_datasets, MIN_CITIES, MAX_CITIES
from solvers import (
    AntColonyOptimizer, GeneticAlgorithmSolver,
    SolverResult, calculate_path_cost
)
from prolog_bridge import (
    solve_with_prolog, PrologSolverResult, PrologStatus,
    get_prolog_role_explanation, PROLOG_MAX_CITIES
)
import joblib

# لاگر اختصاصی این ماژول
logger = logging.getLogger("TSP.GUI")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR).lower() == "src" else _THIS_DIR
ADVISOR_MODEL_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "models", "ai_advisor_model.pkl"),
    os.path.join(_PROJECT_ROOT, "ml_models", "tree_classifier.pkl"),
    os.path.join(_THIS_DIR, "models", "ai_advisor_model.pkl"),
    os.path.join(_THIS_DIR, "..", "models", "ai_advisor_model.pkl"),
    os.path.join(_THIS_DIR, "..", "ml_models", "tree_classifier.pkl"),
]

ADVISOR_FALLBACK_FEATURES = [
    "n_cities", "coord_variance", "dist_mean", "dist_std",
    "dist_cv", "density_proxy", "nn_ratio", "dist_skewness", "max_min_ratio",
]


# =============================================================================
# ثابت‌های ظاهری (Visual Constants)
# =============================================================================

PALETTE = {
    "bg_dark":     "#12121C",   # پس‌زمینه‌ی اصلی (عمیق‌تر از قبل)
    "bg_mid":      "#1C1C2A",   # پس‌زمینه‌ی پنل‌ها
    "bg_light":    "#252538",   # پس‌زمینه‌ی ورودی‌ها
    "bg_card":     "#1E1E30",   # کارت‌های محتوا
    "accent":      "#7C6AF7",   # رنگ تأکید اصلی (بنفش)
    "accent_hot":  "#F7926A",   # نارنجی
    "accent_ok":   "#4ECCA3",   # سبز آکوا (زنده‌تر از قبل)
    "accent_warn": "#F7C948",   # زرد هشدار
    "text_main":   "#E8E8F8",   # متن اصلی
    "text_dim":    "#7878A0",   # متن کم‌رنگ
    "text_bright": "#FFFFFF",   # متن روشن برای تیترها
    "border":      "#35355A",   # حاشیه‌ی ظریف
    "border_glow": "#5A5A9A",   # حاشیه‌ی درخشان (hover)
    "shadow":      "#0A0A14",   # سایه
}

# ─────────────────────────────────────────────────────────────────────────────
# رنگ تأکید مجزا برای هر تب — هویت بصری جداگانه
# ─────────────────────────────────────────────────────────────────────────────
TAB_ACCENTS = {
    0: {"color": "#5B8AF5", "glow": "#3D6BE8", "name": "config",   "hex_dim": "#1E2840"},  # آبی — پیکربندی
    1: {"color": "#A855F7", "glow": "#8B3DDB", "name": "advisor",  "hex_dim": "#261836"},  # بنفش — مشاور
    2: {"color": "#F7926A", "glow": "#E07048", "name": "live",     "hex_dim": "#321E14"},  # نارنجی — انیمیشن
    3: {"color": "#4ECCA3", "glow": "#35AA85", "name": "bench",    "hex_dim": "#122820"},  # آکوا — بنچمارک
}

# ─────────────────────────────────────────────────────────────────────────────
# رنگ‌های مخصوص هر الگوریتم
# ─────────────────────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "ACO":     "#F7926A",   # نارنجی — مورچگان
    "Genetic": "#A855F7",   # بنفش — ژنتیک
    "Prolog":  "#4ECCA3",   # آکوا — پرولاگ (Ground Truth)
}

# فونت پیش‌فرض Matplotlib
MPL_FONT = "DejaVu Sans"


def _tab_group_style(tab_idx: int) -> str:
    """
    استایل QGroupBox با رنگ تأکید مخصوص تب.
    هر تب هویت رنگی کاملاً جداگانه دارد.
    """
    a = TAB_ACCENTS[tab_idx]
    return f"""
        QGroupBox {{
            font-weight: bold; font-size: 12px;
            color: {PALETTE["text_main"]};
            background: {PALETTE["bg_card"]};
            border: 1px solid {a["hex_dim"]};
            border-top: 2px solid {a["color"]};
            border-radius: 8px;
            margin-top: 12px; padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; left: 12px;
            padding: 0 6px;
            color: {a["color"]};
            font-size: 12px;
        }}
        QGroupBox:hover {{
            border-top-color: {a["glow"]};
        }}
    """


def _primary_btn(tab_idx: int) -> str:
    """دکمه‌ی اصلی با رنگ تب."""
    c = TAB_ACCENTS[tab_idx]["color"]
    g = TAB_ACCENTS[tab_idx]["glow"]
    return f"""
        QPushButton {{
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 {c}, stop:1 {g});
            color: {PALETTE["text_bright"]}; border: none;
            border-radius: 6px; padding: 8px 16px;
            font-size: 12px; font-weight: bold;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 {g}, stop:1 {c});
        }}
        QPushButton:pressed {{ padding-top: 10px; }}
        QPushButton:disabled {{
            background: {PALETTE["bg_light"]};
            color: {PALETTE["text_dim"]};
        }}
    """


def _secondary_btn(tab_idx: int) -> str:
    """دکمه‌ی ثانوی (outline) با رنگ تب."""
    c = TAB_ACCENTS[tab_idx]["color"]
    d = TAB_ACCENTS[tab_idx]["hex_dim"]
    return f"""
        QPushButton {{
            background: {d}; color: {c};
            border: 1px solid {c}; border-radius: 6px;
            padding: 7px 14px; font-size: 12px;
        }}
        QPushButton:hover {{
            background: {PALETTE["bg_light"]};
            border-color: {TAB_ACCENTS[tab_idx]["glow"]};
        }}
        QPushButton:disabled {{
            background: transparent;
            color: {PALETTE["text_dim"]};
            border-color: {PALETTE["border"]};
        }}
    """


# =============================================================================
# ابزارهای مشاور ML
# =============================================================================

def _find_advisor_model_path() -> Optional[str]:
    """اولین فایل مدل موجود را مطابق ساختار فعلی یا ساختار src/ پیدا می‌کند."""
    for path in ADVISOR_MODEL_CANDIDATES:
        normalized = os.path.abspath(path)
        if os.path.isfile(normalized):
            return normalized
    return None


def _normalize_algorithm_label(label) -> str:
    """برچسب مدل را به نام داخلی radio buttonهای GUI تبدیل می‌کند."""
    text = str(label)
    if text.startswith("Prolog"):
        return "Prolog"
    if text.startswith("ACO"):
        return "ACO"
    if text.startswith("Genetic"):
        return "Genetic"

    try:
        numeric = int(label)
    except (TypeError, ValueError):
        return "ACO"
    return {0: "Prolog", 1: "ACO", 2: "Genetic"}.get(numeric, "ACO")


def _extract_advisor_features(matrix: list[list[float]], cities: list[City]) -> dict[str, float]:
    """
    همان ۹ feature ذخیره‌شده در train_model.py را برای پیش‌بینی GUI می‌سازد.
    """
    n = len(matrix)
    all_distances = [
        float(matrix[i][j])
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    if not all_distances or not cities:
        return {feat: 0.0 for feat in ADVISOR_FALLBACK_FEATURES}

    f_n_cities = float(n)

    lats = [float(c.lat) for c in cities]
    lons = [float(c.lon) for c in cities]
    lat_mean = sum(lats) / n
    lon_mean = sum(lons) / n
    lat_var = sum((lat - lat_mean) ** 2 for lat in lats) / n
    lon_var = sum((lon - lon_mean) ** 2 for lon in lons) / n
    f_coord_variance = lat_var + lon_var

    dist_mean = sum(all_distances) / len(all_distances)
    dist_variance = sum((d - dist_mean) ** 2 for d in all_distances) / len(all_distances)
    dist_std = math.sqrt(dist_variance)
    f_dist_cv = (dist_std / dist_mean) if dist_mean > 0.0 else 0.0

    lat_range_km = (max(lats) - min(lats)) * 111.0
    lon_range_km = (max(lons) - min(lons)) * 111.0 * math.cos(math.radians(lat_mean))
    approx_area_km2 = max(lat_range_km * lon_range_km, 1.0)
    f_density_proxy = n / math.sqrt(approx_area_km2)

    nn_distances = []
    for i in range(n):
        nn_distances.append(min(float(matrix[i][j]) for j in range(n) if j != i))
    nn_mean = sum(nn_distances) / n
    f_nn_ratio = (nn_mean / dist_mean) if dist_mean > 0.0 else 1.0

    if dist_std > 0.0:
        dist_skewness = (
            sum((d - dist_mean) ** 3 for d in all_distances) / len(all_distances)
        ) / (dist_std ** 3)
    else:
        dist_skewness = 0.0
    f_dist_skewness = max(-5.0, min(5.0, dist_skewness))

    dist_min = min(all_distances)
    dist_max = max(all_distances)
    f_max_min_ratio = (dist_max / dist_min) if dist_min > 0.0 else dist_max

    return {
        "n_cities": f_n_cities,
        "coord_variance": f_coord_variance,
        "dist_mean": dist_mean,
        "dist_std": dist_std,
        "dist_cv": f_dist_cv,
        "density_proxy": f_density_proxy,
        "nn_ratio": f_nn_ratio,
        "dist_skewness": f_dist_skewness,
        "max_min_ratio": f_max_min_ratio,
    }


def _predict_advisor_algorithm(matrix: list[list[float]], cities: list[City]) -> tuple[str, str, bool, dict[str, float]]:
    """
    مدل آموزش‌دیده را load می‌کند، featureها را scale می‌کند و نام الگوریتم را برمی‌گرداند.
    """
    features = _extract_advisor_features(matrix, cities)
    model_path = _find_advisor_model_path()
    if not model_path:
        return "", "مدل آموزش‌دیده پیدا نشد", False, features

    package = joblib.load(model_path)

    if isinstance(package, dict):
        model = package["model"]
        scaler = package.get("scaler")
        feature_names = package.get("feature_names") or ADVISOR_FALLBACK_FEATURES
        label_map = package.get("label_map", {0: "Prolog", 1: "ACO", 2: "Genetic"})
    else:
        model = package
        scaler = None
        feature_names = ADVISOR_FALLBACK_FEATURES[:2]
        label_map = {0: "Prolog", 1: "ACO", 2: "Genetic"}

    feature_vector = [[features[name] for name in feature_names]]
    model_input = scaler.transform(feature_vector) if scaler is not None else feature_vector
    prediction = model.predict(model_input)[0]
    raw_label = label_map.get(int(prediction), prediction) if isinstance(label_map, dict) else prediction
    suggestion = _normalize_algorithm_label(raw_label)

    confidence_text = ""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        confidence = max(float(p) for p in probabilities)
        confidence_text = f" | confidence={confidence:.0%}"

    reason = (
        f"DecisionTree model: {os.path.basename(model_path)} "
        f"(features={len(feature_names)}{confidence_text})"
    )
    return suggestion, reason, True, features


# =============================================================================
# بخش ۱: Worker های QThread (الگوی Thread Worker)
# =============================================================================
#
# ──────────────────────────────────────────────────────────────────────────────
# الگوی طراحی QThread در این پروژه:
#
#   ۱. Worker  ← کلاس QObject که منطق محاسباتی را دارد
#   ۲. Thread  ← QThread که Worker را اجرا می‌کند
#   ۳. Signals ← پل ارتباطی بین Worker و GUI (thread-safe)
#
#   Worker.moveToThread(thread) → thread.start() → Worker.run()
#
# چرا QObject به جای subclass از QThread؟
#   چون اگر مستقیم از QThread ارث ببریم، Worker و Thread یکی میشن و
#   مدیریت lifecycle و سیگنال‌دهی پیچیده‌تر میشه. الگوی moveToThread()
#   پاک‌تر، قابل تست‌تر، و از نظر Qt توصیه‌شده‌ترین روشه.
# ──────────────────────────────────────────────────────────────────────────────


class DataFetchWorker(QObject):
    """
    Worker مسئول دریافت داده از API یا Cache (عملیات I/O سنگین).

    این Worker در یک QThread جداگانه اجرا میشه تا UI در حین دریافت
    اینترنتی یا خواندن فایل فریز نشه.

    سیگنال‌های خروجی:
      finished(cities, matrix) — داده‌ها آماده شدن
      error(message)           — خطا رخ داده
    """

    # سیگنال موفقیت: لیست شهرها + ماتریس فواصل
    finished = pyqtSignal(list, list)
    # سیگنال خطا: پیام خطا برای نمایش در UI
    error = pyqtSignal(str)
    # سیگنال وضعیت: پیام‌های مرحله‌به‌مرحله
    status = pyqtSignal(str)

    def __init__(self, country_name: str, num_cities: int, use_cache: bool = True):
        super().__init__()
        self.country_name = country_name
        self.num_cities   = num_cities
        self.use_cache    = use_cache

    @pyqtSlot()
    def run(self):
        """منطق اصلی دریافت داده — در thread پس‌زمینه اجرا میشه."""
        try:
            self.status.emit(f"در حال دریافت {self.num_cities} شهر از {self.country_name}...")
            logger.info(f"DataFetchWorker شروع: {self.country_name}, {self.num_cities} شهر")

            cities, matrix = get_tsp_data(
                country_name = self.country_name,
                num_cities   = self.num_cities,
                use_cache    = self.use_cache
            )

            self.status.emit(f"✅ {len(cities)} شهر آماده — ماتریس {len(cities)}×{len(cities)} ساخته شد.")
            logger.info("DataFetchWorker: داده با موفقیت دریافت شد.")

            # emit سیگنال موفقیت با داده‌ها
            self.finished.emit(cities, matrix)

        except Exception as e:
            error_msg = f"خطا در دریافت داده:\n{str(e)}"
            logger.error(f"DataFetchWorker خطا: {e}", exc_info=True)
            self.error.emit(error_msg)


class SolverWorker(QObject):
    """
    Worker مسئول اجرای یکی از الگوریتم‌های حل‌کننده (ACO / GA / Prolog).

    این کلاس هسته‌ی اصلی threading در پروژه است و باید با دقت درک بشه.

    ──────────────────────────────────────────────────────────────────────
    نقشه‌ی سیگنال‌ها:

      iteration_update(iter, path, cost)
        └── هر N تکرار یک‌بار emit میشه (توسط callback از داخل solver)
        └── GUI از این برای به‌روزرسانی انیمیشن زنده استفاده می‌کنه

      finished(SolverResult)
        └── یک‌بار در پایان emit میشه
        └── حامل نتیجه‌ی کامل (best_path, best_cost, history, elapsed)

      error(message)
        └── در صورت exception غیرمنتظره emit میشه
    ──────────────────────────────────────────────────────────────────────
    """

    # سیگنال به‌روزرسانی انیمیشن: (شماره تکرار, بهترین مسیر, بهترین هزینه)
    iteration_update = pyqtSignal(int, list, float)

    # سیگنال اتمام: نتیجه‌ی کامل الگوریتم
    finished = pyqtSignal(object)  # object چون SolverResult یک dataclass است

    # سیگنال پیشرفت: برای ProgressBar (۰ تا ۱۰۰)
    progress = pyqtSignal(int)

    # سیگنال خطا
    error = pyqtSignal(str)

    # ─────────────────────────────────────────────────────────────────
    # سیگنال جدید — پچ ادغام (فاز ۴ / main.py):
    # وقتی الگوریتم انتخابی «Prolog» باشد و pyswip وضعیتی غیر از
    # PrologStatus.SOLVED برگرداند (یعنی SKIPPED_SIZE یا
    # ENGINE_UNAVAILABLE یا RUNTIME_ERROR)، این سیگنال با همان
    # PrologSolverResult کامل (شامل ui_message و ui_tooltip) emit
    # می‌شود تا UI بتواند پیام آموزشی صحیح را نشان دهد — به‌جای
    # تبدیل فوری و بی‌صدا به SolverResult خام که این متادیتا را
    # دور می‌ریزد.
    # امضا: prolog_status(PrologSolverResult)
    # ─────────────────────────────────────────────────────────────────
    prolog_status = pyqtSignal(object)

    def __init__(
        self,
        algorithm:  str,             # "ACO", "Genetic", یا "Prolog"
        matrix:     list,            # ماتریس فواصل از core.py
        params:     dict,            # پارامترهای الگوریتم (از UI گرفته میشه)
    ):
        super().__init__()
        self.algorithm = algorithm
        self.matrix    = matrix
        self.params    = params

        # متغیر کنترلی برای توقف اجبار (Stop Button)
        self._stop_requested = False

    def request_stop(self):
        """
        درخواست توقف ظریف (Graceful Stop).

        این تابع از thread اصلی فراخوانی میشه. چون فقط یک bool
        تغییر می‌کنه و الگوریتم‌ها آن را در callback بررسی می‌کنند،
        thread-safe است (GIL از ما محافظت می‌کنه).
        """
        self._stop_requested = True
        logger.info(f"{self.algorithm} Worker: درخواست توقف دریافت شد.")

    def _make_callback(self, total_iters: int):
        """
        Factory: یک تابع callback برای پاس دادن به solver می‌سازه.

        این callback هر N تکرار یک‌بار توسط solver فراخوانی میشه.
        داخل callback، سیگنال‌های PyQt را emit می‌کنیم.

        نکته‌ی مهم: این callback داخل thread پس‌زمینه اجرا میشه،
        اما emit سیگنال در PyQt thread-safe است.
        """
        def callback(iteration: int, best_path: list, best_cost: float):
            if self._stop_requested:
                # توقف ظریف: یک exception خاص throw می‌کنیم که solver می‌گیرد
                raise InterruptedError("توقف توسط کاربر درخواست شده.")

            # emit سیگنال انیمیشن
            self.iteration_update.emit(iteration, best_path[:], best_cost)

            # محاسبه‌ی درصد پیشرفت
            if total_iters > 0:
                pct = int((iteration / total_iters) * 100)
                self.progress.emit(min(pct, 99))  # ۱۰۰٪ فقط در پایان

        return callback

    @pyqtSlot()
    def run(self):
        """
        منطق اصلی اجرای الگوریتم — در thread پس‌زمینه اجرا میشه.

        ساختار if/elif/else برای انتخاب الگوریتم آگاهانه است:
          - هر شاخه solver مناسب را می‌سازه
          - callback مناسب را به آن می‌دهد
          - نتیجه را در قالب SolverResult استاندارد تبدیل می‌کند
        """
        try:
            logger.info(f"SolverWorker شروع: الگوریتم={self.algorithm}")
            result: Optional[SolverResult] = None

            # ─── شاخه‌ی ACO ───
            if self.algorithm == "ACO":
                n_iter = self.params.get("n_iter", 200)
                aco = AntColonyOptimizer(
                    matrix     = self.matrix,
                    n_ants     = self.params.get("n_ants", 20),
                    n_iter     = n_iter,
                    alpha      = self.params.get("alpha", 1.0),
                    beta       = self.params.get("beta", 2.0),
                    rho        = self.params.get("rho", 0.1),
                    q_constant = self.params.get("q_constant", 100.0),
                    seed       = self.params.get("seed", 42),
                )
                callback = self._make_callback(total_iters=n_iter)
                result = aco.solve(
                    callback          = callback,
                    callback_interval = self.params.get("callback_interval", 5)
                )

            # ─── شاخه‌ی Genetic Algorithm ───
            elif self.algorithm == "Genetic":
                n_gen = self.params.get("n_generations", 300)
                ga = GeneticAlgorithmSolver(
                    matrix          = self.matrix,
                    pop_size        = self.params.get("pop_size", 80),
                    n_generations   = n_gen,
                    mutation_rate   = self.params.get("mutation_rate", 0.02),
                    tournament_size = self.params.get("tournament_size", 5),
                    elite_count     = self.params.get("elite_count", 2),
                    seed            = self.params.get("seed", 42),
                )
                callback = self._make_callback(total_iters=n_gen)
                result = ga.solve(
                    callback          = callback,
                    callback_interval = self.params.get("callback_interval", 5)
                )

            # ─── شاخه‌ی Prolog (Exact Solver) ───
            elif self.algorithm == "Prolog":
                # Prolog callback ندارد — یک‌مرحله‌ای حل می‌کند
                prolog_result: PrologSolverResult = solve_with_prolog(self.matrix)

                # ───────────────────────────────────────────────────
                # پچ ادغام (نکته‌ی ۴ کارفرما — رفع باگ نمایش پرولاگ):
                # قبل از این پچ، کد همیشه و بی‌قیدوشرط
                # prolog_result.to_solver_result() را صدا می‌زد، که
                # یعنی حالت‌های SKIPPED_SIZE و ENGINE_UNAVAILABLE و
                # RUNTIME_ERROR (که در آن‌ها best_path=[] است) بدون
                # هیچ توضیحی به GUI می‌رسیدند — کاربر فقط یک گراف
                # خالی/ناقص می‌دید و فکر می‌کرد یک باگ یا شهر گم‌شده
                # در محاسبه رخ داده. حالا اگر وضعیت دقیقاً SOLVED
                # نباشد، سیگنال prolog_status را با خودِ
                # PrologSolverResult کامل (همراه ui_message/ui_tooltip)
                # emit می‌کنیم تا UI پیام آموزشی صحیح (مثلاً «رد شد:
                # N > 15» یا «موتور SWI-Prolog در دسترس نیست») را نشان
                # دهد، سپس با یک return ساده از تابع خارج می‌شویم —
                # تا finished با یک نتیجه‌ی گمراه‌کننده (هزینه=۰، مسیر
                # خالی) emit نشود.
                # ───────────────────────────────────────────────────
                if prolog_result.status != PrologStatus.SOLVED:
                    logger.warning(
                        f"Prolog حل نشد ({prolog_result.status.value}) — "
                        f"emit سیگنال prolog_status برای نمایش پیام UI"
                    )
                    # توجه: عمداً self.error.emit(...) صدا زده نمی‌شود — چون
                    # _on_prolog_status در LiveSolverTab خودش UI (status_label
                    # + result_card + tooltip) را به‌طور کامل به‌روزرسانی
                    # می‌کند؛ emit کردن error هم باعث می‌شد _on_solver_error
                    # فوراً status_label را با یک پیام کلی‌تر و کم‌اطلاع‌تر
                    # رونویسی کند و پیام آموزشی دقیق prolog_bridge.py از
                    # دید کاربر پاک شود.
                    self.prolog_status.emit(prolog_result)
                    return

                # تبدیل به فرمت مشترک SolverResult — فقط وقتی status==SOLVED
                result = prolog_result.to_solver_result()

            else:
                raise ValueError(f"الگوریتم ناشناخته: {self.algorithm}")

            # ─── پایان موفق ───
            self.progress.emit(100)
            logger.info(f"SolverWorker تمام شد: هزینه={result.best_cost:.2f} km")
            self.finished.emit(result)

        except InterruptedError:
            # توقف توسط کاربر — این یک خطا نیست، طبیعی است
            logger.info(f"{self.algorithm} Worker: با موفقیت متوقف شد.")
            self.error.emit("⏹ توسط کاربر متوقف شد.")

        except Exception as e:
            error_msg = f"خطا در اجرای {self.algorithm}:\n{traceback.format_exc()}"
            logger.error(f"SolverWorker خطا: {e}", exc_info=True)
            self.error.emit(error_msg)


class BenchmarkWorker(QObject):
    """
    Worker مسئول اجرای بنچمارک کامل (هر سه الگوریتم به‌ترتیب).

    بنچمارک یعنی اجرای پشت‌سرهم ACO، GA و Prolog روی یک نقشه و مقایسه.
    این Worker هر الگوریتم را به ترتیب اجرا و نتایج را یک‌به‌یک گزارش می‌دهد.

    سیگنال‌ها:
      algo_done(name, result)  — هر بار یک الگوریتم تمام شد
      all_done(results_dict)   — وقتی همه‌ی سه الگوریتم تمام شدن
    """

    algo_done = pyqtSignal(str, object)  # (نام الگوریتم, SolverResult)
    all_done  = pyqtSignal(dict)         # {"ACO": result, "Genetic": result, ...}
    progress  = pyqtSignal(int)          # 0 تا 100
    error     = pyqtSignal(str)

    # ─────────────────────────────────────────────────────────────────
    # سیگنال جدید — پچ ادغام (نکته‌ی ۲ کارفرما — Benchmark Tab):
    # وقتی اجرای پرولاگ در بنچمارک با status != SOLVED برگردد (یعنی
    # SKIPPED_SIZE یا ENGINE_UNAVAILABLE یا RUNTIME_ERROR)، این سیگنال
    # با خودِ PrologSolverResult کامل emit می‌شود تا BenchmarkTab بتواند
    # به‌جای یک ردیف گمراه‌کننده‌ی «Prolog: 0.00 km» در جدول نتایج،
    # پیام و tooltip دقیق prolog_bridge.py را نشان دهد.
    # امضا: prolog_status(PrologSolverResult)
    # ─────────────────────────────────────────────────────────────────
    prolog_status = pyqtSignal(object)

    def __init__(self, matrix: list, benchmark_params: dict):
        super().__init__()
        self.matrix           = matrix
        self.benchmark_params = benchmark_params  # پارامترهای سریع برای بنچمارک

    @pyqtSlot()
    def run(self):
        """اجرای ترتیبی هر سه الگوریتم."""
        results = {}

        # پارامترهای سبک برای بنچمارک (کمتر از حالت معمولی تا سریع‌تر باشه)
        benchmark_aco_params = {
            "n_ants": 15, "n_iter": 100, "alpha": 1.0,
            "beta": 2.0, "rho": 0.1, "q_constant": 100.0, "seed": 0
        }
        benchmark_ga_params = {
            "pop_size": 50, "n_generations": 150, "mutation_rate": 0.02,
            "tournament_size": 5, "elite_count": 2, "seed": 0
        }

        algorithms = [
            ("ACO",     AntColonyOptimizer,    benchmark_aco_params),
            ("Genetic", GeneticAlgorithmSolver, benchmark_ga_params),
        ]

        try:
            # ─── اجرای ACO و GA ───
            for i, (name, SolverClass, params) in enumerate(algorithms):
                self.progress.emit(int(i * 33))
                logger.info(f"BenchmarkWorker: اجرای {name}...")

                if name == "ACO":
                    solver = SolverClass(matrix=self.matrix, **{
                        k: v for k, v in params.items()
                        if k in ["n_ants", "n_iter", "alpha", "beta", "rho", "q_constant", "seed"]
                    })
                else:
                    solver = SolverClass(matrix=self.matrix, **{
                        k: v for k, v in params.items()
                        if k in ["pop_size", "n_generations", "mutation_rate", "tournament_size", "elite_count", "seed"]
                    })

                result = solver.solve(callback=None)
                results[name] = result
                self.algo_done.emit(name, result)

            # ─── اجرای Prolog ───
            self.progress.emit(66)
            logger.info("BenchmarkWorker: اجرای Prolog...")
            prolog_res = solve_with_prolog(self.matrix)

            # ───────────────────────────────────────────────────────
            # پچ ادغام: قبل از تبدیل بی‌قیدوشرط به SolverResult، وضعیت
            # را چک می‌کنیم. اگر SOLVED نباشد (یعنی N > 15 یا موتور
            # پرولاگ در دسترس نیست)، سیگنال prolog_status را با
            # PrologSolverResult کامل (ui_message/ui_tooltip) emit
            # می‌کنیم تا BenchmarkTab پیام دقیق را در جدول نشان دهد،
            # و یک SolverResult «جای‌گذار» با best_cost=NaN می‌سازیم
            # (نه ۰.۰) تا در نمودارهای میله‌ای Cost/Time به‌اشتباه به
            # چشم نیاد که «پرولاگ سریع‌ترین با کمترین هزینه بوده».
            # ───────────────────────────────────────────────────────
            if prolog_res.status != PrologStatus.SOLVED:
                logger.warning(
                    f"BenchmarkWorker: Prolog حل نشد ({prolog_res.status.value})"
                )
                self.prolog_status.emit(prolog_res)
                # نتیجه‌ی جای‌گذار — برای جدول/نمودار بنچمارک، بدون
                # گمراه کردن کاربر با هزینه‌ی صفر
                prolog_solver_result = SolverResult(
                    algorithm   = "Prolog (Held-Karp)",
                    best_path   = [],
                    best_cost   = float("nan"),
                    history     = [],
                    elapsed_sec = prolog_res.elapsed_sec,
                    iterations  = 0
                )
            else:
                prolog_solver_result = prolog_res.to_solver_result()

            results["Prolog"] = prolog_solver_result
            self.algo_done.emit("Prolog", prolog_solver_result)

            self.progress.emit(100)
            self.all_done.emit(results)

        except Exception as e:
            logger.error(f"BenchmarkWorker خطا: {e}", exc_info=True)
            self.error.emit(f"خطا در بنچمارک:\n{traceback.format_exc()}")


# =============================================================================
# بخش ۲: Widget کمکی برای نمودارهای Matplotlib
# =============================================================================

class MplCanvas(FigureCanvas):
    """
    Widget سفارشی برای تعبیه‌ی نمودار Matplotlib درون PyQt6.

    نکته‌ی مهم RTL:
      متن فارسی در Matplotlib به‌صورت بومی پشتیبانی نمی‌شود.
      راه‌حل: تمام تیترها/برچسب‌های Matplotlib به انگلیسی نوشته میشن
      و فقط annotation های شهر (با usetex=False) فارسی هستند.
      برای placeholder ها از ایموجی + انگلیسی استفاده می‌کنیم.
    """

    def __init__(
        self,
        width: float = 5.0,
        height: float = 4.0,
        dpi: int = 96,
        nrows: int = 1,
        ncols: int = 1,
        bg: str = None,          # رنگ پس‌زمینه‌ی figure (اختیاری)
    ):
        fig_bg  = bg or PALETTE["bg_mid"]
        ax_bg   = PALETTE["bg_dark"]

        self.fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(width, height), dpi=dpi,
            facecolor=fig_bg
        )

        if nrows == 1 and ncols == 1:
            self.axes = [axes]
        elif nrows == 1 or ncols == 1:
            self.axes = list(axes)
        else:
            self.axes = [ax for row in axes for ax in row]

        self.ax = self.axes[0]

        for ax in self.axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=PALETTE["text_dim"], labelsize=8)
            ax.xaxis.label.set_color(PALETTE["text_dim"])
            ax.yaxis.label.set_color(PALETTE["text_dim"])
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["border"])
                spine.set_linewidth(0.8)

        self.fig.tight_layout(pad=1.5)

        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

    def style_ax(self, ax, title: str = "", xlabel: str = "", ylabel: str = "",
                 accent: str = None):
        """
        اعمال استایل یکسان روی یک محور.
        تیتر و برچسب‌ها باید انگلیسی باشند (محدودیت Matplotlib + RTL).
        """
        c = accent or PALETTE["accent"]
        if title:
            ax.set_title(title, color=c, fontsize=10, pad=8, fontfamily=MPL_FONT)
        if xlabel:
            ax.set_xlabel(xlabel, color=PALETTE["text_dim"], fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color=PALETTE["text_dim"], fontsize=8)
        ax.set_facecolor(PALETTE["bg_dark"])
        ax.tick_params(colors=PALETTE["text_dim"], labelsize=8)
        ax.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.25, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])

    def clear_all(self):
        for ax in self.axes:
            ax.cla()
            ax.set_facecolor(PALETTE["bg_dark"])


# =============================================================================
# بخش ۳: تب اول — پیکربندی داده (ConfigTab)
# =============================================================================

class ConfigTab(QWidget):
    """
    تب تنظیمات: ورودی کشور/شهر و نمایش اولیه‌ی گراف.

    مسئولیت‌ها:
      - دریافت نام کشور و تعداد شهر از کاربر
      - نمایش لیست dataset های کَش‌شده (حالت آفلاین)
      - لود داده از طریق DataFetchWorker
      - نمایش نقاط شهرها روی نمودار Matplotlib
      - انتقال (cities, matrix) به MainWindow برای استفاده در تب‌های بعدی

    سیگنال‌های خروجی:
      data_ready(cities, matrix) — وقتی داده با موفقیت آماده شد
    """

    # سیگنال خروجی به MainWindow
    data_ready = pyqtSignal(list, list)

    def __init__(self, parent=None):
        super().__init__(parent)

        # وضعیت داده‌ی فعلی
        self._cities: Optional[list] = None
        self._matrix: Optional[list] = None

        # Worker و Thread — None در حالت idle
        self._worker: Optional[DataFetchWorker] = None
        self._thread: Optional[QThread]         = None

        self._build_ui()

    def _build_ui(self):
        """ساخت layout و widget های تب پیکربندی."""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ─── پنل چپ: کنترل‌ها (با رنگ کمی متفاوت) ───
        control_wrapper = QWidget()
        control_wrapper.setFixedWidth(290)
        control_wrapper.setStyleSheet(
            f"background: {PALETTE['bg_mid']}; "
            f"border-right: 1px solid {TAB_ACCENTS[0]['hex_dim']};"
        )
        cw_layout = QVBoxLayout(control_wrapper)
        cw_layout.setContentsMargins(14, 14, 14, 14)
        cw_layout.setSpacing(0)

        # هدر پنل چپ
        panel_header = QLabel("⚙  Data Configuration")
        panel_header.setStyleSheet(
            f"color: {TAB_ACCENTS[0]['color']}; font-size: 13px; font-weight: bold; "
            f"padding-bottom: 10px; border-bottom: 1px solid {TAB_ACCENTS[0]['hex_dim']}; "
            f"margin-bottom: 12px;"
        )
        cw_layout.addWidget(panel_header)

        control_panel = self._build_control_panel()
        cw_layout.addWidget(control_panel)
        cw_layout.addStretch()
        main_layout.addWidget(control_wrapper)

        # ─── پنل راست: نمودار ───
        map_wrapper = QWidget()
        map_wrapper.setStyleSheet(f"background: {PALETTE['bg_dark']};")
        mw_layout = QVBoxLayout(map_wrapper)
        mw_layout.setContentsMargins(16, 16, 16, 16)

        self.map_canvas = MplCanvas(width=7, height=5.5, bg=PALETTE["bg_dark"])
        self._draw_placeholder()
        mw_layout.addWidget(self.map_canvas)
        main_layout.addWidget(map_wrapper, stretch=1)

    def _build_control_panel(self) -> QWidget:
        """ساخت پنل کنترل‌های سمت چپ."""
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── گروه حالت آنلاین ──
        online_group = QGroupBox("🌐  Online — Fetch from API")
        online_group.setStyleSheet(_tab_group_style(0))
        online_layout = QVBoxLayout(online_group)
        online_layout.setSpacing(8)

        lbl_country = QLabel("Country name (English):")
        lbl_country.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 11px; background: transparent;")
        online_layout.addWidget(lbl_country)

        self.country_input = QComboBox()
        self.country_input.setEditable(True)
        self.country_input.addItems([
            "Iran", "Italy", "France", "Germany", "Japan",
            "Brazil", "United States", "India", "Australia", "Spain"
        ])
        self.country_input.setCurrentText("Iran")
        online_layout.addWidget(self.country_input)

        lbl_cities = QLabel(f"Number of cities ({MIN_CITIES}–{MAX_CITIES}):")
        lbl_cities.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 11px; background: transparent;")
        online_layout.addWidget(lbl_cities)

        self.city_count_spin = QSpinBox()
        self.city_count_spin.setRange(MIN_CITIES, MAX_CITIES)
        self.city_count_spin.setValue(10)
        online_layout.addWidget(self.city_count_spin)

        self.fetch_btn = QPushButton("🌍  Fetch from API")
        self.fetch_btn.clicked.connect(self._on_fetch_online)
        self.fetch_btn.setStyleSheet(_primary_btn(0))
        online_layout.addWidget(self.fetch_btn)
        layout.addWidget(online_group)

        # ── گروه حالت آفلاین (کَش) ──
        offline_group = QGroupBox("📂  Offline — Load from Cache")
        offline_group.setStyleSheet(_tab_group_style(0))
        offline_layout = QVBoxLayout(offline_group)
        offline_layout.setSpacing(8)

        lbl_cache = QLabel("Saved datasets:")
        lbl_cache.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 11px; background: transparent;")
        offline_layout.addWidget(lbl_cache)

        self.cache_combo = QComboBox()
        self._refresh_cache_list()
        offline_layout.addWidget(self.cache_combo)

        self.load_cache_btn = QPushButton("📁  Load from Cache")
        self.load_cache_btn.clicked.connect(self._on_load_cache)
        self.load_cache_btn.setStyleSheet(_secondary_btn(0))
        offline_layout.addWidget(self.load_cache_btn)
        layout.addWidget(offline_group)

        # ── وضعیت ──
        self.status_label = QLabel("Ready — select a dataset.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            f"color: {PALETTE['text_dim']}; font-size: 11px; "
            f"padding: 6px; background: transparent;"
        )
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {PALETTE["bg_light"]}; border: 1px solid {TAB_ACCENTS[0]["hex_dim"]};
                border-radius: 4px; height: 6px;
            }}
            QProgressBar::chunk {{
                background: {TAB_ACCENTS[0]["color"]}; border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            f"color: {TAB_ACCENTS[0]['color']}; font-size: 11px; "
            f"border: 1px solid {TAB_ACCENTS[0]['hex_dim']}; "
            f"background: {TAB_ACCENTS[0]['hex_dim']}; "
            f"padding: 8px; border-radius: 6px;"
        )
        self.info_label.setVisible(False)
        layout.addWidget(self.info_label)

        return panel

    def _refresh_cache_list(self):
        """لیست dataset های کَش‌شده را به‌روز می‌کند."""
        self.cache_combo.clear()
        datasets = list_cached_datasets()
        if datasets:
            for ds in datasets:
                label = f"{ds['country']} — {ds['num_cities']} شهر"
                self.cache_combo.addItem(label, userData=ds)
        else:
            self.cache_combo.addItem("کَشی یافت نشد")

    def _on_fetch_online(self):
        """کاربر دکمه‌ی دریافت آنلاین را زد — شروع DataFetchWorker."""
        country = self.country_input.currentText().strip()
        n_cities = self.city_count_spin.value()

        if not country:
            self.status_label.setText("⚠️ نام کشور را وارد کنید.")
            return

        self._start_fetch(country, n_cities, use_cache=True)

    def _on_load_cache(self):
        """کاربر دکمه‌ی لود از کَش را زد."""
        current_data = self.cache_combo.currentData()
        if not current_data:
            self.status_label.setText("⚠️ هیچ dataset کَشی موجود نیست.")
            return

        country  = current_data["country"]
        n_cities = current_data["num_cities"]
        # use_cache=True یعنی core.py فقط از فایل محلی می‌خواند
        self._start_fetch(country, n_cities, use_cache=True)

    def _start_fetch(self, country: str, n_cities: int, use_cache: bool):
        """
        ساخت Worker + Thread و شروع عملیات دریافت داده.

        این تابع الگوی QThread را به‌طور کامل نمایش می‌دهد.
        """
        # جلوگیری از اجرای همزمان دو Worker
        if self._thread and self._thread.isRunning():
            self.status_label.setText("⚠️ عملیات قبلی هنوز در حال اجراست.")
            return

        self._set_loading_state(True)

        # ─── ساخت Worker (QObject) ───
        self._worker = DataFetchWorker(country, n_cities, use_cache)

        # ─── ساخت Thread ───
        self._thread = QThread()

        # ─── انتقال Worker به Thread (مهم‌ترین قدم!) ───
        self._worker.moveToThread(self._thread)

        # ─── اتصال سیگنال‌ها ───
        # وقتی thread شروع میشه، run() اجرا بشه
        self._thread.started.connect(self._worker.run)

        # وقتی Worker کارش تموم شد:
        self._worker.finished.connect(self._on_fetch_finished)
        self._worker.error.connect(self._on_fetch_error)
        self._worker.status.connect(self.status_label.setText)

        # پاک‌سازی بعد از اتمام
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        # ─── شروع Thread ───
        self._thread.start()
        logger.info(f"Thread دریافت داده شروع شد: {country}, {n_cities} شهر")

    @pyqtSlot(list, list)
    def _on_fetch_finished(self, cities: list, matrix: list):
        """وقتی DataFetchWorker با موفقیت تمام شد."""
        self._cities = cities
        self._matrix = matrix
        self._set_loading_state(False)

        # به‌روزرسانی اطلاعات
        info_text = (
            f"✅ {len(cities)} شهر بارگذاری شد\n"
            f"ماتریس: {len(cities)}×{len(cities)}\n"
            f"کشور: {cities[0].name if cities else '؟'}"
        )
        self.info_label.setText(info_text)
        self.info_label.setVisible(True)

        # نمایش روی نمودار
        self._draw_city_map(cities)

        # به‌روزرسانی لیست کَش
        self._refresh_cache_list()

        # انتقال داده به MainWindow از طریق سیگنال
        self.data_ready.emit(cities, matrix)

    @pyqtSlot(str)
    def _on_fetch_error(self, error_msg: str):
        """وقتی DataFetchWorker با خطا مواجه شد."""
        self._set_loading_state(False)
        self.status_label.setText(f"❌ {error_msg}")
        logger.error(f"ConfigTab خطا: {error_msg}")

    def _set_loading_state(self, is_loading: bool):
        """UI را در حالت بارگذاری یا آماده قرار می‌دهد."""
        self.fetch_btn.setEnabled(not is_loading)
        self.load_cache_btn.setEnabled(not is_loading)
        self.progress_bar.setVisible(is_loading)

        if not is_loading:
            self.status_label.setText("عملیات تمام شد.")

    def _draw_placeholder(self):
        """نمایش پیام راهنما در نمودار خالی — متن انگلیسی برای جلوگیری از مشکل RTL."""
        ax = self.map_canvas.ax
        ax.cla()
        ax.set_facecolor(PALETTE["bg_dark"])
        ax.set_xticks([])
        ax.set_yticks([])

        # دایره‌ی نقطه‌چین تزئینی
        circle = plt.Circle(
            (0.5, 0.5), 0.28,
            fill=False, color=TAB_ACCENTS[0]["hex_dim"],
            linestyle="--", linewidth=1.5,
            transform=ax.transAxes
        )
        ax.add_patch(circle)

        # متن مرکزی — انگلیسی تا RTL مشکل نداشته باشیم
        ax.text(
            0.5, 0.54, "🗺",
            ha="center", va="center", fontsize=36,
            transform=ax.transAxes
        )
        ax.text(
            0.5, 0.40,
            "Select a dataset to\nview the city map",
            ha="center", va="center",
            fontsize=12, color=PALETTE["text_dim"],
            transform=ax.transAxes,
            linespacing=1.6
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(TAB_ACCENTS[0]["hex_dim"])

        self.map_canvas.fig.tight_layout(pad=1.5)
        self.map_canvas.draw()

    def _draw_city_map(self, cities: list):
        """
        نمایش نقاط شهرها روی نمودار.
        
        مهم — رفع RTL:
          تیتر و برچسب‌های محور انگلیسی هستند.
          نام شهرها فارسی هستند و با annotate نمایش داده می‌شوند؛
          چون Matplotlib متن Unicode را رندر می‌کند، نام‌ها نمایش داده می‌شوند
          اما ممکن است ترتیب حروف RTL کامل نباشد — این رفتار طبیعی است.
        """
        ax = self.map_canvas.ax
        ax.cla()
        ax.set_facecolor(PALETTE["bg_dark"])

        lons = [c.lon for c in cities]
        lats = [c.lat for c in cities]

        # شبکه‌ی پس‌زمینه
        ax.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.2, linewidth=0.7)

        # رسم یک هاله‌ی ظریف پشت نقاط
        ax.scatter(lons, lats, c=TAB_ACCENTS[0]["hex_dim"], s=200,
                   zorder=2, alpha=0.4)
        # نقاط اصلی
        ax.scatter(lons, lats, c=TAB_ACCENTS[0]["color"], s=60,
                   zorder=3, alpha=0.95, linewidths=0.5, edgecolors="white")

        # نمایش نام شهرها — از annotate با fontfamily استاندارد استفاده می‌کنیم
        for city in cities:
            ax.annotate(
                city.name,
                xy=(city.lon, city.lat),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
                color=PALETTE["text_dim"],
                fontfamily=MPL_FONT
            )

        # تیتر و برچسب‌ها — انگلیسی
        self.map_canvas.style_ax(
            ax,
            title=f"City Map — {len(cities)} cities loaded",
            xlabel="Longitude",
            ylabel="Latitude",
            accent=TAB_ACCENTS[0]["color"]
        )

        self.map_canvas.fig.tight_layout(pad=1.5)
        self.map_canvas.draw()

    # ─── متدهای استایل ConfigTab — از توابع سراسری استفاده می‌کنیم ───

    def _group_style(self) -> str:
        return _tab_group_style(0)

    def _primary_btn_style(self) -> str:
        return _primary_btn(0)

    def _secondary_btn_style(self) -> str:
        return _secondary_btn(0)


# =============================================================================
# بخش ۴: تب دوم — مشاور هوشمند (AIAdvisorTab)
# =============================================================================

class AIAdvisorTab(QWidget):
    """
    تب مشاور هوشمند: پیشنهاد ML + انتخاب الگوریتم + شروع اجرا.

    مسئولیت‌ها:
      - گرفتن ماتریس از ConfigTab
      - استخراج ویژگی‌ها و پیشنهاد الگوریتم توسط Decision Tree
      - نمایش توضیح نقش پرولاگ
      - فراهم کردن کنترل‌های انتخاب پارامتر
      - ارسال سیگنال شروع حل به LiveSolverTab

    سیگنال‌های خروجی:
      solve_requested(algorithm, params) — کاربر دکمه‌ی حل را زد
    """

    solve_requested = pyqtSignal(str, dict)  # (نام الگوریتم, پارامترها)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._matrix: Optional[list] = None
        self._cities: Optional[list] = None
        self._ml_model = None           # مدل pkl بارگذاری‌شده
        self._build_ui()

    def update_data(self, cities: list, matrix: list):
        """
        دریافت داده‌ی جدید از ConfigTab (وقتی کاربر dataset انتخاب کرد).
        از MainWindow فراخوانی می‌شود.
        """
        self._cities = cities
        self._matrix = matrix
        self._run_ml_advisor()

    def _build_ui(self):
        """ساخت layout تب مشاور — تم بنفش."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── هدر رنگی تب (نوار بنفش بالا) ──
        header_bar = QWidget()
        header_bar.setFixedHeight(52)
        header_bar.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {TAB_ACCENTS[1]['hex_dim']}, stop:1 {PALETTE['bg_mid']});"
            f"border-bottom: 2px solid {TAB_ACCENTS[1]['color']};"
        )
        hb_layout = QHBoxLayout(header_bar)
        hb_layout.setContentsMargins(18, 0, 18, 0)

        title_lbl = QLabel("🤖  AI Advisor — Algorithm Recommendation Engine")
        title_lbl.setStyleSheet(
            f"color: {TAB_ACCENTS[1]['color']}; font-size: 14px; font-weight: bold; "
            f"background: transparent;"
        )
        hb_layout.addWidget(title_lbl)
        hb_layout.addStretch()

        self.start_btn = QPushButton("🚀  Solve the Problem")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.start_btn.setFixedHeight(34)
        self.start_btn.setStyleSheet(_primary_btn(1))
        hb_layout.addWidget(self.start_btn)

        main_layout.addWidget(header_bar)

        # ── محتوای اصلی ──
        content = QWidget()
        content.setStyleSheet(f"background: {PALETTE['bg_dark']};")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(14)

        # ───────────────────────────────────────────
        # ستون چپ: ML Suggestion + Feature Chart
        # ───────────────────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # کارت پیشنهاد ML
        ml_card = QGroupBox("💡  Decision Tree Suggestion")
        ml_card.setStyleSheet(_tab_group_style(1))
        ml_layout = QVBoxLayout(ml_card)
        ml_layout.setSpacing(8)

        self.ml_suggestion_label = QLabel(
            "Waiting for dataset...\n\n"
            "Once a map is loaded, the ML model will\n"
            "automatically recommend the best algorithm."
        )
        self.ml_suggestion_label.setWordWrap(True)
        self.ml_suggestion_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ml_suggestion_label.setStyleSheet(
            f"color: {PALETTE['text_main']}; font-size: 12px; "
            f"background: {PALETTE['bg_mid']}; padding: 12px; border-radius: 6px; "
            f"min-height: 85px; line-height: 1.5;"
        )
        ml_layout.addWidget(self.ml_suggestion_label)

        # نمودار ویژگی‌ها
        self.feature_canvas = MplCanvas(width=4.0, height=2.2,
                                        bg=TAB_ACCENTS[1]["hex_dim"])
        self.feature_canvas.ax.text(
            0.5, 0.5, "Graph features\nloaded after dataset",
            ha="center", va="center", color=PALETTE["text_dim"],
            transform=self.feature_canvas.ax.transAxes, fontsize=9
        )
        ml_layout.addWidget(self.feature_canvas)

        left_col.addWidget(ml_card)

        # ── انتخاب الگوریتم ──
        algo_card = QGroupBox("⚙️  Select Algorithm")
        algo_card.setStyleSheet(_tab_group_style(1))
        algo_layout = QVBoxLayout(algo_card)
        algo_layout.setSpacing(6)

        self.algo_button_group = QButtonGroup()
        algo_options = [
            ("ACO",     "🐜  Ant Colony (ACO)",      "Metaheuristic — fast and scalable"),
            ("Genetic", "🧬  Genetic Algorithm (ERX)","Metaheuristic — advanced ERX crossover"),
            ("Prolog",  f"🔮  Prolog (Held-Karp)",    f"Exact solver — up to N={PROLOG_MAX_CITIES} cities only"),
        ]

        self.algo_radios = {}
        for value, label, tooltip in algo_options:
            rb = QRadioButton(label)
            rb.setToolTip(tooltip)
            c = ALGO_COLORS.get(value, TAB_ACCENTS[1]["color"])
            rb.setStyleSheet(f"""
                QRadioButton {{
                    color: {PALETTE['text_main']}; font-size: 12px;
                    spacing: 8px; padding: 5px 8px;
                    border-radius: 4px;
                }}
                QRadioButton:checked {{
                    background: {TAB_ACCENTS[1]['hex_dim']};
                    color: {c};
                    font-weight: bold;
                }}
                QRadioButton::indicator {{ width: 14px; height: 14px; border-radius: 7px;
                    border: 2px solid {PALETTE['border']}; }}
                QRadioButton::indicator:checked {{
                    background: {c}; border-color: {c}; }}
            """)
            self.algo_button_group.addButton(rb)
            algo_layout.addWidget(rb)
            self.algo_radios[value] = rb

        self.algo_radios["ACO"].setChecked(True)
        left_col.addWidget(algo_card)

        content_layout.addLayout(left_col, stretch=5)

        # ───────────────────────────────────────────
        # ستون راست: پارامترها + پرولاگ (فشرده)
        # ───────────────────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # پارامترها
        param_card = QGroupBox("🎛️  Parameters")
        param_card.setStyleSheet(_tab_group_style(1))
        param_grid = QGridLayout(param_card)
        param_grid.setSpacing(8)

        def _lbl(txt):
            l = QLabel(txt)
            l.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 11px; background: transparent;")
            return l

        param_grid.addWidget(_lbl("Iterations:"), 0, 0)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(50, 1000)
        self.iter_spin.setValue(200)
        self.iter_spin.setSingleStep(50)
        param_grid.addWidget(self.iter_spin, 0, 1)

        param_grid.addWidget(_lbl("Population / Ants:"), 1, 0)
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(10, 200)
        self.pop_spin.setValue(20)
        param_grid.addWidget(self.pop_spin, 1, 1)

        right_col.addWidget(param_card)

        # ── بخش پرولاگ — فشرده (ارتفاع ثابت و اسکرول) ──
        prolog_card = QGroupBox("🔷  Prolog Role (Ground Truth)")
        prolog_card.setStyleSheet(_tab_group_style(1))
        prolog_inner = QVBoxLayout(prolog_card)
        prolog_inner.setSpacing(4)

        prolog_text = QTextEdit()
        prolog_text.setPlainText(get_prolog_role_explanation())
        prolog_text.setReadOnly(True)
        prolog_text.setFixedHeight(140)   # ارتفاع ثابت + اسکرول داخلی
        prolog_text.setStyleSheet(f"""
            QTextEdit {{
                background: {PALETTE['bg_mid']}; color: {PALETTE['text_dim']};
                border: none; font-size: 10.5px; line-height: 1.6;
                padding: 6px;
            }}
            QScrollBar:vertical {{ width: 6px; background: {PALETTE['bg_dark']}; }}
            QScrollBar::handle:vertical {{ background: {PALETTE['border']}; border-radius: 3px; }}
        """)
        prolog_inner.addWidget(prolog_text)
        right_col.addWidget(prolog_card)

        right_col.addStretch()
        content_layout.addLayout(right_col, stretch=4)

        main_layout.addWidget(content, stretch=1)

    def _run_ml_advisor(self):
        """
        اجرای مدل ML برای پیشنهاد الگوریتم.

        اگه فایل pkl موجود بود، مدل لود و پیشنهاد داده میشه.
        اگه نه، یک heuristic ساده بر اساس N استفاده میشه.
        """
        if not self._matrix:
            return

        n = len(self._matrix)

        # ─── تلاش برای استفاده از مدل ML ───
        suggestion = ""
        reason = ""
        ml_used = False
        features = _extract_advisor_features(self._matrix, self._cities or [])

        try:
            suggestion, reason, ml_used, features = _predict_advisor_algorithm(
                self._matrix,
                self._cities or []
            )
        except Exception as e:
            reason = f"مدل ML لود نشد: {e}"
            logger.warning(reason, exc_info=True)

        # ─── Heuristic ساده اگه مدل نبود ───
        if not ml_used:
            if n <= PROLOG_MAX_CITIES:
                suggestion = "Prolog"
                reason = f"N={n} ≤ {PROLOG_MAX_CITIES}: پرولاگ جواب دقیق ۱۰۰٪ بهینه می‌دهد"
            elif n <= 40:
                suggestion = "ACO"
                reason = f"N={n}: مورچگان برای این اندازه همگرایی سریع دارد"
            else:
                suggestion = "Genetic"
                reason = f"N={n}: ژنتیک با ERX برای نقشه‌های بزرگ‌تر مناسب‌تر است"

        if suggestion == "Prolog" and n > PROLOG_MAX_CITIES:
            suggestion = "ACO" if n <= 40 else "Genetic"
            reason += f" | Prolog skipped because N>{PROLOG_MAX_CITIES}"

        distance_variance = features.get("dist_std", 0.0) ** 2

        # نمایش پیشنهاد — متن انگلیسی برای جلوگیری از RTL در QLabel
        algo_emoji = {"Prolog": "🔮", "ACO": "🐜", "Genetic": "🧬"}
        algo_color = ALGO_COLORS.get(suggestion, TAB_ACCENTS[1]["color"])
        text = (
            f"Recommended:  {algo_emoji.get(suggestion, '')}  {suggestion}\n\n"
            f"Reason:  {reason}\n\n"
            f"Graph size:  N = {n} cities\n"
            f"Distance variance:  {distance_variance:,.0f} km²\n"
            f"{'[ML model used]' if ml_used else '[Simple heuristic]'}"
        )
        self.ml_suggestion_label.setText(text)
        self.ml_suggestion_label.setStyleSheet(
            f"color: {algo_color}; font-size: 12px; font-weight: bold; "
            f"background: {PALETTE['bg_mid']}; padding: 12px; border-radius: 6px; "
            f"min-height: 85px; border-left: 3px solid {algo_color};"
        )

        # تنظیم radio button پیشنهادی
        if suggestion in self.algo_radios:
            self.algo_radios[suggestion].setChecked(True)

        # فعال‌سازی دکمه‌ی شروع
        self.start_btn.setEnabled(True)

        # رسم نمودار ویژگی‌ها
        self._draw_feature_chart(n, distance_variance)

    def _draw_feature_chart(self, n: int, variance: float):
        """رسم نمودار میله‌ای ویژگی‌های گراف — برچسب‌ها انگلیسی."""
        ax = self.feature_canvas.ax
        ax.cla()
        ax.set_facecolor(PALETTE["bg_dark"])

        labels = ["N (cities)", "Variance / 1k"]
        values = [n, variance / 1000]
        bar_colors = [TAB_ACCENTS[1]["color"], TAB_ACCENTS[2]["color"]]

        bars = ax.bar(labels, values, color=bar_colors, width=0.45, alpha=0.85)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.03,
                f"{val:.1f}", ha="center", va="bottom",
                color=PALETTE["text_main"], fontsize=9, fontweight="bold"
            )

        self.feature_canvas.style_ax(
            ax, title="Graph Features",
            accent=TAB_ACCENTS[1]["color"]
        )
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
        self.feature_canvas.fig.tight_layout(pad=1.0)
        self.feature_canvas.draw()

    def _on_start_clicked(self):
        """کاربر دکمه‌ی شروع را زد — ارسال سیگنال به LiveSolverTab."""
        # تشخیص الگوریتم انتخابی
        selected_algo = "ACO"
        for name, rb in self.algo_radios.items():
            if rb.isChecked():
                selected_algo = name
                break

        # جمع‌آوری پارامترها از UI
        n_iter = self.iter_spin.value()
        pop_n  = self.pop_spin.value()

        params = {
            # پارامترهای ACO
            "n_ants":    pop_n,
            "n_iter":    n_iter,
            "alpha":     1.0,
            "beta":      2.0,
            "rho":       0.1,
            "q_constant": 100.0,
            # پارامترهای GA
            "pop_size":       pop_n,
            "n_generations":  n_iter,
            "mutation_rate":  0.02,
            "tournament_size": 5,
            "elite_count":    2,
            # مشترک
            "seed":              42,
            "callback_interval": 5,
        }

        logger.info(f"AIAdvisorTab: ارسال درخواست حل — الگوریتم={selected_algo}")
        self.solve_requested.emit(selected_algo, params)

    def _group_style(self) -> str:
        return _tab_group_style(1)


# =============================================================================
# بخش ۵: تب سوم — انیمیشن زنده (LiveSolverTab)
# =============================================================================

class LiveSolverTab(QWidget):
    """
    تب انیمیشن زنده: اجرای الگوریتم در QThread + نمایش گام‌به‌گام.

    این تب قلب پروژه از نظر UX است.
    هر بار که callback از داخل solver فراخوانی می‌شود:
      solver callback → SolverWorker.iteration_update signal → _on_iteration_update() → _draw_path()

    مسئولیت‌ها:
      - دریافت (algorithm, params, matrix, cities) از MainWindow
      - ساخت و مدیریت SolverWorker + QThread
      - رسم زنده‌ی مسیر روی نمودار هر N تکرار
      - رسم نمودار همگرایی (history) در پایان
      - نمایش نتیجه‌ی نهایی
    """

    # وقتی حل تمام شد، نتیجه را به BenchmarkTab هم می‌فرستیم
    result_ready = pyqtSignal(str, object)  # (نام الگوریتم, SolverResult)

    def __init__(self, parent=None):
        super().__init__(parent)

        # داده
        self._cities: Optional[list]  = None
        self._matrix: Optional[list]  = None
        self._algorithm: str          = "ACO"
        self._params: dict            = {}

        # نتیجه‌ی فعلی
        self._current_result: Optional[SolverResult] = None

        # Worker / Thread
        self._worker: Optional[SolverWorker] = None
        self._thread: Optional[QThread]      = None

        self._build_ui()

    def prepare_solve(self, algorithm: str, params: dict, cities: list, matrix: list):
        """آماده‌سازی برای حل — از MainWindow فراخوانی می‌شود."""
        self._algorithm = algorithm
        self._params    = params
        self._cities    = cities
        self._matrix    = matrix

        algo_color = ALGO_COLORS.get(algorithm, TAB_ACCENTS[2]["color"])
        self._reset_ui()
        self.algo_label.setText(f"Live Solver  —  {algorithm}")
        self.algo_label.setStyleSheet(
            f"color: {algo_color}; font-size: 13px; font-weight: bold; background: transparent;"
        )
        self.status_label.setText(f"Ready to run {algorithm} — press ▶ Start.")
        self._draw_path(cities, path=[], cost=0.0, iteration=0, final=False)

    def _build_ui(self):
        """ساخت layout تب انیمیشن — تم نارنجی."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── هدر نارنجی ──
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {TAB_ACCENTS[2]['hex_dim']}, stop:1 {PALETTE['bg_mid']});"
            f"border-bottom: 2px solid {TAB_ACCENTS[2]['color']};"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(18, 0, 12, 0)

        self.algo_label = QLabel("Live Solver — waiting for dataset...")
        self.algo_label.setStyleSheet(
            f"color: {TAB_ACCENTS[2]['color']}; font-size: 13px; "
            f"font-weight: bold; background: transparent;"
        )
        h_layout.addWidget(self.algo_label)
        h_layout.addStretch()

        self.run_btn = QPushButton("▶  Start")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.run_btn.setEnabled(False)
        self.run_btn.setFixedWidth(100)
        self.run_btn.setFixedHeight(32)
        self.run_btn.setStyleSheet(_primary_btn(2))
        h_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setFixedWidth(90)
        self.stop_btn.setFixedHeight(32)
        self.stop_btn.setStyleSheet(_secondary_btn(2))
        h_layout.addWidget(self.stop_btn)

        main_layout.addWidget(header)

        # ── نوار وضعیت + Progress ──
        status_bar_w = QWidget()
        status_bar_w.setFixedHeight(36)
        status_bar_w.setStyleSheet(f"background: {PALETTE['bg_mid']};")
        sb_layout = QHBoxLayout(status_bar_w)
        sb_layout.setContentsMargins(16, 0, 16, 0)
        sb_layout.setSpacing(12)

        self.status_label = QLabel("Waiting for dataset...")
        self.status_label.setStyleSheet(
            f"color: {PALETTE['text_dim']}; font-size: 11px; background: transparent;"
        )
        sb_layout.addWidget(self.status_label, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {PALETTE["bg_light"]};
                border: none; border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {TAB_ACCENTS[2]["color"]}, stop:1 {TAB_ACCENTS[2]["glow"]});
                border-radius: 4px;
            }}
        """)
        sb_layout.addWidget(self.progress_bar)

        main_layout.addWidget(status_bar_w)

        # ── نمودارها ──
        charts_area = QWidget()
        charts_area.setStyleSheet(f"background: {PALETTE['bg_dark']};")
        charts_layout = QHBoxLayout(charts_area)
        charts_layout.setContentsMargins(12, 12, 12, 8)
        charts_layout.setSpacing(12)

        # نمودار مسیر (۶۰٪ عرض)
        path_frame = QWidget()
        path_frame.setStyleSheet(
            f"background: {PALETTE['bg_mid']}; border-radius: 8px; "
            f"border: 1px solid {TAB_ACCENTS[2]['hex_dim']};"
        )
        pf_layout = QVBoxLayout(path_frame)
        pf_layout.setContentsMargins(4, 4, 4, 4)
        self.path_canvas = MplCanvas(width=7, height=5, bg=PALETTE["bg_dark"])
        pf_layout.addWidget(self.path_canvas)
        charts_layout.addWidget(path_frame, stretch=6)

        # نمودار همگرایی (۴۰٪ عرض)
        conv_frame = QWidget()
        conv_frame.setStyleSheet(
            f"background: {PALETTE['bg_mid']}; border-radius: 8px; "
            f"border: 1px solid {TAB_ACCENTS[2]['hex_dim']};"
        )
        cf_layout = QVBoxLayout(conv_frame)
        cf_layout.setContentsMargins(4, 4, 4, 4)
        self.convergence_canvas = MplCanvas(width=4, height=5, bg=PALETTE["bg_dark"])
        self.convergence_canvas.style_ax(
            self.convergence_canvas.ax,
            title="Convergence", xlabel="Iteration", ylabel="Cost (km)",
            accent=TAB_ACCENTS[2]["color"]
        )
        cf_layout.addWidget(self.convergence_canvas)
        charts_layout.addWidget(conv_frame, stretch=4)

        main_layout.addWidget(charts_area, stretch=1)

        # ── کارت نتیجه‌ی نهایی ──
        self.result_card = QLabel("")
        self.result_card.setVisible(False)
        self.result_card.setWordWrap(True)
        self.result_card.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_card.setFixedHeight(44)
        self.result_card.setStyleSheet(f"""
            background: {TAB_ACCENTS[2]["hex_dim"]};
            color: {TAB_ACCENTS[2]["color"]};
            border-top: 2px solid {TAB_ACCENTS[2]["color"]};
            padding: 8px 16px;
            font-size: 12px; font-weight: bold;
        """)
        main_layout.addWidget(self.result_card)

    def _on_run_clicked(self):
        """شروع اجرای الگوریتم."""
        if not self._matrix or not self._cities:
            self.status_label.setText("⚠️ ابتدا یک dataset در تب تنظیمات بارگذاری کنید.")
            return

        self._start_solver()

    def _on_stop_clicked(self):
        """درخواست توقف ظریف."""
        if self._worker:
            self._worker.request_stop()
            self.status_label.setText("⏹ درخواست توقف ارسال شد...")
            self.stop_btn.setEnabled(False)

    def _start_solver(self):
        """ساخت SolverWorker + QThread و شروع اجرا."""
        if self._thread and self._thread.isRunning():
            return

        self._reset_progress()

        # ─── ساخت Worker ───
        self._worker = SolverWorker(
            algorithm = self._algorithm,
            matrix    = self._matrix,
            params    = self._params,
        )

        # ─── ساخت Thread ───
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        # ─── اتصال سیگنال‌ها ───
        self._thread.started.connect(self._worker.run)

        # سیگنال‌های Worker → UI
        self._worker.iteration_update.connect(self._on_iteration_update)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_solver_finished)
        self._worker.error.connect(self._on_solver_error)
        # پچ ادغام: سیگنال جدید برای حالت‌های SKIPPED/UNAVAILABLE/ERROR پرولاگ
        self._worker.prolog_status.connect(self._on_prolog_status)

        # پاک‌سازی lifecycle
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        # پچ ادغام: مسیر Prolog هم باید thread را quit کند (دیگر error.emit نمی‌شود)
        self._worker.prolog_status.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        # کنترل UI
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # ─── شروع ───
        self._thread.start()
        logger.info(f"LiveSolverTab: Thread {self._algorithm} شروع شد.")

    @pyqtSlot(int, list, float)
    def _on_iteration_update(self, iteration: int, best_path: list, best_cost: float):
        """
        هر بار که callback از solver فراخوانی شد — به‌روزرسانی UI.

        این slot در thread اصلی (GUI thread) اجرا می‌شود چون
        سیگنال‌ها به‌طور خودکار توسط Qt به thread اصلی کانال می‌شوند.
        """
        self.status_label.setText(
            f"Iteration {iteration:,}   |   Best: {best_cost:,.2f} km"
        )
        # رسم مسیر فعلی
        self._draw_path(
            cities    = self._cities,
            path      = best_path,
            cost      = best_cost,
            iteration = iteration,
            final     = False
        )

    @pyqtSlot(object)
    def _on_solver_finished(self, result: SolverResult):
        """وقتی SolverWorker با موفقیت تمام شد."""
        self._current_result = result
        self.progress_bar.setValue(100)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # رسم مسیر نهایی
        self._draw_path(
            cities    = self._cities,
            path      = result.best_path,
            cost      = result.best_cost,
            iteration = result.iterations,
            final     = True
        )

        # رسم نمودار همگرایی
        if result.history:
            self._draw_convergence(result.history)

        # نمایش کارت نتیجه — متن انگلیسی
        result_text = (
            f"✓  {result.algorithm}   |   "
            f"Best route: {result.best_cost:,.2f} km   |   "
            f"Time: {result.elapsed_sec:.2f} s   |   "
            f"Iterations: {result.iterations:,}"
        )
        self.result_card.setText(result_text)
        self.result_card.setVisible(True)

        # ارسال نتیجه به BenchmarkTab
        self.result_ready.emit(result.algorithm, result)
        logger.info(f"LiveSolverTab: {result.algorithm} تمام شد — {result.best_cost:.2f} km")

    @pyqtSlot(str)
    def _on_solver_error(self, error_msg: str):
        """وقتی SolverWorker با خطا مواجه شد."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"❌ {error_msg}")
        logger.error(f"LiveSolverTab خطا: {error_msg}")

    @pyqtSlot(object)
    def _on_prolog_status(self, prolog_result: "PrologSolverResult"):
        """
        پچ ادغام — نمایش وضعیت دقیق پرولاگ وقتی status != SOLVED است.

        این متد دقیقاً همان منطق و پیام‌هایی را که در prolog_bridge.py
        تعریف شده (ui_message کوتاه برای برچسب اصلی + ui_tooltip کامل
        و آموزشی) بدون هیچ تغییری روی UI پیاده می‌کند — به‌جای اینکه
        کاربر فقط یک گراف خالی یا «هزینه = ۰» بدون توضیح ببیند.

        سه حالت ممکن (طبق PrologStatus در prolog_bridge.py):
          • SKIPPED_SIZE        → N > PROLOG_MAX_CITIES (۱۵): بای‌پس عمدی
          • ENGINE_UNAVAILABLE  → pyswip/SWI-Prolog نصب یا پیکربندی نشده
          • RUNTIME_ERROR       → خطای غیرمنتظره در حین اجرای پرولاگ
        """
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        # رنگ‌بندی بر اساس شدت وضعیت — زرد برای Skip عمدی (طراحی‌شده،
        # نه خطا)، نارنجی/قرمز برای عدم دسترسی یا خطای واقعی
        if prolog_result.status == PrologStatus.SKIPPED_SIZE:
            severity_color = PALETTE["accent_warn"]   # زرد — این یک ویژگی طراحی‌شده است نه باگ
        else:
            severity_color = PALETTE["accent_hot"]    # نارنجی — عدم دسترسی یا خطای واقعی

        # ── نمایش پیام کوتاه در status_label ──
        self.status_label.setText(f"🔷 Prolog — {prolog_result.ui_message}")

        # ── نمایش پیام کامل + tooltip در result_card ──
        # (tooltip به‌صورت QToolTip روی خودِ کارت قرار می‌گیرد تا کاربر
        # با hover کردن، توضیح کامل آموزشی prolog_bridge.py را ببیند)
        self.result_card.setText(f"🔷  Prolog (Held-Karp)   |   {prolog_result.ui_message}")
        self.result_card.setToolTip(prolog_result.ui_tooltip)
        self.result_card.setStyleSheet(f"""
            background: {TAB_ACCENTS[2]["hex_dim"]};
            color: {severity_color};
            border-top: 2px solid {severity_color};
            padding: 8px 16px;
            font-size: 12px; font-weight: bold;
        """)
        self.result_card.setVisible(True)

        # ── نمودار را به حالت «فقط نقاط شهر، بدون مسیر» برمی‌گردانیم ──
        # چون best_path خالی است، رسم مسیر معنی ندارد؛ ولی نقشه‌ی
        # شهرها باید دیده بشه تا کاربر گیج نشه که چیزی کرش کرده.
        self._draw_path(
            cities    = self._cities,
            path      = [],
            cost      = 0.0,
            iteration = 0,
            final     = False
        )

        logger.warning(
            f"LiveSolverTab: Prolog در وضعیت {prolog_result.status.value} — "
            f"بدون مسیر/هزینه قابل نمایش."
        )

    def _draw_path(
        self,
        cities:    list,
        path:      list,
        cost:      float,
        iteration: int,
        final:     bool
    ):
        """
        رسم شهرها + مسیر — تیتر و برچسب‌ها انگلیسی برای رفع RTL.
        رنگ مسیر از ALGO_COLORS متناسب با الگوریتم انتخاب می‌شود.
        """
        ax = self.path_canvas.ax
        ax.cla()
        ax.set_facecolor(PALETTE["bg_dark"])

        if not cities:
            return

        lons = [c.lon for c in cities]
        lats = [c.lat for c in cities]
        algo_color = ALGO_COLORS.get(self._algorithm, TAB_ACCENTS[2]["color"])

        # شبکه
        ax.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.2, linewidth=0.6)

        # ── رسم یال‌های مسیر ──
        if path and len(path) > 1:
            path_color = PALETTE["accent_ok"] if final else algo_color
            path_lw    = 2.2 if final else 1.4
            alpha      = 0.95 if final else 0.65

            for i in range(len(path)):
                c_from = cities[path[i]]
                c_to   = cities[path[(i + 1) % len(path)]]
                ax.plot(
                    [c_from.lon, c_to.lon],
                    [c_from.lat, c_to.lat],
                    color=path_color, lw=path_lw, alpha=alpha, zorder=2
                )

            # رسم پیکان جهت اول مسیر
            if final and len(path) >= 2:
                c0 = cities[path[0]]
                c1 = cities[path[1]]
                ax.annotate(
                    "", xy=(c1.lon, c1.lat), xytext=(c0.lon, c0.lat),
                    arrowprops=dict(
                        arrowstyle="->", color=PALETTE["accent_ok"],
                        lw=2.0, mutation_scale=16
                    ), zorder=5
                )

        # ── نقاط شهرها ──
        # هاله
        ax.scatter(lons, lats, c=algo_color, s=180, zorder=2, alpha=0.15)
        # نقاط اصلی
        scatter = ax.scatter(
            lons, lats,
            c=[algo_color] * len(cities), s=55, zorder=4,
            alpha=0.95, linewidths=0.8, edgecolors="white"
        )

        # نمایش نام — کوچک و کم‌رنگ
        for city in cities:
            ax.annotate(
                city.name,
                xy=(city.lon, city.lat),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7, color=PALETTE["text_dim"],
                fontfamily=MPL_FONT
            )

        # ── عنوان ──
        if path and cost > 0:
            suffix = " ✓ Final" if final else f"  iter {iteration:,}"
            title  = f"{self._algorithm}{suffix}   |   {cost:,.1f} km"
            t_color = PALETTE["accent_ok"] if final else PALETTE["text_main"]
        else:
            title   = f"City Map  —  {len(cities)} cities"
            t_color = PALETTE["text_dim"]

        ax.set_title(title, color=t_color, fontsize=10, pad=8, fontfamily=MPL_FONT)
        ax.set_xlabel("Longitude", color=PALETTE["text_dim"], fontsize=8)
        ax.set_ylabel("Latitude",  color=PALETTE["text_dim"], fontsize=8)
        ax.tick_params(colors=PALETTE["text_dim"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])

        self.path_canvas.fig.tight_layout(pad=1.0)
        self.path_canvas.draw()

    def _draw_convergence(self, history: list):
        """رسم نمودار همگرایی — برچسب‌ها انگلیسی."""
        ax = self.convergence_canvas.ax
        ax.cla()
        ax.set_facecolor(PALETTE["bg_dark"])

        iters = [h[0] for h in history]
        costs = [h[1] for h in history]
        algo_color = ALGO_COLORS.get(self._algorithm, TAB_ACCENTS[2]["color"])

        ax.plot(iters, costs, color=algo_color, lw=2.0, alpha=0.9)
        ax.fill_between(iters, costs, min(costs),
                        alpha=0.12, color=algo_color)

        # خط بهترین هزینه
        ax.axhline(min(costs), color=PALETTE["accent_ok"],
                   linestyle=":", lw=1.2, alpha=0.7)
        ax.text(
            iters[-1] * 0.05, min(costs) * 0.998,
            f"Best: {min(costs):,.0f}",
            color=PALETTE["accent_ok"], fontsize=7.5
        )

        self.convergence_canvas.style_ax(
            ax,
            title=f"{self._algorithm} Convergence",
            xlabel="Iteration", ylabel="Cost (km)",
            accent=algo_color
        )
        self.convergence_canvas.fig.tight_layout(pad=1.0)
        self.convergence_canvas.draw()

    def _reset_ui(self):
        """پاک کردن UI برای شروع جدید."""
        self.result_card.setVisible(False)
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _reset_progress(self):
        """ریست ProgressBar و لیبل وضعیت."""
        self.progress_bar.setValue(0)
        self.result_card.setVisible(False)
        self.status_label.setText(f"در حال اجرای {self._algorithm}...")


# =============================================================================
# بخش ۶: تب چهارم — بنچمارک و مقایسه (BenchmarkTab)
# =============================================================================

class BenchmarkTab(QWidget):
    """
    تب بنچمارک: اجرای هر سه الگوریتم + نمودارهای مقایسه‌ای.

    این تب نتایج هر سه الگوریتم را کنار هم رسم می‌کند:
      - نمودار میله‌ای: مقایسه‌ی هزینه‌ی نهایی
      - نمودار میله‌ای: مقایسه‌ی زمان اجرا
      - نمودار خطی: همگرایی هر دو الگوریتم متاهیوریستیک

    سیگنال‌های ورودی (از MainWindow):
      - نتایج از LiveSolverTab (تدریجی — هر بار یک الگوریتم)
      - یا اجرای کامل بنچمارک با دکمه
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # نتایج جمع‌آوری‌شده
        self._results: dict = {}   # {"ACO": SolverResult, "Genetic": ..., "Prolog": ...}

        # پچ ادغام: اگر پرولاگ Skip/Unavailable/Error شده باشد، پیام و
        # tooltip دقیقش اینجا نگه داشته می‌شود تا جدول نتایج به‌جای
        # هزینه‌ی صفر گمراه‌کننده، توضیح صحیح را نشان دهد.
        self._prolog_status_note: Optional[PrologSolverResult] = None

        # Worker / Thread برای بنچمارک کامل
        self._worker: Optional[BenchmarkWorker] = None
        self._thread: Optional[QThread]         = None

        # ذخیره‌ی ماتریس برای بنچمارک کامل
        self._matrix: Optional[list] = None

        self._build_ui()

    def update_matrix(self, matrix: list):
        """دریافت ماتریس از MainWindow."""
        self._matrix = matrix

    def add_result(self, algorithm: str, result: SolverResult):
        """
        اضافه کردن نتیجه‌ی یک الگوریتم (از LiveSolverTab یا BenchmarkWorker).
        وقتی نتیجه‌ای اضافه میشه، نمودارها به‌روز میشن.
        """
        self._results[algorithm] = result
        self._update_charts()
        self._update_result_table()

    def _build_ui(self):
        """ساخت layout تب بنچمارک — تم آکوا."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── هدر آکوا ──
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {TAB_ACCENTS[3]['hex_dim']}, stop:1 {PALETTE['bg_mid']});"
            f"border-bottom: 2px solid {TAB_ACCENTS[3]['color']};"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(18, 0, 12, 0)

        hdr_lbl = QLabel("📊  Benchmark — Algorithm Comparison")
        hdr_lbl.setStyleSheet(
            f"color: {TAB_ACCENTS[3]['color']}; font-size: 14px; "
            f"font-weight: bold; background: transparent;"
        )
        h_layout.addWidget(hdr_lbl)
        h_layout.addStretch()

        self.run_all_btn = QPushButton("🔄  Run Full Benchmark")
        self.run_all_btn.clicked.connect(self._on_run_benchmark)
        self.run_all_btn.setEnabled(False)
        self.run_all_btn.setFixedHeight(32)
        self.run_all_btn.setStyleSheet(_primary_btn(3))
        h_layout.addWidget(self.run_all_btn)

        main_layout.addWidget(header)

        # ── Progress Bar بنچمارک ──
        self.bench_progress = QProgressBar()
        self.bench_progress.setRange(0, 100)
        self.bench_progress.setVisible(False)
        self.bench_progress.setFixedHeight(5)
        self.bench_progress.setTextVisible(False)
        self.bench_progress.setStyleSheet(f"""
            QProgressBar {{ background: {PALETTE["bg_mid"]}; border: none; }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {TAB_ACCENTS[3]["color"]}, stop:1 {TAB_ACCENTS[3]["glow"]});
            }}
        """)
        main_layout.addWidget(self.bench_progress)

        # ── محتوا ──
        content = QWidget()
        content.setStyleSheet(f"background: {PALETTE['bg_dark']};")
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(14, 14, 14, 10)
        c_layout.setSpacing(12)

        # ── ردیف نمودارها ──
        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)

        def _chart_frame(title_en: str) -> tuple:
            """فریم کارتی برای نمودار."""
            frame = QGroupBox(title_en)
            frame.setStyleSheet(_tab_group_style(3))
            inner = QVBoxLayout(frame)
            inner.setContentsMargins(4, 6, 4, 4)
            canvas = MplCanvas(width=3.5, height=3.2, bg=PALETTE["bg_dark"])
            inner.addWidget(canvas)
            return frame, canvas

        cost_frame, self.cost_canvas = _chart_frame("Route Cost (km)")
        charts_row.addWidget(cost_frame)

        time_frame, self.time_canvas = _chart_frame("Execution Time (s)")
        charts_row.addWidget(time_frame)

        conv_frame, self.conv_canvas = _chart_frame("Convergence Comparison (ACO vs GA)")
        charts_row.addWidget(conv_frame, stretch=1)

        c_layout.addLayout(charts_row)

        # ── جدول نتایج ──
        result_frame = QGroupBox("Results Summary")
        result_frame.setStyleSheet(_tab_group_style(3))
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(8, 6, 8, 8)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFixedHeight(130)
        self.result_text.setStyleSheet(f"""
            QTextEdit {{
                background: {PALETTE['bg_mid']}; color: {PALETTE['text_main']};
                border: none; font-family: monospace; font-size: 11.5px;
                padding: 8px;
            }}
            QScrollBar:vertical {{ width: 6px; background: {PALETTE['bg_dark']}; }}
            QScrollBar::handle:vertical {{ background: {PALETTE['border']}; border-radius: 3px; }}
        """)
        self.result_text.setPlainText(
            "No results yet.\n"
            "Run an algorithm from the Live Solver tab, or press Run Full Benchmark."
        )
        result_layout.addWidget(self.result_text)
        c_layout.addWidget(result_frame)

        main_layout.addWidget(content, stretch=1)

    def _on_run_benchmark(self):
        """اجرای بنچمارک کامل (هر سه الگوریتم پشت سرهم)."""
        if not self._matrix:
            return

        if self._thread and self._thread.isRunning():
            return

        self._results.clear()
        self._prolog_status_note = None   # پچ ادغام: ریست یادداشت قبلی وضعیت پرولاگ
        self.bench_progress.setVisible(True)
        self.bench_progress.setValue(0)
        self.run_all_btn.setEnabled(False)

        # ─── ساخت Worker ───
        self._worker = BenchmarkWorker(matrix=self._matrix, benchmark_params={})
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        # ─── اتصال سیگنال‌ها ───
        self._thread.started.connect(self._worker.run)
        self._worker.algo_done.connect(self._on_algo_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.progress.connect(self.bench_progress.setValue)
        self._worker.error.connect(self._on_bench_error)
        # پچ ادغام: سیگنال وضعیت پرولاگ (Skipped/Unavailable/RuntimeError)
        self._worker.prolog_status.connect(self._on_prolog_status)

        # lifecycle
        self._worker.all_done.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    @pyqtSlot(str, object)
    def _on_algo_done(self, name: str, result: SolverResult):
        """هر الگوریتم که تمام شد، نتیجه‌اش را اضافه کن."""
        self.add_result(name, result)
        logger.info(f"BenchmarkTab: {name} تمام شد — {result.best_cost:.2f} km")

    @pyqtSlot(object)
    def _on_prolog_status(self, prolog_result: "PrologSolverResult"):
        """
        پچ ادغام — وقتی پرولاگ در بنچمارک Skip/Unavailable/Error شده باشد.

        پیام و tooltip دقیق prolog_bridge.py را نگه می‌داریم تا
        _update_result_table() به‌جای نمایش هزینه‌ی صفر یا NaN خام،
        توضیح صحیح («رد شد: N > 15» یا «در دسترس نیست») را چاپ کند.
        """
        self._prolog_status_note = prolog_result
        # tooltip روی کل کادر نتایج قرار می‌گیرد تا با hover دیده شود
        self.result_text.setToolTip(prolog_result.ui_tooltip)
        logger.info(
            f"BenchmarkTab: وضعیت پرولاگ ثبت شد — {prolog_result.status.value}"
        )
        self._update_result_table()

    @pyqtSlot(dict)
    def _on_all_done(self, results: dict):
        """وقتی همه‌ی الگوریتم‌ها تمام شدن."""
        self.bench_progress.setVisible(False)
        self.run_all_btn.setEnabled(True)
        logger.info("BenchmarkTab: بنچمارک کامل تمام شد.")

    @pyqtSlot(str)
    def _on_bench_error(self, error_msg: str):
        self.bench_progress.setVisible(False)
        self.run_all_btn.setEnabled(True)
        self.result_text.setPlainText(f"❌ {error_msg}")

    def _update_charts(self):
        """به‌روزرسانی نمودارهای مقایسه‌ای — برچسب‌ها انگلیسی."""
        if not self._results:
            return

        # ───────────────────────────────────────────────────────────
        # پچ ادغام: اگر پرولاگ Skip/Unavailable/Error شده باشد،
        # best_cost آن NaN است (طبق BenchmarkWorker پچ‌شده). چنین
        # موردی را از نمودارهای میله‌ای حذف می‌کنیم — چون رسم NaN با
        # ax.bar باعث می‌شد یا میله‌ای در ارتفاع صفر گمراه‌کننده دیده
        # شود یا matplotlib آن را اصلاً رسم نکند بدون هیچ توضیحی.
        # پیام واقعی («رد شد / در دسترس نیست») در جدول متنی پایین
        # (_update_result_table) و نه در نمودار میله‌ای نشان داده می‌شود.
        # ───────────────────────────────────────────────────────────
        chart_names = [
            n for n in self._results
            if not math.isnan(self._results[n].best_cost)
        ]

        if not chart_names:
            # همه‌ی نتایج فعلی NaN هستند (مثلاً فقط پرولاگ اجرا شده و Skip شده)
            for canvas in (self.cost_canvas, self.time_canvas):
                ax = canvas.ax
                ax.cla(); ax.set_facecolor(PALETTE["bg_dark"])
                ax.text(
                    0.5, 0.5, "No comparable\nresult yet",
                    ha="center", va="center", color=PALETTE["text_dim"],
                    transform=ax.transAxes, fontsize=9
                )
                canvas.draw()
        else:
            names  = chart_names
            costs  = [self._results[n].best_cost   for n in names]
            times  = [self._results[n].elapsed_sec  for n in names]
            colors = [ALGO_COLORS.get(n, TAB_ACCENTS[3]["color"]) for n in names]

            # ─── نمودار هزینه ───
            ax = self.cost_canvas.ax
            ax.cla(); ax.set_facecolor(PALETTE["bg_dark"])
            bars = ax.bar(names, costs, color=colors, alpha=0.82, width=0.45)
            for bar, val in zip(bars, costs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.012,
                    f"{val:,.0f}", ha="center", va="bottom",
                    color=PALETTE["text_main"], fontsize=8.5, fontweight="bold"
                )
            self.cost_canvas.style_ax(
                ax, title="Route Cost", ylabel="km",
                accent=TAB_ACCENTS[3]["color"]
            )
            self.cost_canvas.fig.tight_layout(pad=1.0)
            self.cost_canvas.draw()

            # ─── نمودار زمان ───
            ax2 = self.time_canvas.ax
            ax2.cla(); ax2.set_facecolor(PALETTE["bg_dark"])
            bars2 = ax2.bar(names, times, color=colors, alpha=0.82, width=0.45)
            for bar, val in zip(bars2, times):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.012,
                    f"{val:.2f}s", ha="center", va="bottom",
                    color=PALETTE["text_main"], fontsize=8.5, fontweight="bold"
                )
            self.time_canvas.style_ax(
                ax2, title="Execution Time", ylabel="seconds",
                accent=TAB_ACCENTS[3]["color"]
            )
            self.time_canvas.fig.tight_layout(pad=1.0)
            self.time_canvas.draw()

        # ─── نمودار همگرایی مقایسه‌ای ───
        ax3 = self.conv_canvas.ax
        ax3.cla(); ax3.set_facecolor(PALETTE["bg_dark"])

        has_conv = False
        for name in ["ACO", "Genetic"]:
            if name in self._results and self._results[name].history:
                hist  = self._results[name].history
                iters = [h[0] for h in hist]
                vals  = [h[1] for h in hist]
                ax3.plot(iters, vals, label=name,
                         color=ALGO_COLORS[name], lw=1.8, alpha=0.9)
                ax3.fill_between(iters, vals, min(vals),
                                 alpha=0.08, color=ALGO_COLORS[name])
                has_conv = True

        if has_conv:
            ax3.legend(
                facecolor=PALETTE["bg_mid"], edgecolor=PALETTE["border"],
                labelcolor=PALETTE["text_main"], fontsize=8
            )

        self.conv_canvas.style_ax(
            ax3, title="ACO vs GA Convergence",
            xlabel="Iteration", ylabel="Cost (km)",
            accent=TAB_ACCENTS[3]["color"]
        )
        self.conv_canvas.fig.tight_layout(pad=1.0)
        self.conv_canvas.draw()

    def _update_result_table(self):
        """به‌روزرسانی جدول متنی نتایج — انگلیسی."""
        if not self._results:
            return

        lines = [
            f"{'Algorithm':<18}{'Cost (km)':<18}{'Time (s)':<14}{'Iterations'}",
            "─" * 58
        ]

        for name, result in self._results.items():
            exact = " ✓" if name == "Prolog" else "  "

            # ───────────────────────────────────────────────────────
            # پچ ادغام: اگر این ردیف مربوط به پرولاگ باشد و وضعیت آن
            # SOLVED نباشد (best_cost == NaN)، به‌جای چاپ خام "nan"،
            # پیام کوتاه دقیق prolog_bridge.py (مثلاً «رد شد: N > 15»)
            # را در همان ردیف نشان می‌دهیم — کاربر باید بفهمد این یک
            # ویژگی طراحی‌شده است، نه یک مقدار محاسباتی خراب.
            # ───────────────────────────────────────────────────────
            if name == "Prolog" and math.isnan(result.best_cost) and self._prolog_status_note:
                lines.append(
                    f"{name + exact:<18}{self._prolog_status_note.ui_message}"
                )
                continue

            lines.append(
                f"{name + exact:<18}"
                f"{result.best_cost:,.2f}{'':>4}"[:18] +
                f"   {result.elapsed_sec:.3f}{'':>4}"[:14] +
                f"   {result.iterations:,}"
            )

        if "Prolog" in self._results:
            pc = self._results["Prolog"].best_cost
            if not math.isnan(pc) and pc > 0:
                lines += ["", "Gap vs. Prolog (Ground Truth):"]
                for name in ["ACO", "Genetic"]:
                    if name in self._results:
                        gap = ((self._results[name].best_cost - pc) / pc) * 100
                        lines.append(f"  {name}: {gap:+.1f}%")
            elif self._prolog_status_note:
                # پچ ادغام: یادآوری کوتاه چرا مقایسه‌ی Gap موجود نیست
                lines += ["", f"ℹ️  Gap vs. Prolog unavailable — {self._prolog_status_note.ui_message}"]

        self.result_text.setPlainText("\n".join(lines))

    def _group_style(self) -> str:
        return _tab_group_style(3)


# =============================================================================
# بخش ۷: پنجره‌ی اصلی (MainWindow)
# =============================================================================

class MainWindow(QMainWindow):
    """
    پنجره‌ی اصلی برنامه — هماهنگ‌کننده‌ی مرکزی.

    MainWindow مسئول:
      ۱. نگه‌داری وضعیت مشترک: (cities, matrix)
      ۲. هماهنگی بین تب‌ها از طریق سیگنال‌ها
      ۳. تنظیم استایل سراسری برنامه
      ۴. StatusBar پایین برای پیام‌های سیستمی
    """

    def __init__(self):
        super().__init__()

        # وضعیت مشترک — توسط ConfigTab پر میشه
        self._cities: Optional[list] = None
        self._matrix: Optional[list] = None

        self._build_window()
        self._apply_dark_theme()

    def _build_window(self):
        """ساخت ساختار اصلی پنجره."""
        self.setWindowTitle("TSP Solver — Traveling Salesman Problem")
        self.setMinimumSize(1100, 720)
        self.resize(1350, 860)

        # ── Tab Widget ──
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # هر تب رنگ تأکید مخصوص خودش را دارد
        c0 = TAB_ACCENTS[0]["color"]
        c1 = TAB_ACCENTS[1]["color"]
        c2 = TAB_ACCENTS[2]["color"]
        c3 = TAB_ACCENTS[3]["color"]

        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background: {PALETTE["bg_dark"]};
            }}
            QTabBar {{
                background: {PALETTE["bg_mid"]};
                border-bottom: 1px solid {PALETTE["border"]};
            }}
            QTabBar::tab {{
                background: {PALETTE["bg_mid"]};
                color: {PALETTE["text_dim"]};
                padding: 10px 20px;
                border: none;
                border-bottom: 3px solid transparent;
                font-size: 12px;
                min-width: 160px;
                margin-right: 2px;
            }}
            QTabBar::tab:nth-child(1):selected {{ color: {c0}; border-bottom-color: {c0}; font-weight: bold; background: {TAB_ACCENTS[0]["hex_dim"]}; }}
            QTabBar::tab:nth-child(2):selected {{ color: {c1}; border-bottom-color: {c1}; font-weight: bold; background: {TAB_ACCENTS[1]["hex_dim"]}; }}
            QTabBar::tab:nth-child(3):selected {{ color: {c2}; border-bottom-color: {c2}; font-weight: bold; background: {TAB_ACCENTS[2]["hex_dim"]}; }}
            QTabBar::tab:nth-child(4):selected {{ color: {c3}; border-bottom-color: {c3}; font-weight: bold; background: {TAB_ACCENTS[3]["hex_dim"]}; }}
            QTabBar::tab:hover:!selected {{
                background: {PALETTE["bg_light"]};
                color: {PALETTE["text_main"]};
            }}
        """)

        # ─── ساخت تب‌ها ───
        self.config_tab    = ConfigTab()
        self.advisor_tab   = AIAdvisorTab()
        self.live_tab      = LiveSolverTab()
        self.benchmark_tab = BenchmarkTab()

        self.tabs.addTab(self.config_tab,    "⚙️   Data Config")
        self.tabs.addTab(self.advisor_tab,   "🤖   AI Advisor")
        self.tabs.addTab(self.live_tab,      "🎬   Live Solver")
        self.tabs.addTab(self.benchmark_tab, "📊   Benchmark")

        self.setCentralWidget(self.tabs)

        # ─── اتصال سیگنال‌ها بین تب‌ها ───
        self.config_tab.data_ready.connect(self._on_data_ready)
        self.advisor_tab.solve_requested.connect(self._on_solve_requested)
        self.live_tab.result_ready.connect(self._on_result_ready)

        # ── StatusBar ──
        self.status_bar = QStatusBar()
        self.status_bar.setFixedHeight(26)
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background: {PALETTE["bg_mid"]};
                color: {PALETTE["text_dim"]};
                font-size: 11px;
                border-top: 1px solid {PALETTE["border"]};
            }}
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — start from the Data Config tab.")

    @pyqtSlot(list, list)
    def _on_data_ready(self, cities: list, matrix: list):
        """
        وقتی ConfigTab داده را آماده کرد:
          ۱. وضعیت مشترک را به‌روز کن
          ۲. به تمام تب‌های دیگر اطلاع بده
          ۳. به تب مشاور هوشمند برو
        """
        self._cities = cities
        self._matrix = matrix

        # اطلاع‌رسانی به تب‌های دیگر
        self.advisor_tab.update_data(cities, matrix)
        self.benchmark_tab.update_matrix(matrix)
        self.benchmark_tab.run_all_btn.setEnabled(True)

        self.status_bar.showMessage(
            f"✓ {len(cities)} cities loaded  |  {len(cities)}×{len(cities)} matrix ready"
        )

        # رفتن به تب مشاور
        self.tabs.setCurrentIndex(1)
        logger.info(f"MainWindow: داده آماده — {len(cities)} شهر")

    @pyqtSlot(str, dict)
    def _on_solve_requested(self, algorithm: str, params: dict):
        """
        وقتی AIAdvisorTab درخواست حل ارسال کرد:
          ۱. داده‌ها را به LiveSolverTab بده
          ۲. به تب انیمیشن برو
        """
        if not self._cities or not self._matrix:
            self.status_bar.showMessage("⚠️ ابتدا یک dataset بارگذاری کنید.")
            return

        # آماده‌سازی LiveSolverTab
        self.live_tab.prepare_solve(algorithm, params, self._cities, self._matrix)

        # رفتن به تب انیمیشن
        self.tabs.setCurrentIndex(2)
        self.status_bar.showMessage(f"🚀 {algorithm} ready — press ▶ Start.")

    @pyqtSlot(str, object)
    def _on_result_ready(self, algorithm: str, result: SolverResult):
        """وقتی LiveSolverTab نتیجه داشت — به BenchmarkTab اضافه کن."""
        self.benchmark_tab.add_result(algorithm, result)
        self.status_bar.showMessage(
            f"✅ {algorithm} تمام شد | هزینه: {result.best_cost:,.2f} km | "
            f"زمان: {result.elapsed_sec:.2f} s"
        )

    def _apply_dark_theme(self):
        """اعمال تم تاریک سراسری عمیق‌تر و زنده‌تر."""
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {PALETTE["bg_dark"]};
                color: {PALETTE["text_main"]};
            }}
            QComboBox, QSpinBox {{
                background: {PALETTE["bg_light"]};
                color: {PALETTE["text_main"]};
                border: 1px solid {PALETTE["border"]};
                border-radius: 5px;
                padding: 5px 9px;
                font-size: 12px;
                selection-background-color: {PALETTE["bg_card"]};
            }}
            QComboBox:hover, QSpinBox:hover {{
                border-color: {PALETTE["border_glow"]};
                background: {PALETTE["bg_card"]};
            }}
            QComboBox:focus, QSpinBox:focus {{
                border-color: {TAB_ACCENTS[0]["color"]};
            }}
            QComboBox::drop-down {{
                border: none; width: 18px;
            }}
            QComboBox QAbstractItemView {{
                background: {PALETTE["bg_light"]};
                color: {PALETTE["text_main"]};
                border: 1px solid {PALETTE["border"]};
                selection-background-color: {PALETTE["bg_card"]};
                outline: none;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background: {PALETTE["bg_card"]};
                border: none; width: 16px;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background: {PALETTE["border"]};
            }}
            QLabel {{
                color: {PALETTE["text_main"]};
                background: transparent;
            }}
            QTextEdit {{
                background: {PALETTE["bg_dark"]};
                color: {PALETTE["text_main"]};
                border: 1px solid {PALETTE["border"]};
                border-radius: 5px;
            }}
            QScrollBar:vertical {{
                background: {PALETTE["bg_mid"]}; width: 8px; border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {PALETTE["border"]}; border-radius: 4px; min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {PALETTE["border_glow"]};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QSplitter::handle {{
                background: {PALETTE["border"]};
            }}
            QToolTip {{
                background: {PALETTE["bg_card"]};
                color: {PALETTE["text_main"]};
                border: 1px solid {PALETTE["border_glow"]};
                padding: 4px 8px; border-radius: 4px;
                font-size: 11px;
            }}
        """)


# =============================================================================
# بخش ۸: نقطه‌ی ورود (Entry Point)
# =============================================================================

def main():
    """
    تابع اصلی اجرای برنامه.
    از main.py صدا زده میشه: from gui import main; main()
    """
    # تنظیم لاگینگ برای حالت اجرای کامل
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s → %(message)s",
        datefmt="%H:%M:%S"
    )

    app = QApplication(sys.argv)
    app.setApplicationName("TSP Solver")
    app.setApplicationVersion("4.0")

    # تنظیم فونت پیش‌فرض
    font = QFont("Segoe UI", 11)
    app.setFont(font)

    # ساخت و نمایش پنجره‌ی اصلی
    window = MainWindow()
    window.show()

    logger.info("TSP Solver GUI راه‌اندازی شد.")
    sys.exit(app.exec())


# اجرای مستقیم (python src/gui.py)
if __name__ == "__main__":
    main()
