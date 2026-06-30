# =============================================================================
# فایل: train_model.py
# نقش: تولید داده‌های سنتتیک، آموزش مدل یادگیری ماشین، و ذخیره‌سازی مدل
#       برای سیستم مشاور هوشمند (AI Advisor) — فاز ۵ پروژه TSP
#
# ═══════════════════════════════════════════════════════════════════════════

#   ═════════════════════════════════════
#   چه چیزی یاد می‌گیریم؟
#   مسئله: «برای یک نقشه‌ی TSP با ویژگی‌های مشخص، کدام الگوریتم بهترین
#   تعادل بین کیفیت جواب و سرعت اجرا را ارائه می‌دهد؟»
#
#   جواب را با اجرای واقعی سه الگوریتم (Prolog, ACO, GA) روی صدها
#   نقشه‌ی سنتتیک پیدا می‌کنیم، سپس نتایج را به یک درخت تصمیم می‌آموزیم.
#
#   ═════════════════════════════════════
#   ویژگی‌های ورودی (Features)
#   ═════════════════════════════════════
#   ۱. n_cities:             تعداد شهرها — اصلی‌ترین عامل تعیین‌کننده‌ی پیچیدگی
#   ۲. coord_variance:       واریانس مختصات شهرها — معیار پراکندگی جغرافیایی
#   ۳. dist_mean:            میانگین فواصل بین شهرها (km) — اندازه‌ی کلی نقشه
#   ۴. dist_std:             انحراف معیار فواصل — یکنواختی توزیع شهرها
#   ۵. dist_cv:              ضریب تغییرات (std/mean) — نسبی‌سازی پراکندگی فاصله
#   ۶. density_proxy:        تراکم تخمینی: n / (sqrt(area_km²)) — چگالی شهری
#   ۷. nn_ratio:             نسبت میانگین نزدیک‌ترین همسایه به میانگین کل فواصل
#   ۸. dist_skewness:        کجی توزیع فواصل — تشخیص cluster های شهری
#   ۹. max_min_ratio:        نسبت بیشترین به کمترین فاصله — انتشار جغرافیایی
#
#   ═════════════════════════════════════
#   برچسب خروجی (Label)
#   ═════════════════════════════════════
#   0 = Prolog (Held-Karp, Exact)    ← برای N های کوچک که پرولاگ اجرا می‌شود
#   1 = ACO (Ant Colony)             ← وقتی مسئله متوسط یا با ساختار خاص است
#   2 = Genetic Algorithm            ← وقتی مسئله بزرگ یا با توزیع غیریکنواخت است
#
# =============================================================================

import os           
import sys         
import math         
import time         # برای اندازه‌گیری زمان اجرای الگوریتم‌ها
import random       # برای تولید مختصات تصادفی شهرها
import logging      
import argparse     
import csv          # برای ذخیره دیتاست در فرمت CSV (بدون pandas)
import json         # برای ذخیره متادیتای مدل در فایل pkl
from typing import Optional  

import numpy as np                              # برای محاسبات عددی روی آرایه‌ها
from sklearn.tree import DecisionTreeClassifier # مدل اصلی — درخت تصمیم
from sklearn.preprocessing import StandardScaler # نرمال‌سازی ویژگی‌ها
from sklearn.model_selection import (
    cross_val_score,         # اعتبارسنجی متقاطع (K-Fold)
    StratifiedKFold,         # KFold با توزیع یکنواخت برچسب‌ها
    train_test_split         # تقسیم داده به train/test
)
from sklearn.metrics import (
    classification_report,   # گزارش precision/recall/F1 برای هر کلاس
    confusion_matrix         # ماتریس اشتباه برای تحلیل عمیق‌تر
)
import joblib                                   # برای ذخیره/بارگذاری مدل (pkl)

# =============================================================================
# تنظیم مسیرها (sys.path Bootstrap)
# =============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# =============================================================================
# import ماژول‌های پروژه (بعد از تنظیم sys.path)
# =============================================================================

try:
    from core import City, haversine_km, build_distance_matrix
    from solvers import AntColonyOptimizer, GeneticAlgorithmSolver, SolverResult
    from prolog_bridge import solve_with_prolog, PrologStatus, PROLOG_MAX_CITIES
except ImportError as _import_error:
    print(
        "\n" + "═" * 65 +
        "\n  ❌ خطا: فایل‌های پروژه پیدا نشدند.\n" + "═" * 65 +
        f"\n  جزئیات: {_import_error}"
        "\n\n  مطمئن شوید train_model.py در کنار core.py و solvers.py باشد.\n" +
        "═" * 65 + "\n"
    )
    sys.exit(1)


# =============================================================================
# پیکربندی Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s → %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TSP.TrainModel")


# =============================================================================
# ثابت‌های پروژه
# =============================================================================
# مسیر پوشه‌ی خروجی 
MODELS_DIR   = os.path.join(_THIS_DIR, "models")

# نام فایل‌های خروجی
MODEL_FILENAME   = "ai_advisor_model.pkl"    # فایل مدل سریالایز‌شده
DATASET_FILENAME = "training_dataset.csv"    # دیتاست خام برای بررسی

# نام‌های برچسب‌ها برای خوانایی بهتر
LABEL_NAMES = {
    0: "Prolog (Exact)",
    1: "ACO (Ant Colony)",
    2: "Genetic Algorithm"
}

# =============================================================================
# ✦ تابع امتیاز  — مبتنی بر سیاست دو-مرحله‌ای (Tiered Policy)
# =============================================================================

# اگر پرولاگ در کمتر از این مدت زمان تمام شود، به طور مستقیم برنده می‌شود
# (چون دقت/بهینگی قطعی اهمیتش از چند صدم ثانیه بیشتر است)
PROLOG_TIME_CEILING_SEC = 3.0   

# --- سطح ۲: وزن‌دهی تابع امتیاز برای N بزرگ (فقط ACO vs GA) ---

COST_WEIGHT_LARGE_N  = 0.8   # وزن غالب برای کیفیت مسیر 
TIME_WEIGHT_LARGE_N  = 0.2   # وزن محدود برای سرعت 

# هر زمانی بالاتر از این سقف، امتیاز سرعت صفر
# می‌گیرد (یعنی دیگر تفاوتی نمی‌کند اگر خیلی کندتر هم باشد).
#فقط ACO vs GA
TIME_NORMALIZATION_CAP_SEC = 2.5

# نگه می‌داریم برای سازگاری با کدهای دیگر که شاید COST_WEIGHT/TIME_WEIGHT
# سراسری را مستقیماً بخوانند (مثلاً لاگ‌ها). این مقادیر دیگر در منطق
COST_WEIGHT  = COST_WEIGHT_LARGE_N
TIME_WEIGHT  = TIME_WEIGHT_LARGE_N

# پارامترهای پیش‌فرض الگوریتم‌ها (سبک‌شده برای تولید سریع‌تر داده)
# در GUI، کاربر می‌تواند تعداد بیشتری iteration بدهد.
ACO_CONFIG = {
    "n_ants":     12,     # تعداد مورچه 
    "n_iter":     80,     # تعداد تکرار
    "alpha":      1.0,    # وزن فرومون
    "beta":       2.0,    # وزن دید (معکوس فاصله)
    "rho":        0.1,    # نرخ تبخیر فرومون
    "q_constant": 100.0,  # ثابت تقویت فرومون
}

GA_CONFIG = {
    "pop_size":        50,    # اندازه جمعیت
    "n_generations":   150,   # تعداد نسل‌ها
    "mutation_rate":   0.02,  # احتمال جهش
    "tournament_size": 5,     # اندازه تورنامنت
    "elite_count":     2,     # تعداد نخبگان
}

# آستانه‌ی seed — یک seed پایه برای تکرارپذیری کامل آموزش
BASE_SEED = 2024


# =============================================================================
# بخش ۱: موتور تولید داده — ساخت شهرهای مصنوعی
# =============================================================================

def _generate_synthetic_cities(
    n_cities:      int,
    lat_center:    float = 35.0,     # مرکز عرض جغرافیایی (پیش‌فرض: ایران)
    lon_center:    float = 50.0,     # مرکز طول جغرافیایی
    spread_km:     float = 500.0,    # شعاع پراکندگی شهرها به کیلومتر
    cluster_mode:  bool  = False,    # آیا شهرها به صورت خوشه‌ای باشند؟
    n_clusters:    int   = 3,        # تعداد خوشه‌ها (اگر cluster_mode=True)
    seed:          Optional[int] = None  # seed برای تکرارپذیری
) -> list[City]:
    """
    تولید لیستی از شهرهای مصنوعی با مختصات جغرافیایی تصادفی.

    Args:
        n_cities:     تعداد شهرهای مورد نیاز
        lat_center:   مرکز عرض جغرافیایی (درجه)
        lon_center:   مرکز طول جغرافیایی (درجه)
        spread_km:    شعاع پراکندگی کل به کیلومتر
        cluster_mode: اگر True باشد، شهرها به صورت خوشه‌ای تولید می‌شوند
        n_clusters:   تعداد خوشه‌های جداگانه (فقط در cluster_mode=True)
        seed:         عدد تصادفی برای تکرارپذیری

    Returns:
        list[City]: لیست شهرهای مصنوعی با اندیس‌گذاری صفر-پایه
    """
    # تنظیم seed برای تکرارپذیری نتایج — هر بار با همین seed، همان نقشه
    rng = random.Random(seed)

    # تبدیل شعاع کیلومتری به درجه‌ی جغرافیایی (تقریب ساده)
    # هر درجه‌ی عرض جغرافیایی ≈ ۱۱۱ کیلومتر
    # هر درجه‌ی طول جغرافیایی ≈ ۱۱۱ × cos(lat) کیلومتر
    lat_spread_deg = spread_km / 111.0
    lon_spread_deg = spread_km / (111.0 * max(0.1, math.cos(math.radians(lat_center))))

    cities = []  # لیست شهرهای خروجی

    if not cluster_mode:
        # ─── حالت ۱: توزیع یکنواخت (Uniform Distribution) ───
        # شهرها به صورت کاملاً تصادفی در یک مستطیل جغرافیایی پراکنده می‌شوند.
        # این حالت نقشه‌ی کشورهایی مثل ایران یا آلمان را شبیه‌سازی می‌کند.
        for i in range(n_cities):
            lat = rng.uniform(
                lat_center - lat_spread_deg,    # حد جنوبی
                lat_center + lat_spread_deg     # حد شمالی
            )
            lon = rng.uniform(
                lon_center - lon_spread_deg,    # حد غربی
                lon_center + lon_spread_deg     # حد شرقی
            )
            cities.append(City(index=i, name=f"City_{i}", lat=lat, lon=lon))

    else:
        # ─── حالت ۲: توزیع خوشه‌ای (Clustered Distribution) ───
        # n_clusters مرکز تصادفی تعریف می‌کنیم. شهرها دور هر مرکز
        # با توزیع گاوسی (نرمال) پراکنده می‌شوند.
        # این حالت نقشه‌هایی با چند شهر بزرگ محور را شبیه‌سازی می‌کند.

      
        cluster_spread = spread_km / (n_clusters * 1.5)
        cluster_lat_deg = cluster_spread / 111.0
        cluster_lon_deg = cluster_spread / (111.0 * max(0.1, math.cos(math.radians(lat_center))))

        # تعیین مرکز هر خوشه به صورت تصادفی
        cluster_centers = [
            (
                rng.uniform(lat_center - lat_spread_deg * 0.7,
                            lat_center + lat_spread_deg * 0.7),
                rng.uniform(lon_center - lon_spread_deg * 0.7,
                            lon_center + lon_spread_deg * 0.7)
            )
            for _ in range(n_clusters)
        ]

        # توزیع شهرها بین خوشه‌ها — هر شهر به یک خوشه تصادفی تعلق می‌گیرد
        for i in range(n_cities):
            # انتخاب تصادفی مرکز خوشه برای این شهر
            center_lat, center_lon = rng.choice(cluster_centers)

            # پراکنده کردن شهر دور مرکز با توزیع گاوسی
            # gauss(mean, sigma) — sigma = یک‌سوم فاصله‌ی پراکندگی
            lat = rng.gauss(center_lat, cluster_lat_deg / 3.0)
            lon = rng.gauss(center_lon, cluster_lon_deg / 3.0)

            # مطمئن شوید مختصات در محدوده‌ی مجاز (-90 to 90, -180 to 180) هستند
            lat = max(-89.9, min(89.9, lat))
            lon = max(-179.9, min(179.9, lon))

            cities.append(City(index=i, name=f"City_{i}", lat=lat, lon=lon))

    return cities


# =============================================================================
# بخش ۲: استخراج ویژگی‌ها (Feature Engineering)
# =============================================================================

def extract_features(matrix: list[list[float]], cities: list[City]) -> dict[str, float]:
    """
    Args:
        matrix: ماتریس N×N فاصله (خروجی build_distance_matrix)
        cities: لیست شهرها (برای محاسبه‌ی مساحت محیطی)

    Returns:
        dict[str, float]: دیکشنری از نام ویژگی → مقدار عددی
    """
    n = len(matrix)  # تعداد شهرها 

    # ─── محاسبه‌ی تمام فاصله‌های جفتی (بدون قطر اصلی که صفر است) ───
    all_distances = [
        matrix[i][j]
        for i in range(n)
        for j in range(n)
        if i != j  # فاصله‌ی شهر با خودش صفر است و مفهومی ندارد
    ]

   
    if not all_distances:
        return {feat: 0.0 for feat in [
            "n_cities", "coord_variance", "dist_mean", "dist_std",
            "dist_cv", "density_proxy", "nn_ratio", "dist_skewness", "max_min_ratio"
        ]}

    # ─── ویژگی ۱: تعداد شهرها (n_cities) ───
    # مستقیم‌ترین عامل — پرولاگ فقط تا N=15 عمل می‌کند
    f_n_cities = float(n)

    # ─── ویژگی ۲: واریانس مختصات (coord_variance) ───
    # پراکندگی هندسی شهرها روی نقشه — ترکیب واریانس latitude و longitude
    # عدد بزرگ‌تر = شهرها پراکنده‌تر هستند
    lats = [c.lat for c in cities]
    lons = [c.lon for c in cities]
    lat_mean = sum(lats) / n
    lon_mean = sum(lons) / n
    lat_var  = sum((l - lat_mean) ** 2 for l in lats) / n
    lon_var  = sum((l - lon_mean) ** 2 for l in lons) / n
    f_coord_variance = lat_var + lon_var  # واریانس کل (جمع دو بُعد)

    # ─── ویژگی ۳: میانگین فواصل (dist_mean) ───
    dist_mean = sum(all_distances) / len(all_distances)
    f_dist_mean = dist_mean

    # ─── ویژگی ۴: انحراف معیار فواصل (dist_std) ───
    # پراکندگی فواصل — یکنواختی توزیع شهرها
    dist_variance = sum((d - dist_mean) ** 2 for d in all_distances) / len(all_distances)
    dist_std = math.sqrt(dist_variance)
    f_dist_std = dist_std

    # ─── ویژگی ۵: ضریب تغییرات (dist_cv = Coefficient of Variation) ───
    # عدد بالا = توزیع فاصله‌ها بسیار نامنظم (خوشه‌ای)
    # عدد پایین = توزیع یکنواخت‌تر
    f_dist_cv = (dist_std / dist_mean) if dist_mean > 0.0 else 0.0

    # ─── ویژگی ۶: پروکسی تراکم (density_proxy) ───
    # تخمین چگالی شهری: چند شهر به ازای هر واحد مساحت؟
    lat_range_km = (max(lats) - min(lats)) * 111.0    # تبدیل درجه به کیلومتر
    lon_range_km = (max(lons) - min(lons)) * 111.0 * math.cos(math.radians(lat_mean))
    approx_area_km2 = max(lat_range_km * lon_range_km, 1.0)  # حداقل ۱ km² برای جلوگیری از تقسیم بر صفر
    f_density_proxy = n / math.sqrt(approx_area_km2)   # ریشه‌ی مربعی برای کاهش مقیاس

    # ─── ویژگی ۷: نسبت نزدیک‌ترین همسایه (nn_ratio) ───
    # برای هر شهر، فاصله‌ی نزدیک‌ترین همسایه را پیدا می‌کنیم.
    # میانگین این فاصله‌ها تقسیم بر میانگین کل فواصل = nn_ratio.

    nn_distances = []  # فاصله‌ی نزدیک‌ترین همسایه برای هر شهر
    for i in range(n):
        # کوچک‌ترین فاصله‌ی غیرصفر (همسایه‌ی نزدیک‌ترین)
        nn_dist = min(matrix[i][j] for j in range(n) if j != i)
        nn_distances.append(nn_dist)
    nn_mean = sum(nn_distances) / n
    f_nn_ratio = (nn_mean / dist_mean) if dist_mean > 0.0 else 1.0

    # ─── ویژگی ۸: کجی توزیع فاصله‌ها (dist_skewness) ───
    # کجی مثبت = اکثر فاصله‌ها کوچک‌اند، تعداد کمی فاصله‌ی بسیار زیاد دارند
    # کجی نزدیک به صفر = توزیع متقارن (نقشه‌ی یکنواخت‌تر)
    if dist_std > 0.0:
        # فرمول استاندارد کجی Pearson
        dist_skewness = (
            sum((d - dist_mean) ** 3 for d in all_distances) / len(all_distances)
        ) / (dist_std ** 3)
    else:
        dist_skewness = 0.0
    # محدود کردن مقدار کجی برای جلوگیری از outlier های شدید
    f_dist_skewness = max(-5.0, min(5.0, dist_skewness))

    # ─── ویژگی ۹: نسبت بیشترین به کمترین فاصله (max_min_ratio) ───
    # اگر این نسبت خیلی بزرگ باشد، شهرهای بسیار نزدیک و بسیار دور
    # همزمان وجود دارند — نشانه‌ی ساختار پیچیده‌تر (خوشه‌ای)
    dist_min = min(all_distances)
    dist_max = max(all_distances)
    f_max_min_ratio = (dist_max / dist_min) if dist_min > 0.0 else dist_max

    # ─── بسته‌بندی همه‌ی ویژگی‌ها در یک دیکشنری ───
    return {
        "n_cities":       f_n_cities,        # تعداد شهرها
        "coord_variance": f_coord_variance,  # واریانس مختصات جغرافیایی
        "dist_mean":      f_dist_mean,       # میانگین فواصل (km)
        "dist_std":       f_dist_std,        # انحراف معیار فواصل
        "dist_cv":        f_dist_cv,         # ضریب تغییرات
        "density_proxy":  f_density_proxy,   # پروکسی تراکم شهری
        "nn_ratio":       f_nn_ratio,        # نسبت میانگین نزدیک‌ترین همسایه
        "dist_skewness":  f_dist_skewness,   # کجی توزیع فاصله‌ها
        "max_min_ratio":  f_max_min_ratio,   # نسبت بیشترین به کمترین فاصله
    }


# =============================================================================
# بخش ۳: اجرای الگوریتم‌ها و محاسبه‌ی برنده
# =============================================================================

def _run_solvers_and_label(
    matrix:  list[list[float]],
    cities:  list[City],
    n:       int,
    seed:    int
) -> Optional[dict]:
    """
    Args:
        matrix: ماتریس N×N فواصل
        cities: لیست شهرها (برای extract_features)
        n:      تعداد شهرها (برای تصمیم‌گیری درباره‌ی پرولاگ)
        seed:   seed تصادفی برای تکرارپذیری

    Returns:
        dict یا None: یک ردیف داده شامل ویژگی‌ها + label + نتایج خام الگوریتم‌ها
        (None اگر همه الگوریتم‌ها شکست بخورند — نباید اتفاق بیفتد)
    """
    # ─── اجرای الگوریتم‌ها ───
    results = {}   # نتایج هر الگوریتم
    times   = {}   # زمان اجرای هر الگوریتم (ثانیه)

    # --- الگوریتم ۱: پرولاگ ---
    prolog_cost = None    # None یعنی پرولاگ اجرا نشد یا ناموجود بود
    prolog_time = None

    if n <= PROLOG_MAX_CITIES:
        # پرولاگ فقط برای N های کوچک که از آستانه‌ی امنیتی پایین‌تر هستند
        t_start = time.perf_counter()
        prolog_result = solve_with_prolog(matrix)  # تابع از prolog_bridge.py
        t_end = time.perf_counter()

        if prolog_result.status == PrologStatus.SOLVED:
            # پرولاگ با موفقیت حل کرد — نتیجه‌ی قطعی و بهینه
            prolog_cost = prolog_result.best_cost
            prolog_time = t_end - t_start
            logger.debug(f"  Prolog: cost={prolog_cost:.2f} km | time={prolog_time:.3f}s")
        else:
            # پرولاگ موجود نیست یا خطا داد — این الگوریتم را نادیده می‌گیریم
            logger.debug(f"  Prolog: {prolog_result.status.value} — نادیده گرفته شد")

    # --- الگوریتم ۲: ACO (کلونی مورچگان) ---
    t_start = time.perf_counter()
    aco_solver = AntColonyOptimizer(
        matrix      = matrix,
        n_ants      = ACO_CONFIG["n_ants"],
        n_iter      = ACO_CONFIG["n_iter"],
        alpha       = ACO_CONFIG["alpha"],
        beta        = ACO_CONFIG["beta"],
        rho         = ACO_CONFIG["rho"],
        q_constant  = ACO_CONFIG["q_constant"],
        seed        = seed    # seed برای تکرارپذیری
    )
    aco_result = aco_solver.solve(callback=None)  # بدون callback (نه GUI)
    t_end = time.perf_counter()
    aco_time = t_end - t_start
    logger.debug(f"  ACO:    cost={aco_result.best_cost:.2f} km | time={aco_time:.3f}s")

    # --- الگوریتم ۳: GA (الگوریتم ژنتیک) ---
    t_start = time.perf_counter()
    ga_solver = GeneticAlgorithmSolver(
        matrix          = matrix,
        pop_size        = GA_CONFIG["pop_size"],
        n_generations   = GA_CONFIG["n_generations"],
        mutation_rate   = GA_CONFIG["mutation_rate"],
        tournament_size = GA_CONFIG["tournament_size"],
        elite_count     = GA_CONFIG["elite_count"],
        seed            = seed + 1    # seed کمی متفاوت برای GA تا تنوع داشته باشیم
    )
    ga_result = ga_solver.solve(callback=None)  # بدون callback (نه GUI)
    t_end = time.perf_counter()
    ga_time = t_end - t_start
    logger.debug(f"  GA:     cost={ga_result.best_cost:.2f} km | time={ga_time:.3f}s")

    # ═══════════════════════════════════════════════════════════════
    # تعیین برنده با سیاست دو-مرحله‌ای (Tiered Decision Policy)
    # ═══════════════════════════════════════════════════════════════


    scores: dict[str, float] = {}   # امتیاز نهایی هر الگوریتم (برای دیباگ/CSV)
    winner_name: str

    # ─── سطح ۱: آیا پرولاگ در دسترس و در زمان معقول حل کرده؟ ───
    prolog_is_decisive = (
        prolog_cost is not None
        and prolog_time is not None
        and prolog_time <= PROLOG_TIME_CEILING_SEC
    )

    if prolog_is_decisive:
        # ─── قانون مستقیم: پرولاگ = جواب قطعاً بهینه، پس برنده است ───
        winner_name = "Prolog"

        # امتیازهای نمایشی (فقط برای ثبت در CSV/دیباگ — در تصمیم‌گیری
        # نقشی ندارند). به پرولاگ امتیاز کامل و به بقیه بر اساس کیفیت
        # نسبی‌شان امتیاز نسبی می‌دهیم.
        candidate_costs_for_display = {"ACO": aco_result.best_cost, "Genetic": ga_result.best_cost, "Prolog": prolog_cost}
        worst_cost = max(candidate_costs_for_display.values())
        for algo_name, cost_val in candidate_costs_for_display.items():
            scores[algo_name] = 1.0 if algo_name == "Prolog" else round(1.0 - (cost_val / worst_cost if worst_cost > 0 else 0.0), 4) * 0.5

    else:
        # ─── سطح ۲: فقط ACO و GA رقابت می‌کنند (پرولاگ غیرفعال/خیلی کند) ───
        candidate_costs = {
            "ACO":     aco_result.best_cost,
            "Genetic": ga_result.best_cost,
        }
        candidate_times = {
            "ACO":     aco_time,
            "Genetic": ga_time,
        }

        max_cost = max(candidate_costs.values())

        for algo_name in candidate_costs:
            norm_cost = candidate_costs[algo_name] / max_cost if max_cost > 0 else 1.0

            # نرمال‌سازی زمان با سقف ثابت (نه نسبت به max بین ۲ کاندیدا)
            # این کار باعث می‌شود اگر GA چند برابر کندتر از ACO باشد،
            # امتیاز سرعتش به صفر مطلق سقوط نکند و فقط به نسبت واقعی‌اش
            # از سقف معقول جریمه شود.
            norm_time = min(1.0, candidate_times[algo_name] / TIME_NORMALIZATION_CAP_SEC)

            score = (
                COST_WEIGHT_LARGE_N * (1.0 - norm_cost) +
                TIME_WEIGHT_LARGE_N * (1.0 - norm_time)
            )
            scores[algo_name] = score

        # ─── پاداش ساختاری برای GA در نقشه‌های مناسب آن ───
        coord_var = extract_features(matrix, cities)["coord_variance"]
        # نرمال‌سازی تقریبی واریانس به بازه‌ی [0,1] با یک مقیاس‌گذاری ساده
        # (واریانس‌های مشاهده‌شده در دیتاست معمولاً بین ۰ تا ~۸۰ هستند)
        scattered_score = min(1.0, coord_var / 50.0)
        ga_bonus = 0.05 * scattered_score
        scores["Genetic"] += ga_bonus

        winner_name = max(scores, key=lambda k: scores[k])

    # تبدیل نام برنده به برچسب عددی (label)
    label_map = {"Prolog": 0, "ACO": 1, "Genetic": 2}
    label = label_map[winner_name]

    # ─── ساختن ردیف داده برای دیتاست ───
    # ابتدا ویژگی‌های هندسی را استخراج می‌کنیم
    features = extract_features(matrix, cities)

    # سپس اطلاعات اجرا و نتایج را اضافه می‌کنیم
    row = {
        **features,                              # ویژگی‌های ۹‌گانه
        "label":          label,                 # برچسب برنده (0/1/2)
        "winner":         winner_name,           # نام برنده (برای دیباگ)
        "prolog_cost":    prolog_cost if prolog_cost is not None else -1.0,
        "prolog_time":    prolog_time if prolog_time is not None else -1.0,
        "aco_cost":       aco_result.best_cost,
        "aco_time":       aco_time,
        "ga_cost":        ga_result.best_cost,
        "ga_time":        ga_time,
        "prolog_score":   scores.get("Prolog", -1.0),  # -1 اگر اجرا نشد
        "aco_score":      scores["ACO"],
        "ga_score":       scores["Genetic"],
        "decision_tier":  "prolog_decisive" if prolog_is_decisive else "speed_quality_balance",
    }

    return row



# =============================================================================
# بخش ۴: موتور اصلی تولید دیتاست
# =============================================================================

def generate_training_dataset(
    n_samples:    int = 300,    # تعداد کل نمونه‌ها
    min_n:        int = 5,      # حداقل تعداد شهرها
    max_n:        int = 20,     # حداکثر تعداد شهرها
    verbose:      bool = True   # آیا جزئیات نمایش داده شود؟
) -> list[dict]:
    """
    ═══════════════════════════════════════
    استراتژی تولید نمونه
    ═══════════════════════════════════════
    برای جلوگیری از یک‌نواختی و اطمینان از تنوع کافی:

    ۱. توزیع یکنواخت N: نمونه‌ها به طور مساوی بین مقادیر مختلف N
       (از min_n تا max_n) توزیع می‌شوند — تا مدل برای همه‌ی اندازه‌ها
       به اندازه کافی داده داشته باشد.

    ۲. تنوع نوع نقشه (بهبود‌یافته): سه سناریوی هندسی جداگانه با نسبت
       ۴۰٪ خوشه‌ای (clustered) / ۲۰٪ پراکنده‌ی شدید (high-scatter) /
       ۴۰٪ یکنواخت (uniform). سناریوی «پراکنده‌ی شدید» به‌طور خاص برای
       این اضافه شده که coord_variance بالا برود — دقیقاً محیطی که در
       آن GA با ERX باید روی ACO برتری نشان دهد، چون ACO در نقشه‌های با
       فاصله‌ی نامنظم بین شهرها بیشتر در کمینه‌ی محلی گیر می‌کند.

    ۳. تنوع جغرافیایی: مرکز نقشه‌ها در مناطق مختلف دنیا (با spread های
       متفاوت) انتخاب می‌شود — از نقشه‌های کوچک محلی تا نقشه‌های بین‌قاره‌ای.

    ۴. seeds ثابت: هر نمونه یک seed منحصربه‌فرد دارد تا دیتاست کاملاً
       تکرارپذیر باشد — هر بار که این اسکریپت اجرا شود، همان دیتاست تولید می‌شود.

    Args:
        n_samples:  تعداد کل نمونه‌های مورد نیاز
        min_n:      حداقل تعداد شهرها در هر نمونه
        max_n:      حداکثر تعداد شهرها در هر نمونه
        verbose:    اگر True باشد، پیشرفت به صورت زنده نمایش داده می‌شود

    Returns:
        list[dict]: دیتاست — هر آیتم یک ردیف داده شامل ویژگی‌ها + label
    """
    logger.info(f"🚀 شروع تولید دیتاست | {n_samples} نمونه | N ∈ [{min_n}, {max_n}]")

    dataset = []  # لیست ردیف‌های داده
    failed  = 0   # تعداد نمونه‌هایی که شکست خوردند (برای آمار)

    # ─── پیکربندی‌های جغرافیایی متنوع ───
    # هر تاپل: (lat_center, lon_center, max_spread_km, label)
    # مناطق مختلف دنیا برای تنوع بیشتر
    geo_configs = [
        (35.0,   50.0,  800.0,  "Middle East"),    # خاورمیانه
        (46.0,   10.0,  600.0,  "Europe"),          # اروپا
        (37.0,  127.0,  500.0,  "East Asia"),       # شرق آسیا
        (38.0,  -95.0, 1200.0,  "North America"),   # آمریکای شمالی
        (-15.0, -55.0,  900.0,  "South America"),   # آمریکای جنوبی
        (-25.0,  135.0, 700.0,  "Australia"),        # استرالیا
        (20.0,   20.0,  900.0,  "Africa"),           # آفریقا
        (20.0,   80.0,  700.0,  "South Asia"),       # جنوب آسیا
    ]

    # ─── حلقه‌ی اصلی تولید نمونه ───
    for sample_idx in range(n_samples):

        # seed منحصربه‌فرد برای این نمونه (تکرارپذیر)
        sample_seed = BASE_SEED + sample_idx * 7  # ضرب در 7 تا seed ها متنوع‌تر باشند

        # تعداد شهرها برای این نمونه (توزیع یکنواخت)
        rng_local = random.Random(sample_seed)
        n = rng_local.randint(min_n, max_n)

        # انتخاب پیکربندی جغرافیایی (چرخشی برای پوشش یکنواخت)
        geo_idx = sample_idx % len(geo_configs)
        lat_c, lon_c, max_spread, region_name = geo_configs[geo_idx]

        # شعاع پراکندگی تصادفی (بین 20٪ تا 100٪ از max_spread)
        spread_km = rng_local.uniform(max_spread * 0.2, max_spread)

        # ─── تنوع نوع نقشه (بهبود‌یافته برای رفع بایاس) ───

        scenario_roll = sample_idx % 5
        is_clustered    = scenario_roll in (0, 1)        # ۴۰٪ خوشه‌ای
        is_high_scatter = scenario_roll == 2              # ۲۰٪ پراکنده‌ی شدید
        n_clusters   = rng_local.randint(2, min(4, max(2, n // 3))) if is_clustered else 3

        # برای حالت پراکنده‌ی شدید، شعاع پراکندگی را تا ۱.۸ برابر max_spread
        # افزایش می‌دهیم تا واریانس مختصات (coord_variance) به‌طور قابل‌توجه
        # بالا برود — این دقیقاً ویژگی‌ای است که در پاداش GA استفاده می‌شود.
        if is_high_scatter:
            spread_km = rng_local.uniform(max_spread * 1.1, max_spread * 1.8)

        if verbose and (sample_idx % 50 == 0 or sample_idx == n_samples - 1):
            # نمایش پیشرفت هر ۵۰ نمونه
            scenario_label = (
                "خوشه‌ای" if is_clustered else
                "پراکنده‌ی شدید" if is_high_scatter else
                "یکنواخت"
            )
            print(
                f"  [{sample_idx + 1:4d}/{n_samples}] "
                f"N={n:2d} | {region_name:<15} | "
                f"{scenario_label}"
            )

        try:
            # ─── گام ۱: تولید شهرهای مصنوعی ───
            cities = _generate_synthetic_cities(
                n_cities     = n,
                lat_center   = lat_c,
                lon_center   = lon_c,
                spread_km    = spread_km,
                cluster_mode = is_clustered,
                n_clusters   = n_clusters,
                seed         = sample_seed
            )

            # ─── گام ۲: ساخت ماتریس فواصل ───
            matrix = build_distance_matrix(cities)

            # ─── گام ۳: اجرای الگوریتم‌ها و تعیین برنده ───
            row = _run_solvers_and_label(
                matrix = matrix,
                cities = cities,
                n      = n,
                seed   = sample_seed
            )

            # ─── گام ۴: اضافه کردن به دیتاست ───
            if row is not None:
                # اضافه کردن متادیتای تولید برای دیباگ
                row["sample_idx"]      = sample_idx
                row["region"]          = region_name
                row["spread_km"]       = spread_km
                row["is_clustered"]    = int(is_clustered)
                row["is_high_scatter"] = int(is_high_scatter)
                dataset.append(row)


        except Exception as e:
            # اگر یک نمونه شکست بخورد، آن را رد می‌کنیم ولی ادامه می‌دهیم
            failed += 1
            logger.warning(f"  ⚠️  نمونه {sample_idx} شکست خورد: {e}")
            continue

    # ─── آمار نهایی ───
    logger.info(
        f"✅ دیتاست آماده | "
        f"موفق: {len(dataset)} | "
        f"شکست‌خورده: {failed} | "
        f"نرخ موفقیت: {len(dataset) / n_samples * 100:.1f}%"
    )

    # نمایش توزیع برچسب‌ها
    label_counts = {}
    for row in dataset:
        lbl = row["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    logger.info("📊 توزیع برچسب‌ها:")
    for lbl, name in LABEL_NAMES.items():
        count = label_counts.get(lbl, 0)
        pct   = count / len(dataset) * 100 if dataset else 0
        logger.info(f"   {lbl} ({name}): {count} نمونه ({pct:.1f}%)")

    return dataset


# =============================================================================
# بخش ۵: آموزش مدل درخت تصمیم
# =============================================================================

def train_decision_tree(
    dataset:         list[dict],
    max_depth:       Optional[int] = None,  # عمق مجاز درخت (None = نامحدود)
    min_samples_leaf: int = 2,               # حداقل نمونه در هر برگ
    cv_folds:        int  = 5               # تعداد fold های اعتبارسنجی
) -> tuple:
    """
    آموزش مدل DecisionTreeClassifier با اعتبارسنجی متقاطع K-Fold.

    مراحل:
      ۱. آماده‌سازی ویژگی‌ها (X) و برچسب‌ها (y) از دیتاست
      ۲. نرمال‌سازی ویژگی‌ها با StandardScaler
      ۳. تقسیم به train/test (80/20)
      ۴. آموزش درخت تصمیم روی train
      ۵. اعتبارسنجی K-Fold برای ارزیابی پایداری مدل
      ۶. گزارش دقت روی test set
      ۷. بازآموزی روی کل داده (برای استفاده در production)

    Args:
        dataset:          دیتاست خروجی generate_training_dataset()
        max_depth:        حداکثر عمق درخت (None = بدون محدودیت)
        min_samples_leaf: حداقل نمونه در هر برگ (جلوگیری از overfitting)
        cv_folds:         تعداد fold های K-Fold CV

    Returns:
        tuple: (model, scaler, feature_names, metrics_dict)
               model:         مدل آموزش‌دیده بر روی کل داده
               scaler:        StandardScaler آموزش‌دیده
               feature_names: لیست نام ویژگی‌ها (به ترتیب)
               metrics:       دیکشنری معیارهای ارزیابی
    """
    logger.info("🌳 شروع آموزش درخت تصمیم...")

    # ─── آماده‌سازی ویژگی‌ها و برچسب‌ها ───
    # نام ویژگی‌هایی که وارد مدل می‌شوند (به همین ترتیب دقیق)
    # GUI باید این ترتیب را برای پیش‌بینی رعایت کند — در pkl ذخیره می‌شود
    feature_names = [
        "n_cities",       # ویژگی اصلی — تعداد شهرها
        "coord_variance", # پراکندگی مختصات جغرافیایی
        "dist_mean",      # میانگین فواصل
        "dist_std",       # انحراف معیار فواصل
        "dist_cv",        # ضریب تغییرات فواصل
        "density_proxy",  # پروکسی تراکم شهری
        "nn_ratio",       # نسبت میانگین نزدیک‌ترین همسایه
        "dist_skewness",  # کجی توزیع فاصله‌ها
        "max_min_ratio",  # نسبت بیشترین به کمترین فاصله
    ]

    # ساختن آرایه‌های numpy از دیتاست
    X = np.array([
        [row[feat] for feat in feature_names]
        for row in dataset
    ], dtype=np.float64)   # float64 برای دقت محاسباتی بیشتر

    y = np.array([row["label"] for row in dataset], dtype=np.int32)

    logger.info(f"  ابعاد داده: X={X.shape} | y={y.shape}")
    logger.info(f"  ویژگی‌ها: {feature_names}")

    # ─── بررسی کیفیت داده ───
    # اگر مقادیر NaN یا Inf داریم، مدل آموزش نمی‌بیند
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"  ⚠️  داده ناسالم: {nan_count} NaN | {inf_count} Inf — جایگزینی با صفر")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # ─── تقسیم Train / Test (80% / 20%) ───
    # stratify=y مطمئن می‌کند توزیع برچسب‌ها در هر دو بخش یکسان است
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,         # ۲۰٪ برای تست
        random_state = BASE_SEED,   # seed ثابت برای تکرارپذیری
        stratify     = y            # توزیع یکنواخت برچسب‌ها
    )
    logger.info(f"  تقسیم: Train={len(X_train)} | Test={len(X_test)}")

    # ─── نرمال‌سازی ویژگی‌ها (StandardScaler) ───

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform روی train
    X_test_scaled  = scaler.transform(X_test)        # فقط transform روی test

    # ─── آموزش درخت تصمیم روی train set ───
    model = DecisionTreeClassifier(
        max_depth         = max_depth,        # عمق مجاز (None = نامحدود)
        min_samples_leaf  = min_samples_leaf, # حداقل نمونه در هر برگ
        criterion         = "gini",           # معیار تقسیم: Gini impurity
        random_state      = BASE_SEED,        # seed ثابت برای تکرارپذیری
        class_weight      = "balanced"        # وزن‌دهی برای جبران imbalance
    )
    model.fit(X_train_scaled, y_train)

    logger.info(f"  درخت آموزش دید | عمق واقعی: {model.get_depth()} | برگ‌ها: {model.get_n_leaves()}")

    # ─── ارزیابی روی test set ───
    y_pred = model.predict(X_test_scaled)
    test_accuracy = float(np.mean(y_pred == y_test))
    logger.info(f"  دقت روی Test Set: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # گزارش کامل (precision, recall, F1 برای هر کلاس)
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
    print("\n" + "─" * 65)
    print("  📊 گزارش دقت کامل (Classification Report):")
    print("─" * 65)
    print(classification_report(
        y_test, y_pred,
        target_names = target_names,
        zero_division = 0  # اگر یک کلاس اصلاً پیش‌بینی نشد، به جای Warning صفر بگذار
    ))

    # ماتریس اشتباه (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    print("  ماتریس اشتباه (Confusion Matrix):")
    print("  " + "  ".join(f"{name[:6]:>8}" for name in target_names))
    for i, row_vals in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row_vals)
        print(f"  {target_names[i][:6]:>8}: {row_str}")
    print("─" * 65)

    # ─── اعتبارسنجی متقاطع K-Fold ───
    # برای ارزیابی پایداری مدل — آیا دقت به تقسیم‌بندی تصادفی حساس است؟
    logger.info(f"  در حال اجرای {cv_folds}-Fold Cross Validation...")

    # StratifiedKFold: در هر fold، توزیع برچسب‌ها یکسان است
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=BASE_SEED)

    # برای CV باید کل X را scale کنیم، اما به درستی (بدون data leakage)
    # اینجا برای سادگی از X کامل استفاده می‌کنیم — چون دیتاست سنتتیک است
    # و leakage در این context نگران‌کننده نیست
    X_all_scaled = scaler.transform(X)  # با scaler ای که روی X_train fit شد

    cv_scores = cross_val_score(
        model, X_all_scaled, y,
        cv      = cv,
        scoring = "accuracy"  # معیار: دقت کلاسیفیکیشن
    )
    cv_mean = float(cv_scores.mean())
    cv_std  = float(cv_scores.std())
    logger.info(f"  {cv_folds}-Fold CV: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"\n  اعتبارسنجی متقاطع ({cv_folds}-Fold):")
    print(f"  میانگین دقت: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  fold ها: " + " | ".join(f"{s:.4f}" for s in cv_scores))

    # ─── بازآموزی روی کل داده (for production) ───
    # مدل نهایی که در pkl ذخیره می‌شود باید از همه‌ی داده‌ها یاد گرفته باشد
    # (نه فقط 80٪ train). دقت را از CV برمی‌داریم (عادلانه‌تر از test).
    logger.info("  بازآموزی روی کل دیتاست (production model)...")
    X_all_scaled_fit = scaler.fit_transform(X)  # این بار scaler را دوباره روی کل داده fit می‌کنیم
    model_final = DecisionTreeClassifier(
        max_depth         = max_depth,
        min_samples_leaf  = min_samples_leaf,
        criterion         = "gini",
        random_state      = BASE_SEED,
        class_weight      = "balanced"
    )
    model_final.fit(X_all_scaled_fit, y)

    logger.info(
        f"  مدل نهایی آماده | "
        f"عمق: {model_final.get_depth()} | "
        f"برگ: {model_final.get_n_leaves()}"
    )

    # ─── اهمیت ویژگی‌ها (Feature Importance) ───
    importances = model_final.feature_importances_
    print("\n  اهمیت ویژگی‌ها (Feature Importance):")
    print("  ─" * 30)
    for feat_name, importance in sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True  # مهم‌ترین اول
    ):
        bar = "█" * int(importance * 30)  # نمودار ستونی ASCII
        print(f"  {feat_name:<20} {importance:.4f}  {bar}")

    # ─── جمع‌آوری معیارهای ارزیابی ───
    metrics = {
        "test_accuracy": test_accuracy,
        "cv_mean":       cv_mean,
        "cv_std":        cv_std,
        "cv_scores":     cv_scores.tolist(),
        "tree_depth":    model_final.get_depth(),
        "n_leaves":      model_final.get_n_leaves(),
        "feature_importances": dict(zip(feature_names, importances.tolist())),
        "n_training_samples": len(X),
        "label_distribution": {
            LABEL_NAMES[i]: int((y == i).sum())
            for i in range(len(LABEL_NAMES))
        }
    }

    return model_final, scaler, feature_names, metrics


# =============================================================================
# بخش ۶: ذخیره‌سازی مدل و دیتاست
# =============================================================================

def save_model(
    model:         DecisionTreeClassifier,
    scaler:        StandardScaler,
    feature_names: list[str],
    metrics:       dict,
    n_samples:     int,
    min_n:         int,
    max_n:         int
) -> str:
    """

    ساختار فایل pkl (یک dict):
    {
      "model":         DecisionTreeClassifier آموزش‌دیده (روی کل داده)
      "scaler":        StandardScaler آموزش‌دیده (برای پیش‌پردازش ویژگی‌های جدید)
      "feature_names": لیست نام ویژگی‌ها به ترتیب دقیق ورودی مدل
      "label_map":     {0: "Prolog", 1: "ACO", 2: "Genetic"}
      "prolog_max_n":  آستانه‌ی ایمنی پرولاگ (PROLOG_MAX_CITIES از prolog_bridge)
      "metrics":       معیارهای ارزیابی (دقت، CV، اهمیت ویژگی‌ها)
      "training_config": پیکربندی آموزش (تعداد نمونه، بازه‌ی N)
      "version":       شماره‌ی نسخه برای سازگاری آینده
    }

    GUI از این ساختار استفاده می‌کند:
      1. مدل را بارگذاری می‌کند
      2. feature_names را برای ساخت صحیح آرایه‌ی ویژگی می‌خواند
      3. scaler را برای نرمال‌سازی ویژگی‌های ورودی جدید اعمال می‌کند
      4. label_map را برای تبدیل عدد به نام الگوریتم استفاده می‌کند
      5. prolog_max_n را برای تصمیم‌گیری درباره‌ی پیشنهاد پرولاگ می‌خواند

    Args:
        model:         مدل آموزش‌دیده
        scaler:        scaler آموزش‌دیده
        feature_names: لیست نام ویژگی‌ها
        metrics:       معیارهای ارزیابی
        n_samples:     تعداد نمونه‌های استفاده‌شده
        min_n, max_n:  بازه‌ی N در دیتاست

    Returns:
        str: مسیر کامل فایل ذخیره‌شده
    """
    # مطمئن شوید پوشه‌ی models وجود دارد
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)

    # بسته‌بندی کامل برای ذخیره‌سازی
    model_package = {
        "model":          model,           # مدل اصلی (DecisionTreeClassifier)
        "scaler":         scaler,          # StandardScaler برای transform داده‌ی جدید
        "feature_names":  feature_names,   # ترتیب دقیق ویژگی‌ها (بسیار مهم!)
        "label_map":      LABEL_NAMES,     # {0: "Prolog (Exact)", 1: "ACO", 2: "Genetic"}
        "prolog_max_n":   PROLOG_MAX_CITIES,  # از prolog_bridge — برای تصمیم GUI
        "metrics":        metrics,          # دقت، CV، اهمیت ویژگی‌ها
        "training_config": {
            "n_samples":                  n_samples,
            "min_n":                      min_n,
            "max_n":                      max_n,
            # وزن‌های قدیمی برای سازگاری با کدهای دیگر که ممکن است این
            # کلیدها را بخوانند — اکنون برابر با COST/TIME_WEIGHT_LARGE_N
            "cost_weight":                COST_WEIGHT,
            "time_weight":                TIME_WEIGHT,
            # ─── متادیتای سیاست دو-مرحله‌ای جدید (برای شفافیت/دیباگ) ───
            "prolog_time_ceiling_sec":    PROLOG_TIME_CEILING_SEC,
            "cost_weight_large_n":        COST_WEIGHT_LARGE_N,
            "time_weight_large_n":        TIME_WEIGHT_LARGE_N,
            "time_normalization_cap_sec": TIME_NORMALIZATION_CAP_SEC,
            "aco_config":                 ACO_CONFIG,
            "ga_config":                  GA_CONFIG,
            "base_seed":                  BASE_SEED,
        },
        "version": "1.1.0",   # نسخه‌ی جدید — رفع بایاس کلاس با سیاست دو-مرحله‌ای
    }

    # ذخیره‌سازی با joblib (مناسب‌تر از pickle برای آبجکت‌های sklearn)
    joblib.dump(model_package, model_path, compress=3)  # فشرده‌سازی سطح ۳
    logger.info(f"✅ مدل ذخیره شد ← {model_path}")

    # نمایش اندازه‌ی فایل
    file_size_kb = os.path.getsize(model_path) / 1024
    logger.info(f"   اندازه‌ی فایل: {file_size_kb:.1f} KB")

    return model_path


def save_dataset_csv(dataset: list[dict]) -> str:

    os.makedirs(MODELS_DIR, exist_ok=True)
    csv_path = os.path.join(MODELS_DIR, DATASET_FILENAME)

    if not dataset:
        logger.warning("دیتاست خالی است — CSV ذخیره نشد.")
        return ""

    # نام ستون‌ها از کلیدهای اولین ردیف
    fieldnames = list(dataset[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    logger.info(f"✅ دیتاست CSV ذخیره شد ← {csv_path} ({len(dataset)} ردیف)")
    return csv_path


# =============================================================================
# بخش ۷: تابع ابزاری — آزمایش سریع مدل با نمونه‌های جدید
# =============================================================================

def quick_verify_model(model_path: str) -> None:
    """
    Args:
        model_path: مسیر فایل pkl مدل
    """
    logger.info(f"🔍 تأیید سریع مدل از: {model_path}")

    # بارگذاری کامل بسته‌ی مدل
    package = joblib.load(model_path)

    model_loaded  = package["model"]
    scaler_loaded = package["scaler"]
    feat_names    = package["feature_names"]
    label_map     = package["label_map"]
    prolog_max_n  = package["prolog_max_n"]
    metrics       = package["metrics"]

    print("\n" + "═" * 65)
    print("  🧪 تأیید سریع مدل بارگذاری‌شده")
    print("═" * 65)
    print(f"  نسخه:             {package.get('version', 'N/A')}")
    print(f"  تعداد ویژگی‌ها:    {len(feat_names)}")
    print(f"  آستانه‌ی پرولاگ:   N ≤ {prolog_max_n}")
    print(f"  دقت CV:            {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"  دقت Test Set:      {metrics['test_accuracy']:.4f}")

    # ─── سناریوهای آزمایشی ───
    # چند حالت نمونه که پیش‌بینی مدل باید منطقی باشد:
    test_scenarios = [
        {
            "description": "نقشه‌ی کوچک (N=6) — انتظار: پرولاگ یا ACO",
            "features": {
                "n_cities":       6.0,     # کوچک
                "coord_variance": 15.0,    # متوسط
                "dist_mean":      350.0,   # متوسط
                "dist_std":       120.0,
                "dist_cv":        0.34,
                "density_proxy":  0.08,
                "nn_ratio":       0.25,
                "dist_skewness":  0.3,
                "max_min_ratio":  8.0,
            }
        },
        {
            "description": "نقشه‌ی بزرگ یکنواخت (N=18) — انتظار: ACO یا GA",
            "features": {
                "n_cities":       18.0,    # بزرگ
                "coord_variance": 180.0,   # بالا
                "dist_mean":      800.0,   # بزرگ
                "dist_std":       200.0,
                "dist_cv":        0.25,
                "density_proxy":  0.04,
                "nn_ratio":       0.3,
                "dist_skewness":  0.1,
                "max_min_ratio":  12.0,
            }
        },
        {
            "description": "نقشه‌ی خوشه‌ای متوسط (N=12) — انتظار: GA یا ACO",
            "features": {
                "n_cities":       12.0,    # متوسط
                "coord_variance": 90.0,
                "dist_mean":      500.0,
                "dist_std":       350.0,   # انحراف معیار بالا = توزیع نامنظم
                "dist_cv":        0.70,    # ضریب تغییرات بالا = خوشه‌ای
                "density_proxy":  0.06,
                "nn_ratio":       0.15,    # شهرها به هم نزدیک‌اند (خوشه)
                "dist_skewness":  1.5,     # کجی بالا = توزیع نامتقارن
                "max_min_ratio":  35.0,    # بالا = فاصله‌های بسیار متفاوت
            }
        },
    ]

    print("\n  پیش‌بینی برای سناریوهای آزمایشی:")
    print("  " + "─" * 60)

    for scenario in test_scenarios:
        # ساخت آرایه‌ی ویژگی به ترتیب دقیق feat_names
        feat_vector = np.array(
            [[scenario["features"][f] for f in feat_names]],
            dtype=np.float64
        )

        # نرمال‌سازی با scaler بارگذاری‌شده
        feat_scaled = scaler_loaded.transform(feat_vector)

        # پیش‌بینی
        pred_label = model_loaded.predict(feat_scaled)[0]
        pred_proba = model_loaded.predict_proba(feat_scaled)[0]  # احتمال هر کلاس

        # نمایش نتیجه
        print(f"\n  سناریو: {scenario['description']}")
        print(f"  پیش‌بینی: {label_map[pred_label]}")
        for cls_idx, prob in enumerate(pred_proba):
            bar = "█" * int(prob * 20)
            print(f"    {label_map[cls_idx]:<25} {prob:.3f}  {bar}")

    print("═" * 65)


# =============================================================================
# بخش ۸: پردازش آرگومان‌های خط فرمان
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """
    تعریف و پردازش آرگومان‌های خط فرمان.

    نمونه استفاده:
      python train_model.py                          # اجرا با مقادیر پیش‌فرض
      python train_model.py --samples 500            # ۵۰۰ نمونه
      python train_model.py --min-n 5 --max-n 15    # فقط نقشه‌های کوچک
      python train_model.py --max-depth 8            # محدود کردن عمق درخت
      python train_model.py --verbose                # نمایش جزئیات بیشتر
    """
    parser = argparse.ArgumentParser(
        description="آموزش مدل AI Advisor برای پروژه TSP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال‌های استفاده:
  python train_model.py --samples 400 --verbose
  python train_model.py --max-depth 6 --samples 200
  python train_model.py --min-n 5 --max-n 18 --samples 350
        """
    )

    parser.add_argument(
        "--samples", type=int, default=300,
        help="تعداد نمونه‌های آموزشی (پیش‌فرض: ۳۰۰)"
    )
    parser.add_argument(
        "--min-n", type=int, default=5,
        help="حداقل تعداد شهرها در هر نمونه (پیش‌فرض: ۵)"
    )
    parser.add_argument(
        "--max-n", type=int, default=20,
        help="حداکثر تعداد شهرها در هر نمونه (پیش‌فرض: ۲۰)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="حداکثر عمق درخت تصمیم (پیش‌فرض: نامحدود)"
    )
    parser.add_argument(
        "--min-samples-leaf", type=int, default=2,
        help="حداقل نمونه در هر برگ درخت (پیش‌فرض: ۲)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="تعداد fold های K-Fold CV (پیش‌فرض: ۵)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="نمایش جزئیات بیشتر در حین تولید داده"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="بدون تأیید سریع مدل بعد از ذخیره"
    )

    return parser.parse_args()


# =============================================================================
# بخش ۹: تابع اصلی — جریان کامل فاز ۵
# =============================================================================

def main() -> int:
    """
    تابع اصلی — ارکستراسیون کامل فرآیند آموزش.

    جریان کار (Pipeline):
      ۱. پردازش آرگومان‌های خط فرمان
      ۲. اعتبارسنجی ورودی‌ها
      ۳. تولید دیتاست سنتتیک
      ۴. آموزش درخت تصمیم با اعتبارسنجی K-Fold
      ۵. ذخیره‌سازی مدل + دیتاست
      ۶. تأیید سریع صحت pkl
      ۷. گزارش نهایی

    Returns:
        int: کد خروج (۰ = موفق، ۱ = خطا)
    """
    args = _parse_args()

    # ─── نمایش بنر شروع ───
    print("\n" + "★" * 65)
    print("  🧠 فاز ۵ پروژه TSP — آموزش مدل AI Advisor")
    print("  (DecisionTree Classifier برای انتخاب هوشمند الگوریتم)")
    print("★" * 65)
    print(f"\n  پیکربندی آموزش:")
    print(f"    تعداد نمونه:     {args.samples}")
    print(f"    بازه‌ی N:         [{args.min_n}, {args.max_n}] شهر")
    print(f"    عمق درخت:        {args.max_depth or 'نامحدود'}")
    print(f"    حداقل برگ:       {args.min_samples_leaf}")
    print(f"    K-Fold CV:       {args.cv_folds}")
    print(f"    آستانه‌ی پرولاگ: N ≤ {PROLOG_MAX_CITIES}")
    print(f"    وزن کیفیت:       {COST_WEIGHT} | وزن سرعت: {TIME_WEIGHT}")
    print()

    # ─── اعتبارسنجی ورودی‌ها ───
    if args.min_n < 2:
        logger.error("❌ min_n باید حداقل ۲ باشد (TSP حداقل ۲ شهر نیاز دارد)")
        return 1
    if args.max_n < args.min_n:
        logger.error("❌ max_n باید بزرگ‌تر یا مساوی min_n باشد")
        return 1
    if args.samples < 30:
        logger.error("❌ برای آموزش معنادار، حداقل ۳۰ نمونه لازم است")
        return 1

    # هشدار: اگر max_n از آستانه‌ی پرولاگ بیشتر باشد، داده‌هایی وجود دارند
    # که پرولاگ اجرا نمی‌شود — این طبیعی است و مدل باید یاد بگیرد.
    if args.max_n > PROLOG_MAX_CITIES:
        logger.info(
            f"ℹ️  max_n={args.max_n} > آستانه‌ی پرولاگ ({PROLOG_MAX_CITIES}). "
            f"برای N های بزرگ‌تر از {PROLOG_MAX_CITIES}، پرولاگ بای‌پس می‌شود."
        )

    # ─── گام ۱: تولید دیتاست ───
    print("─" * 65)
    print("  📡 گام ۱: تولید دیتاست سنتتیک")
    print("─" * 65)

    t_data_start = time.perf_counter()
    dataset = generate_training_dataset(
        n_samples = args.samples,
        min_n     = args.min_n,
        max_n     = args.max_n,
        verbose   = args.verbose
    )
    t_data_end = time.perf_counter()

    print(f"\n  ✅ {len(dataset)} نمونه تولید شد در {t_data_end - t_data_start:.1f} ثانیه")

    if len(dataset) < 20:
        logger.error("❌ دیتاست خیلی کوچک است — آموزش متوقف شد")
        return 1

    # ─── گام ۲: آموزش مدل ───
    print("\n" + "─" * 65)
    print("  🌳 گام ۲: آموزش درخت تصمیم")
    print("─" * 65)

    t_train_start = time.perf_counter()
    model, scaler, feature_names, metrics = train_decision_tree(
        dataset          = dataset,
        max_depth        = args.max_depth,
        min_samples_leaf = args.min_samples_leaf,
        cv_folds         = args.cv_folds
    )
    t_train_end = time.perf_counter()

    print(f"\n  ✅ آموزش در {t_train_end - t_train_start:.1f} ثانیه تمام شد")

    # ─── گام ۳: ذخیره‌سازی ───
    print("\n" + "─" * 65)
    print("  💾 گام ۳: ذخیره‌سازی مدل و دیتاست")
    print("─" * 65)

    model_path = save_model(
        model         = model,
        scaler        = scaler,
        feature_names = feature_names,
        metrics       = metrics,
        n_samples     = args.samples,
        min_n         = args.min_n,
        max_n         = args.max_n
    )

    csv_path = save_dataset_csv(dataset)

    # ─── گام ۴: تأیید سریع (اختیاری) ───
    if not args.no_verify:
        print("\n" + "─" * 65)
        print("  🔍 گام ۴: تأیید سریع مدل بارگذاری‌شده")
        print("─" * 65)
        quick_verify_model(model_path)

    # ─── گزارش نهایی ───
    total_time = time.perf_counter() - t_data_start
    print("\n" + "★" * 65)
    print("  🎉 آموزش با موفقیت تمام شد!")
    print("★" * 65)
    print(f"\n  خلاصه‌ی نتایج:")
    print(f"    دیتاست:        {len(dataset)} نمونه")
    print(f"    دقت Test Set:  {metrics['test_accuracy']:.4f} ({metrics['test_accuracy'] * 100:.2f}%)")
    print(f"    دقت CV:        {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"    عمق درخت:      {metrics['tree_depth']}")
    print(f"    زمان کل:       {total_time:.1f} ثانیه")
    print(f"\n  فایل‌های خروجی:")
    print(f"    مدل:      {model_path}")
    if csv_path:
        print(f"    دیتاست:   {csv_path}")
    print(f"\n  برای استفاده در GUI:")
    print(f"    from gui import MainWindow  # مدل از models/ به‌صورت خودکار لود می‌شود")
    print("★" * 65 + "\n")

    return 0  # کد خروج موفق


# =============================================================================
# اجرای مستقیم فایل
# =============================================================================
if __name__ == "__main__":
    sys.exit(main())
