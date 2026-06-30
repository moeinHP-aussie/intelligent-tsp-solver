# =============================================================================
# فایل: src/prolog_bridge.py
# لایه‌ی واسط بین پایتون و حلگر دقیق پرولاگ 
#
# این فایل مسئول سه کار اصلی است:
#   ۱. مدیریت اتصال به موتور SWI-Prolog از طریق کتابخانه‌ی pyswip
#      (بارگذاری tsp_solver.pl، فرستادن ماتریس فاصله، گرفتن جواب)
#   ۲. اعمال «آستانه‌ی ایمنی» (Safety Threshold): اگر تعداد شهرها از
#      حدی بیشتر شود (N > 15)، اجرای پرولاگ به‌صورت کنترل‌شده بای‌پس
#      می‌شود — نه کرش می‌کند، نه برنامه را برای ساعت‌ها فریز می‌کند.
#   ۳. تولید پیام‌های واضح برای UI (PyQt6) که توضیح بدهد *چرا* پرولاگ
#      اجرا نشده — نه یک برچسب مبهم "Skipped"، بلکه توضیح کامل و آموزشی.
#

# =============================================================================

import os          # برای ساخت مسیر فایل tsp_solver.pl
import time        # برای اندازه‌گیری زمان اجرا در سطح پایتون (علاوه بر زمان داخلی پرولاگ)
import logging     
from enum import Enum                      # برای وضعیت دقیق و type-safe نتیجه
from typing import Optional               
from dataclasses import dataclass, field   # برای ساختار نتیجه‌ی پرولاگ

# لاگر اختصاصی این ماژول — هم‌خانواده با core.py و solvers.py
logger = logging.getLogger("TSP.PrologBridge")


# =============================================================================
# بخش ۱: ثابت‌های آستانه‌ی ایمنی 
# =============================================================================

PROLOG_MAX_CITIES = 15

# مسیر فایل پرولاگ  prolog_core
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROLOG_FILE_PATH = os.path.join(_THIS_DIR, "tsp_solver.pl")

# =============================================================================
# بخش ۲: وضعیت نتیجه (Status Enum) — برای ارتباط دقیق و type-safe با UI
# =============================================================================
class PrologStatus(Enum):
    """
    وضعیت ممکن بعد از تلاش برای اجرای حلگر پرولاگ.

    این enum به GUI اجازه می‌دهد دقیقاً بفهمد چه اتفاقی افتاده،
    بدون نیاز به parse کردن رشته‌های متنی مبهم.
    """
    SOLVED          = "solved"            # با موفقیت حل شد — جواب قطعی موجود است
    SKIPPED_SIZE    = "skipped_size"      # به‌خاطر N > آستانه، عمداً اجرا نشد
    ENGINE_UNAVAILABLE = "engine_unavailable"  # SWI-Prolog/pyswip در دسترس نبود
    RUNTIME_ERROR   = "runtime_error"     # خطای غیرمنتظره در حین اجرای پرولاگ


# =============================================================================
# بخش ۳: ساختار نتیجه‌ی پرولاگ 
# =============================================================================
@dataclass
class PrologSolverResult:
    """
    ساختار خروجی اختصاصی حلگر پرولاگ.

    چرا از SolverResult موجود در solvers.py استفاده نکردیم؟
      چون پرولاگ یک حالت خاص دارد که ACO/GA هرگز ندارند: «عمداً اجرا
      نشدن» (Skipped). این حالت نیاز به متادیتای اضافه برای UI دارد
      (پیام توضیحی + tooltip) که در ساختار مشترک GA/ACO جایی ندارد.
      برای همین یک ساختار جدا تعریف کردیم تا این دو دنیا با هم قاطی
      نشوند، اما لایه‌ی بنچمارک (فاز ۴) به‌راحتی می‌تواند از
      to_solver_result() برای تبدیل به فرمت مشترک استفاده کند.

    فیلدها:
      status:        وضعیت اجرا — یکی از مقادیر PrologStatus
      best_path:      بهترین مسیر (لیست اندیس شهرها) — خالی اگر status != SOLVED
      best_cost:      هزینه‌ی کل مسیر بهینه — 0.0 اگر status != SOLVED
      elapsed_sec:    زمان اجرای کل (شامل سربار pyswip) به ثانیه
      is_exact:       همیشه True وقتی status==SOLVED — یادآوری صریح که
                      این جواب، بر خلاف ACO/GA، تضمین‌شده بهینه است
      ui_message:     پیام کوتاه برای نمایش در ستون اصلی جدول/برچسب UI
      ui_tooltip:      توضیح کامل و آموزشی — برای حالت Skipped حیاتی است
                      تا کاربر آن را با کرش یا باگ اشتباه نگیرد
    """
    status:      PrologStatus
    best_path:   list[int] = field(default_factory=list)
    best_cost:   float     = 0.0
    elapsed_sec: float     = 0.0
    is_exact:    bool      = False
    ui_message:  str       = ""
    ui_tooltip:  str       = ""

    def to_solver_result(self):
        """
        تبدیل به ساختار مشترک SolverResult (همان که ACO/GA برمی‌گردانند)
        تا لایه‌ی بنچمارک فاز ۴ بتواند هر سه روش را یک‌دست رسم/مقایسه کند.

        نکته: این تابع import محلی solvers.py را انجام می‌دهد تا از
        وابستگی حلقوی (circular import) بین این دو ماژول جلوگیری شود.
        """
        from solvers import SolverResult  # import محلی — جلوگیری از حلقه‌ی وابستگی

        return SolverResult(
            algorithm   = "Prolog (Held-Karp)",
            best_path   = self.best_path,
            best_cost   = self.best_cost,
            history     = [],  # پرولاگ "تاریخچه‌ی همگرایی" ندارد — جواب یک‌مرحله‌ای است
            elapsed_sec = self.elapsed_sec,
            iterations  = 1    # از نظر مفهومی، پرولاگ فقط یک "محاسبه‌ی قطعی" دارد
        )


# =============================================================================
# بخش ۴: پیام‌های آموزشی برای UI (UX Messaging)
# =============================================================================

def _build_skipped_message(num_cities: int) -> tuple[str, str]:

    short_msg = f"رد شد (محدودیت اندازه: N={num_cities} > {PROLOG_MAX_CITIES})"

    tooltip = (
        f"پرولاگ به‌صورت خودکار برای این نقشه ({num_cities} شهر) اجرا نشد.\n\n"
        f"دلیل: پرولاگ در این پروژه نقش «حلگر دقیق» (Exact Solver) را بازی "
        f"می‌کند — یعنی همیشه ۱۰۰٪ بهینه‌ی مطلق را پیدا می‌کند. اما همین دقت "
        f"مطلق، هزینه‌ی زمانی نمایی (O(2^N)) به همراه دارد.\n\n"
        f"برای جلوگیری از فریز شدن کامل برنامه، آستانه‌ی ایمنی روی "
        f"N = {PROLOG_MAX_CITIES} شهر تعیین شده است. برای نقشه‌های بزرگ‌تر، "
        f"الگوریتم‌های ابتکاری (مورچگان و ژنتیک) وارد عمل می‌شوند که در "
        f"زمانی بسیار کوتاه‌تر، جوابی «نزدیک به بهینه» (نه قطعاً بهینه) "
        f"پیدا می‌کنند.\n\n"
        f"این یک محدودیت طراحی‌شده است، نه یک خطا یا کرش برنامه."
    )

    return short_msg, tooltip


def _build_engine_unavailable_message(reason: str) -> tuple[str, str]:
    """
    پیام مربوط به حالتی که اصلاً نمی‌توان به موتور SWI-Prolog وصل شد
    (مثلاً pyswip نصب نیست یا SWI-Prolog روی سیستم موجود نیست).
    """
    short_msg = "در دسترس نیست (خطای اتصال به موتور پرولاگ)"

    tooltip = (
        f"اتصال به موتور SWI-Prolog برقرار نشد.\n\n"
        f"جزئیات خطا: {reason}\n\n"
        f"لطفاً مطمئن شوید SWI-Prolog روی سیستم نصب است و کتابخانه‌ی "
        f"pyswip به‌درستی پیکربندی شده است. تا رفع این مشکل، فقط "
        f"الگوریتم‌های مورچگان و ژنتیک قابل اجرا هستند."
    )

    return short_msg, tooltip


# =============================================================================
# بخش ۵: مدیریت اتصال به موتور پرولاگ (Lazy Singleton Connection)
# =============================================================================
# اتصال به pyswip را فقط یک‌بار برقرار می‌کنیم و فایل tsp_solver.pl را
# فقط یک‌بار consult می‌کنیم — اتصال مجدد در هر بنچمارک هم کند است و
# هم بی‌فایده، چون موتور Prolog در طول کل عمر برنامه می‌تواند زنده بماند.

_prolog_engine = None   # نمونه‌ی سراسری Prolog() از pyswip — فقط یک‌بار ساخته می‌شود
_engine_ready  = False  # آیا تسک consult فایل .pl با موفقیت انجام شده؟


def _get_prolog_engine():
    """
    نمونه‌ی Prolog (از pyswip) را برمی‌گرداند — با الگوی Lazy Singleton.

    اگر قبلاً ساخته شده، همان نمونه را برمی‌گرداند (بدون consult دوباره).
    اگر نه، تلاش می‌کند pyswip را import و فایل .pl را consult کند.

    Raises:
        ImportError:   اگر کتابخانه‌ی pyswip نصب نباشد
        RuntimeError:  اگر SWI-Prolog در دسترس نباشد یا consult شکست بخورد
    """
    global _prolog_engine, _engine_ready

    if _prolog_engine is not None and _engine_ready:
        return _prolog_engine

    # --- مرحله ۱: import کتابخانه‌ی pyswip ---
    # این import را داخل تابع گذاشتیم (نه بالای فایل) تا اگر pyswip
    # نصب نبود، کل برنامه (و الگوریتم‌های ACO/GA) کرش نکنند — فقط
    # قابلیت پرولاگ غیرفعال می‌شود.
    try:
        from pyswip import Prolog
    except ImportError as e:
        raise ImportError(
            f"کتابخانه‌ی pyswip نصب نیست. برای فعال‌سازی حلگر پرولاگ، "
            f"دستور 'pip install pyswip' را اجرا کنید. خطای اصلی: {e}"
        ) from e

    # --- مرحله ۲: ساخت نمونه‌ی Prolog و consult فایل tsp_solver.pl ---
    try:
        prolog = Prolog()

        if not os.path.isfile(PROLOG_FILE_PATH):
            raise RuntimeError(
                f"فایل tsp_solver.pl پیدا نشد. مسیر مورد انتظار: {PROLOG_FILE_PATH}"
            )

        # تبدیل مسیر به فرمتی که Prolog می‌فهمد (اسلش رو به جلو، حتی در ویندوز)
        prolog_safe_path = PROLOG_FILE_PATH.replace("\\", "/")
        list(prolog.query(f"consult('{prolog_safe_path}')"))

        logger.info(f"✅ موتور SWI-Prolog متصل و tsp_solver.pl لود شد ← {PROLOG_FILE_PATH}")

        _prolog_engine = prolog
        _engine_ready  = True
        return _prolog_engine

    except Exception as e:
        # هر خطایی در این مرحله یعنی موتور پرولاگ قابل استفاده نیست
        _prolog_engine = None
        _engine_ready  = False
        raise RuntimeError(f"اتصال به SWI-Prolog یا consult فایل .pl شکست خورد: {e}") from e


def reset_prolog_engine() -> None:
    """
    اتصال فعلی به پرولاگ را کاملاً ریست می‌کند.

    کاربرد: عمدتاً برای تست‌ها یا اگر کاربر بخواهد بعد از تغییر دستی
    فایل tsp_solver.pl، بدون ری‌استارت کامل برنامه، دوباره لودش کند.
    """
    global _prolog_engine, _engine_ready
    _prolog_engine = None
    _engine_ready  = False
    logger.info("🔄 اتصال موتور پرولاگ ریست شد — consult بعدی از نو انجام می‌شود.")


# =============================================================================
# بخش ۶: تابع اصلی — solve_with_prolog()
# =============================================================================
def solve_with_prolog(matrix: list[list[float]]) -> PrologSolverResult:
    """
    نقطه‌ی ورود اصلی این ماژول — همان چیزی که GUI (در QThread) و
    لایه‌ی بنچمارک (فاز ۴) صدا می‌زنند.

    این تابع کل جریان را مدیریت می‌کند:
      ۱. چک کردن آستانه‌ی ایمنی (N > 15؟) → اگر بله، بای‌پس فوری
      ۲. اتصال (یا استفاده از اتصال موجود) به موتور پرولاگ
      ۳. فرستادن ماتریس فاصله با set_distance_matrix/1
      ۴. صدا زدن solve_tsp/4 و گرفتن جواب
      ۵. بسته‌بندی نتیجه در PrologSolverResult با پیام‌های UI مناسب

    Args:
        matrix: ماتریس N×N فاصله — دقیقاً همان خروجی
                core.build_distance_matrix()

    Returns:
        PrologSolverResult: نتیجه‌ی کامل به همراه وضعیت و پیام‌های UI
    """
    num_cities = len(matrix)
    logger.info(f"🧮 درخواست حل پرولاگ برای {num_cities} شهر...")

    # ─────────────────────────────────────────────────────────────
    # گام ۱: آستانه‌ی ایمنی — خط دفاعی اول (در سطح پایتون)
    # ─────────────────────────────────────────────────────────────
    # این چک، قبل از هر تلاشی برای اتصال به پرولاگ انجام می‌شود تا
    # حتی اگر pyswip/SWI-Prolog نصب نباشد، بای‌پس درست کار کند.
    if num_cities > PROLOG_MAX_CITIES:
        short_msg, tooltip = _build_skipped_message(num_cities)
        logger.warning(
            f"⏭️  پرولاگ بای‌پس شد | N={num_cities} > آستانه={PROLOG_MAX_CITIES}"
        )
        return PrologSolverResult(
            status      = PrologStatus.SKIPPED_SIZE,
            best_path   = [],
            best_cost   = 0.0,
            elapsed_sec = 0.0,
            is_exact    = False,
            ui_message  = short_msg,
            ui_tooltip  = tooltip
        )

    # ─────────────────────────────────────────────────────────────
    # گام ۲: تلاش برای اتصال به موتور پرولاگ
    # ─────────────────────────────────────────────────────────────
    try:
        prolog = _get_prolog_engine()
    except (ImportError, RuntimeError) as e:
        short_msg, tooltip = _build_engine_unavailable_message(str(e))
        logger.error(f"❌ موتور پرولاگ در دسترس نیست: {e}")
        return PrologSolverResult(
            status      = PrologStatus.ENGINE_UNAVAILABLE,
            ui_message  = short_msg,
            ui_tooltip  = tooltip
        )

    # ─────────────────────────────────────────────────────────────
    # گام ۳: اجرای واقعی — فرستادن ماتریس و گرفتن جواب
    # ─────────────────────────────────────────────────────────────
    start_time = time.perf_counter()

    try:
        # تبدیل ماتریس پایتون به سینتکس لیست پرولاگ به‌صورت متنی
        # مثال: [[0,10,15],[10,0,20],[15,20,0]]
        # این روش (ساخت رشته) از pyswip's Python-to-Prolog list marshalling
        # که گاهی با اعداد اعشاری مشکل دارد، مطمئن‌تر است.
        matrix_str = "[" + ",".join(
            "[" + ",".join(repr(float(val)) for val in row) + "]"
            for row in matrix
        ) + "]"

        # گام ۳-الف: بارگذاری ماتریس در حافظه‌ی پرولاگ
        list(prolog.query(f"set_distance_matrix({matrix_str})"))

        # گام ۳-ب: صدا زدن solve_tsp/4 و گرفتن اولین (و تنها) جواب
        query_str = (
            f"solve_tsp({num_cities}, BestPath, BestCost, ElapsedMs)"
        )
        results = list(prolog.query(query_str))

        if not results:
            # اگر پرولاگ هیچ جوابی برنگرداند (نباید عادی پیش بیاید)
            raise RuntimeError("پرولاگ هیچ جوابی برای solve_tsp/4 برنگرداند.")

        solution = results[0]
        best_path = [int(c) for c in solution["BestPath"]]
        best_cost = float(solution["BestCost"])
        prolog_elapsed_ms = float(solution["ElapsedMs"])

        elapsed_sec = time.perf_counter() - start_time

        logger.info(
            f"✅ پرولاگ حل کرد | هزینه={best_cost:.2f} کیلومتر | "
            f"زمان داخلی پرولاگ={prolog_elapsed_ms:.2f}ms | "
            f"زمان کل (با سربار pyswip)={elapsed_sec:.3f}s"
        )

        return PrologSolverResult(
            status      = PrologStatus.SOLVED,
            best_path   = best_path,
            best_cost   = best_cost,
            elapsed_sec = elapsed_sec,
            is_exact    = True,
            ui_message  = "حل شد ✅ (جواب قطعی و ۱۰۰٪ بهینه)",
            ui_tooltip  = (
                "این مسیر توسط حلگر دقیق پرولاگ (Held-Karp) به‌دست آمده و "
                "ریاضیاتاً تضمین‌شده بهینه‌ترین مسیر ممکن است — برخلاف "
                "ACO/GA که فقط تقریب سریع ارائه می‌دهند."
            )
        )

    except Exception as e:
        # هر خطای اجرایی غیرمنتظره (مثل خطای syntax در پرولاگ یا داده‌ی نامعتبر)
        elapsed_sec = time.perf_counter() - start_time
        logger.error(f"❌ خطای اجرای پرولاگ: {e}")

        return PrologSolverResult(
            status      = PrologStatus.RUNTIME_ERROR,
            elapsed_sec = elapsed_sec,
            ui_message  = "خطا در اجرای پرولاگ ⚠️",
            ui_tooltip  = f"یک خطای غیرمنتظره در حین اجرای حلگر پرولاگ رخ داد:\n{e}"
        )


# =============================================================================
# بخش ۷: تابع کمکی برای UI — متن دائمی توضیح‌دهنده‌ی نقش پرولاگ
# =============================================================================
def get_prolog_role_explanation() -> str:
    """
    متن ثابتی که در تب «مشاور هوشمند و اجرا» یا تب «بنچمارک» (فاز ۴)
    کنار بخش پرولاگ نمایش داده می‌شود — طبق خواسته‌ی فاز ۳ پروژه:
    «کاربر باید بفهمد دقیقاً چرا پرولاگ بخشی از این پروژه است».

    این تابع هیچ منطق محاسباتی ندارد — فقط محتوای ثابت UI را برمی‌گرداند
    تا متن آموزشی، یک‌بار در یک‌جا نوشته شود و در همه‌ی تب‌ها یکسان بماند.
    """
    return (
        "🔷 نقش پرولاگ در این پروژه:\n\n"
        "پرولاگ تنها بخشی از این برنامه است که «جواب قطعی و ۱۰۰٪ بهینه» "
        "(Ground Truth) را تضمین می‌کند. کلونی مورچگان و الگوریتم ژنتیک "
        "هر دو متاهیوریستیک (Metaheuristic) هستند — یعنی سریع‌اند اما هیچ "
        "تضمینی برای رسیدن به بهینه‌ی مطلق نمی‌دهند.\n\n"
        f"به همین دلیل، پرولاگ فقط تا N = {PROLOG_MAX_CITIES} شهر فعال است: "
        "برای نقشه‌های کوچک، می‌توانیم مطمئن باشیم GA/ACO چقدر به جواب "
        "واقعی نزدیک شده‌اند. برای نقشه‌های بزرگ‌تر، پرولاگ به‌صورت خودکار "
        "کنار می‌رود تا برنامه فریز نشود — این یک ویژگی طراحی‌شده است."
    )


# =============================================================================
# تست مستقیم این ماژول (اجرای مستقیم با python prolog_bridge.py)
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s → %(message)s",
        datefmt="%H:%M:%S"
    )

    print("\n" + "★" * 60)
    print("  تست فاز ۳ — پل ارتباطی پایتون ↔ پرولاگ (Held-Karp)")
    print("★" * 60)

    # --- تست ۱: نمونه‌ی استاندارد پروژه (۴ شهر — A,B,C,D) ---
    print("\n[تست ۱] نمونه‌ی استاندارد پروژه (انتظار: هزینه = 80):")
    INF = 100000.0
    sample_matrix = [
        [0,   10,  15,  INF],   # A
        [10,  0,   20,  25],    # B
        [15,  20,  0,   30],    # C
        [INF, 25,  30,  0],     # D
    ]
    result = solve_with_prolog(sample_matrix)
    print(f"  وضعیت:    {result.status.value}")
    print(f"  مسیر:     {result.best_path}")
    print(f"  هزینه:    {result.best_cost}")
    print(f"  پیام UI:  {result.ui_message}")

    # --- تست ۲: بای‌پس به‌خاطر اندازه‌ی بزرگ (N=20 > 15) ---
    print("\n[تست ۲] بای‌پس آستانه‌ی ایمنی (N=20):")
    big_matrix = [[0.0] * 20 for _ in range(20)]
    result2 = solve_with_prolog(big_matrix)
    print(f"  وضعیت:    {result2.status.value}")
    print(f"  پیام UI:  {result2.ui_message}")
    print(f"  Tooltip:\n{result2.ui_tooltip}")

    print("\n" + "★" * 60 + "\n")
