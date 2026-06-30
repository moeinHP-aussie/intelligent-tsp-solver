# =============================================================================
# فایل: src/main.py
# نقش: (Entry Point) — فاز ۴ پروژه TSP

# =============================================================================

import os
import sys
import logging
import joblib   

# =============================================================================
# بخش ۱: تنظیم مسیرها (sys.path Bootstrap)
# =============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# =============================================================================
# بخش ۲: تنظیم Logging سراسری
# =============================================================================
def _setup_logging() -> logging.Logger:
    """
    پیکربندی سیستم لاگ سراسری برنامه و برگرداندن لاگر اختصاصی main.py.

    سطح INFO برای حالت عادی مناسبه. اگه متغیر محیطی TSP_DEBUG=1 ست
    شده باشه، سطح رو به DEBUG می‌بریم تا جزئیات بیشتری از solvers.py
    و prolog_bridge.py (مثل iteration به iteration ACO/GA) دیده بشه.
    """
    debug_mode = os.environ.get("TSP_DEBUG", "0") == "1"
    level = logging.DEBUG if debug_mode else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s  [%(levelname)s]  %(name)s → %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("TSP.Main")


# =============================================================================
# بخش ۳: بررسی سلامت محیط اجرا 
# =============================================================================
# هدف: قبل از باز شدن پنجره‌ی PyQt6، اگه یک وابستگی حیاتی (مثل
# PyQt6 یا requests) نصب نباشه، یک پیام فارسی واضح در ترمینال چاپ
# کنیم — به‌جای اینکه کاربر یک ImportError خام و گیج‌کننده وسط
# اجرای GUI ببینه.

def _check_critical_dependencies(logger: logging.Logger) -> bool:
    """
    بررسی وابستگی‌های *حیاتی* (نه اختیاری) برنامه.

    Returns:
        bool: True اگه همه چیز آماده باشه، False اگه یک وابستگی
              حیاتی گم باشه (در این صورت main() نباید ادامه بده).
    """
    missing: list[str] = []

    try:
        import PyQt6  # noqa: F401
    except ImportError:
        missing.append("PyQt6")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")

    try:
        import requests  # noqa: F401
    except ImportError:
        missing.append("requests")

    if missing:
        logger.error(
            "❌ وابستگی‌های حیاتی زیر نصب نیستند: " + ", ".join(missing)
        )
        print(
            "\n" + "═" * 64 +
            "\n  ❌ خطای راه‌اندازی — وابستگی نصب نشده\n" + "═" * 64 +
            f"\n\n  کتابخانه(های) زیر برای اجرای برنامه لازم‌اند ولی پیدا نشدن:\n" +
            "\n".join(f"     • {pkg}" for pkg in missing) +
            "\n\n  راه‌حل: دستور زیر را در ترمینال اجرا کنید:\n" +
            "     pip install -r requirements.txt\n" +
            "\n" + "═" * 64 + "\n"
        )
        return False

    # --- بررسی وجود فایل tsp_solver.pl کنار prolog_bridge.py ---
    # این فقط یک *هشدار* است، نه خطای بحرانی — چون طبق طراحی
    # prolog_bridge.py، نبودِ این فایل فقط یعنی الگوریتم پرولاگ در
    # دسترس نیست (ENGINE_UNAVAILABLE)؛ ACO و GA بدون مشکل کار می‌کنن.
    prolog_file = os.path.join(_THIS_DIR, "tsp_solver.pl")
    if not os.path.isfile(prolog_file):
        logger.warning(
            f"⚠️  فایل tsp_solver.pl کنار main.py پیدا نشد "
            f"(مسیر انتظاری: {prolog_file}). "
            f"الگوریتم Prolog در دسترس نخواهد بود، اما ACO و Genetic "
            f"بدون مشکل اجرا می‌شوند."
        )

    return True


# =============================================================================
# بخش ۴: نقطه‌ی ورود اصلی (main)
# =============================================================================
def main() -> int:
    """
    تابع اصلی اجرای برنامه.

    جریان کار:
      ۱. تنظیم logging سراسری
      ۲. بررسی وابستگی‌های حیاتی — اگه ناقص بود، خروج تمیز با کد خطا
      ۳. import کردن gui.py *بعد* از چک‌های بالا — تا اگه PyQt6 نصب
         نباشه، خطای ImportError خام و وحشتناک وسط ترمینال چاپ نشه و
         به‌جاش پیام تمیز بالا دیده بشه
      ۴. ساخت QApplication و نمایش MainWindow (که خودش از قبل در
         gui.py تمام Worker/Thread ها رو مدیریت می‌کنه)

    Returns:
        int: کد خروج برنامه (۰ = موفق، غیر صفر = خطا) — برای sys.exit()
    """
    logger = _setup_logging()
    logger.info("🚀 شروع راه‌اندازی برنامه‌ی TSP Solver...")

    # ─── گام ۱: بررسی سلامت محیط ───
    if not _check_critical_dependencies(logger):
        return 1  #  اجرای بی‌فایده‌ی GUI ناقص رو متوقف می‌کنیم

    # ─── گام ۲: import تأخیری gui.py ───
    # عمداً این import رو بالای فایل نگذاشتیم؛ اگه PyQt6 نصب نباشه،
    # import gui.py همون لحظه با ImportError کرش می‌کرد، قبل از اینکه
    # حتی به پیام تمیز بالا برسیم. با تأخیر انداختنش به اینجا (بعد از
    # چک وابستگی‌ها)، مطمئنیم پیام خطای ما همیشه اول دیده میشه.
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QFont
        from gui import MainWindow
    except ImportError as e:
        # محافظت در برابر خطاهای ناشناخته‌ی import (مثل نسخه‌ی ناقص
        # نصب PyQt6) اینجا هم می‌گیریمش.
        logger.error(f"❌ خطای import غیرمنتظره: {e}", exc_info=True)
        print(f"\n❌ بارگذاری gui.py با خطا مواجه شد:\n{e}\n")
        return 1

    # ─── گام ۳: ساخت QApplication ───
    app = QApplication(sys.argv)
    app.setApplicationName("TSP Solver")
    app.setApplicationVersion("4.0")

    app.setFont(QFont("Segoe UI", 11))

    # ─── گام ۴: ساخت و نمایش پنجره‌ی اصلی ───
    # DataFetchWorker / SolverWorker / BenchmarkWorker است. اینجا
    # هیچ Worker یا QThread جدیدی نمی‌سازیم — فقط پنجره رو فرامی‌خونیم.
    window = MainWindow()
    window.show()

    logger.info("✅ پنجره‌ی اصلی نمایش داده شد — برنامه آماده‌ی استفاده‌ست.")

    # ─── گام ۵: ورود به حلقه‌ی اصلی Qt (Event Loop) ───
    # app.exec() تا وقتی پنجره بسته نشه، اجرا رو مسدود می‌کنه و تمام
    # event های UI (کلیک دکمه، سیگنال‌های QThread و غیره) رو پردازش
    # می‌کنه. مقدار برگشتی‌اش کد خروج برنامه‌ست.
    return app.exec()


# =============================================================================
# بخش ۵: اجرای مستقیم فایل (python main.py)
# =============================================================================
if __name__ == "__main__":
    sys.exit(main())
