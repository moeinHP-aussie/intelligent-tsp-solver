# =============================================================================
# فایل: src/core.py

# این فایل سه مسئولیت اصلی داره:
#   ۱. دریافت مختصات شهرها از API آنلاین (GeoNames)
#   ۲. ذخیره‌سازی و خواندن داده‌ها از کَش محلی (آفلاین)
#   ۳. محاسبه فاصله‌های واقعی با فرمول هاورساین و ساخت ماتریس مجاورت
#
# =============================================================================

import os          
import json        
import math        # برای توابع ریاضی فرمول هاورساین (sin, cos, atan2, sqrt, radians)
import random      
import time        # برای اضافه کردن تأخیر بین درخواست‌ها 
import logging     
from typing import Optional  # برای type hint‌های واضح‌تر
from dataclasses import dataclass, field, asdict  # برای ساختار داده تمیز و خوانا

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

# =============================================================================
# پیکربندی سیستم ثبت وقایع (Logging)
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s → %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TSP.Core")  # لاگر اختصاصی این ماژول


# =============================================================================
# ساختار داده شهر 
# =============================================================================
@dataclass
class City:
    """
    ساختار داده برای نگهداری اطلاعات یک شهر.

    دلیل استفاده از dataclass:
      - خواناتر از dict ساده
      - قابل تبدیل به dict برای ذخیره JSON (با asdict)
      - قابل استفاده مستقیم در لیست‌های Python

    فیلدها:
      index (int):    شناسه عددی یکتا — کلید اصلی ارتباط با ماتریس و Prolog
      name  (str):    نام شهر (مثلاً "Tehran" یا "Rome")
      lat   (float):  عرض جغرافیایی (Latitude)  — بین -90 تا +90
      lon   (float):  طول جغرافیایی (Longitude) — بین -180 تا +180
    """
    index: int          # اندیس عددی (مهم برای Prolog)
    name:  str          
    lat:   float        
    lon:   float        


# =============================================================================
# ثابت‌های پروژه 
# =============================================================================

# مسیر پوشه کَش  
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR).lower() == "src" else _THIS_DIR
CACHE_DIR = os.path.join(_PROJECT_ROOT, "data_cache")

# شعاع زمین بر حسب کیلومتر — برای فرمول هاورساین
EARTH_RADIUS_KM = 6371.0

# تنظیمات GeoNames API
# ───────────────────────────────────────────────────────────────────────
# آدرس: https://www.geonames.org/export/web-services.html
# ───────────────────────────────────────────────────────────────────────
GEONAMES_BASE_URL = "http://api.geonames.org/searchJSON"
GEONAMES_USERNAME = "moein" 

# ───────────────────────────────────────────────────────────────────────
# API جایگزین
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# تنظیمات  درخواست‌های HTTP
HTTP_TIMEOUT_SECONDS = 15     # حداکثر زمان انتظار برای پاسخ API
API_RETRY_COUNT      = 3      # تعداد تلاش مجدد در صورت خطا
API_RETRY_DELAY_SEC  = 2.0    # تأخیر بین تلاش‌های مجدد (ثانیه)

# حداقل و حداکثر شهرهای قابل انتخاب
MIN_CITIES = 5
MAX_CITIES = 30


# =============================================================================
# بخش ۱: مدیریت کَش محلی (Cache Manager)
# =============================================================================

def _build_cache_filename(country_name: str, num_cities: int) -> str:
    """
    نام فایل کَش رو بر اساس نام کشور و تعداد شهر می‌سازه.

    مثال: country_name="Iran", num_cities=15 → "iran_15_cities.json"
    """
    # نام کشور رو lowercase و بدون فاصله می‌کنیم تا در همه OS ها کار کنه
    safe_name = country_name.lower().strip().replace(" ", "_")
    return f"{safe_name}_{num_cities}_cities.json"


def _ensure_cache_dir() -> None:
    """
    مطمئن میشه که پوشه data_cache وجود داره.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.debug(f"پوشه کَش تأیید شد: {CACHE_DIR}")


def save_cities_to_cache(cities: list[City], country_name: str,
                          num_cities: int) -> str:
    """
    لیست شهرها رو به فایل JSON در پوشه data_cache ذخیره می‌کنه.

    ساختار JSON ذخیره‌شده:
    {
        "country":    "Iran",
        "num_cities": 15,
        "cities": [
            {"index": 0, "name": "Tehran", "lat": 35.69, "lon": 51.42},
            {"index": 1, "name": "Isfahan", "lat": 32.66, "lon": 51.68},
            ...  
    
    """
    _ensure_cache_dir()

    # ساخت نام و مسیر کامل فایل
    filename  = _build_cache_filename(country_name, num_cities)
    filepath  = os.path.join(CACHE_DIR, filename)

    #  تبدیل هر City به dict
    payload = {
        "country":    country_name,
        "num_cities": num_cities,
        "cities": [asdict(city) for city in cities]  # asdict از dataclasses
    }

    # نوشتن فایل با encoding صحیح برای پشتیبانی از کاراکترهای فارسی
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ {num_cities} شهر کَش شد ← {filepath}")
    return filepath


def load_cities_from_cache(country_name: str, num_cities: int) -> Optional[list[City]]:
    """
    اگه فایل کَش برای این کشور و تعداد شهر وجود داشت، داده‌ها رو می‌خونه.
    اگه نبود، None برمی‌گردونه (علامت برای اینکه باید به API رجوع بشه).
    """
    filename = _build_cache_filename(country_name, num_cities)
    filepath = os.path.join(CACHE_DIR, filename)

    # بررسی وجود فایل
    if not os.path.isfile(filepath):
        logger.info(f"📂 کَش یافت نشد برای: {country_name} / {num_cities} شهر")
        return None

    # خواندن و پارس کردن JSON
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # بازسازی لیست City از دیکشنری‌های JSON
        cities = [
            City(
                index = item["index"],
                name  = item["name"],
                lat   = float(item["lat"]),
                lon   = float(item["lon"])
            )
            for item in payload["cities"]
        ]

        logger.info(f"✅ {len(cities)} شهر از کَش خوانده شد ← {filepath}")
        return cities

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # فایل کَش خراب است — نادیده می‌گیریم و از API می‌گیریم
        logger.warning(f"⚠️  فایل کَش خراب بود، دوباره از API می‌گیریم. خطا: {e}")
        return None


def list_cached_datasets() -> list[dict]:
    """
    Returns:
        list[dict]: هر آیتم شامل {filename, country, num_cities} است
    """
    _ensure_cache_dir()
    results = []

    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(CACHE_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            results.append({
                "filename":   fname,
                "country":    meta.get("country", "نامشخص"),
                "num_cities": meta.get("num_cities", 0),
                "filepath":   fpath
            })
        except Exception:
            pass  # فایل‌های خراب رو نادیده می‌گیریم

    return sorted(results, key=lambda x: x["country"])


# =============================================================================
# part2: اتصال به API و دریافت داده (Online Fetcher)
# =============================================================================

def _fetch_raw_geonames(country_code: str, max_rows: int = 100) -> list[dict]:
    """
    تابع داخلی — مستقیماً با GeoNames API صحبت می‌کنه.
    """
    # پارامترهای query برای GeoNames API
    params = {
        "country":      country_code,     # کد کشور
        "featureClass": "P",              # P = Populated Places (شهرها)
        "featureCode":  "PPL",            # PPL = Populated Place (شهر عمومی)
        "orderby":      "population",     # مرتب‌سازی بر اساس جمعیت
        "maxRows":      max_rows,         # تعداد نتایج
        "username":     GEONAMES_USERNAME,
        "type":         "json"
    }

    last_exception = None

    # حلقه تلاش مجدد (Retry Loop)
    for attempt in range(1, API_RETRY_COUNT + 1):
        try:
            logger.info(f"🌐 درخواست به GeoNames API — تلاش {attempt}/{API_RETRY_COUNT}")

            response = requests.get(
                GEONAMES_BASE_URL,
                params=params,
                timeout=HTTP_TIMEOUT_SECONDS
            )

            # بررسی کد وضعیت HTTP
            response.raise_for_status() 

            # پارس کردن پاسخ JSON
            data = response.json()

            if "status" in data:
                status_msg = data["status"].get("message", "خطای ناشناخته")
                raise ValueError(f"GeoNames API خطا داد: {status_msg}")

            # گرفتن لیست شهرها از پاسخ
            geonames_list = data.get("geonames", [])

            if not geonames_list:
                raise ValueError(f"هیچ شهری برای کد کشور '{country_code}' پیدا نشد.")

            logger.info(f"✅ {len(geonames_list)} شهر از API دریافت شد.")
            return geonames_list

        except (Timeout, ConnectionError) as e:
            # خطاهای شبکه — تلاش مجدد
            last_exception = e
            logger.warning(f"⚠️  خطای شبکه (تلاش {attempt}): {e}")
            if attempt < API_RETRY_COUNT:
                time.sleep(API_RETRY_DELAY_SEC)

        except RequestException as e:
            # سایر خطاهای HTTP — تلاش مجدد
            last_exception = e
            logger.warning(f"⚠️  خطای HTTP (تلاش {attempt}): {e}")
            if attempt < API_RETRY_COUNT:
                time.sleep(API_RETRY_DELAY_SEC)

        except ValueError:
            # خطای منطقی (نه کشور، نه شهر) — تلاش مجدد معنی نداره
            raise

    # اگه تمام تلاش‌ها شکست خورد
    raise RuntimeError(
        f"بعد از {API_RETRY_COUNT} تلاش، اتصال به GeoNames برقرار نشد. "
        f"آخرین خطا: {last_exception}"
    )


# نگاشت نام کامل کشورها به کد دو حرفی GeoNames
# — GUI ازش استفاده می‌کنه
COUNTRY_CODE_MAP: dict[str, str] = {
    # خاورمیانه
    "ایران":         "IR",
    "Iran":          "IR",
    "عربستان":       "SA",
    "Saudi Arabia":  "SA",
    "ترکیه":         "TR",
    "Turkey":        "TR",
    "Iraq":          "IQ",
    "عراق":          "IQ",

    # اروپا
    "Italy":         "IT",
    "ایتالیا":       "IT",
    "France":        "FR",
    "فرانسه":        "FR",
    "Germany":       "DE",
    "آلمان":         "DE",
    "Spain":         "ES",
    "اسپانیا":       "ES",
    "United Kingdom":"GB",
    "انگلستان":      "GB",

    # آمریکا
    "United States": "US",
    "آمریکا":        "US",
    "Brazil":        "BR",
    "برزیل":         "BR",

    # آسیا
    "China":         "CN",
    "چین":           "CN",
    "Japan":         "JP",
    "ژاپن":          "JP",
    "India":         "IN",
    "هند":           "IN",
    "Australia":     "AU",
    "استرالیا":      "AU",
}


def _fetch_raw_overpass(country_code: str, max_rows: int = 100) -> list[dict]:
    """
    تابع داخلی — دریافت شهرها از Overpass API (جایگزین  GeoNames).

    """
    overpass_query = f"""
    [out:json][timeout:30];
    area["ISO3166-1"="{country_code}"]["admin_level"="2"]->.country;
    (
      node["place"="city"](area.country);
      node["place"="town"](area.country);
    );
    out body {max_rows};
    """

    logger.info(f"🌍 تلاش با Overpass API (OpenStreetMap) برای کد کشور: {country_code}")

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": overpass_query},
            timeout=HTTP_TIMEOUT_SECONDS * 2  
        )
        response.raise_for_status()

        data = response.json()
        elements = data.get("elements", [])

        if not elements:
            raise ValueError(f"Overpass هم نتیجه‌ای برای '{country_code}' برنگردوند.")

        results = []
        for elem in elements:
            try:
                tags = elem.get("tags", {})
                name = (
                    tags.get("name:en") or    
                    tags.get("name")    or    
                    "نامشخص"
                ).strip()

                lat = float(elem["lat"])
                lon = float(elem["lon"])

                pop_str = tags.get("population", "0").replace(",", "").replace(".", "")
                population = int(pop_str) if pop_str.isdigit() else 0

                results.append({
                    "name": name,
                    "lat":  lat,
                    "lon":  lon,
                    "population": population
                })
            except (ValueError, KeyError, TypeError):
                continue

        results.sort(key=lambda x: x.get("population", 0), reverse=True)

        logger.info(f"✅ Overpass: {len(results)} شهر برگردوند.")
        return results

    except RequestException as e:
        raise RuntimeError(f"Overpass API هم در دسترس نبود: {e}")


def _fetch_cities_with_fallback(country_code: str, max_rows: int = 100) -> list[dict]:
    """
    تابع داخلی — ابتدا GeoNames امتحان می‌کنه، اگه شکست خورد Overpass رو امتحان می‌کنه.

    این الگوی Fallback Chain مطمئن می‌کنه که حتی اگه یک API قطع باشه،
    برنامه همچنان کار می‌کنه.

    Returns:
        list[dict]: لیست شهرها (از هر منبعی که موفق شده)
    """
    # --- تلاش اول: GeoNames ---
    try:
        return _fetch_raw_geonames(country_code, max_rows)
    except RuntimeError as e:
        logger.warning(f"⚠️  GeoNames شکست خورد: {e}")
        logger.info("🔄 تلاش با Overpass API به عنوان جایگزین...")

    # --- تلاش دوم: Overpass (OpenStreetMap) ---
    try:
        return _fetch_raw_overpass(country_code, max_rows)
    except (RuntimeError, ValueError) as e:
        raise RuntimeError(
            f"هر دو API (GeoNames و Overpass) در دسترس نیستند.\n"
            f"آخرین خطا: {e}\n"
            f"💡 راه‌حل: فایل کَش JSON رو مستقیماً بارگذاری کنید (حالت آفلاین)."
        )


def fetch_cities_online(
    country_name: str,
    num_cities:   int,
    use_cache:    bool = True
) -> list[City]:
    """
    تابع اصلی دریافت شهرها — با پشتیبانی از کَش.

    منطق کار:
      ۱. اگه use_cache=True باشه، اول کَش رو چک می‌کنه.
      ۲. اگه کَش پیدا شد، از اون برمی‌گردونه (حالت آفلاین).
      ۳. اگه نبود، به API وصل میشه (حالت آنلاین).
      ۴. بعد از دریافت آنلاین، نتیجه رو کَش می‌کنه.

    Args:
        country_name: نام کشور (فارسی یا انگلیسی — طبق COUNTRY_CODE_MAP)
        num_cities:   تعداد شهرهای مورد نیاز
        use_cache:    آیا از کَش استفاده بشه؟ (پیش‌فرض: True)

    Returns:
        list[City]: لیست نهایی شهرها با اندیس‌گذاری صحیح

    """
    # --- اعتبارسنجی ورودی‌ها ---
    country_name = country_name.strip()

    if country_name not in COUNTRY_CODE_MAP:
        available = ", ".join(sorted(set(COUNTRY_CODE_MAP.keys())))
        raise ValueError(
            f"کشور '{country_name}' شناخته نشد.\n"
            f"کشورهای موجود: {available}"
        )

    if not (MIN_CITIES <= num_cities <= MAX_CITIES):
        raise ValueError(
            f"تعداد شهرها باید بین {MIN_CITIES} و {MAX_CITIES} باشه. "
            f"مقدار ورودی: {num_cities}"
        )

    country_code = COUNTRY_CODE_MAP[country_name]
    logger.info(f"🗺️  درخواست: {country_name} ({country_code}) | {num_cities} شهر")

    # --- مرحله ۱: چک کردن کَش ---
    if use_cache:
        cached = load_cities_from_cache(country_name, num_cities)
        if cached is not None:
            return cached

    # --- مرحله ۲: دریافت از API (با fallback خودکار) ---
    # بیشتر از num_cities می‌گیریم تا بتونیم رندوم انتخاب کنیم
    fetch_count = min(max(num_cities * 5, 50), 500)
    raw_data = _fetch_cities_with_fallback(country_code, max_rows=fetch_count)

    # فیلتر کردن شهرهایی که مختصات معتبر دارن
    valid_raw = []
    for item in raw_data:
        try:
            lat = float(item["lat"])
            lon = float(item["lng"])  
            name = str(item.get("toponymName", item.get("name", "نامشخص"))).strip()

            # فیلتر کردن مختصات غیرمنطقی
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and name:
                valid_raw.append({"name": name, "lat": lat, "lon": lon})
        except (ValueError, KeyError, TypeError):
            continue  # این شهر رو نادیده می‌گیریم

    if len(valid_raw) < num_cities:
        raise ValueError(
            f"کافی نیست: API فقط {len(valid_raw)} شهر معتبر برگردوند "
            f"ولی {num_cities} شهر خواستیم."
        )

    # --- مرحله ۳: انتخاب تصادفی ---
    # از بین شهرهای دریافتی، به تعداد num_cities انتخاب می‌کنیم
    # random.sample مطمئن میشه که تکراری نباشه
    selected_raw = random.sample(valid_raw, num_cities)

    # --- مرحله ۴: ساخت آبجکت‌های City با اندیس‌گذاری عددی ---
    # اندیس عددی از ۰ شروع میشه — این اندیس در Prolog هم استفاده میشه
    cities: list[City] = [
        City(index=i, name=item["name"], lat=item["lat"], lon=item["lon"])
        for i, item in enumerate(selected_raw)
    ]

    # --- مرحله ۵: ذخیره در کَش ---
    if use_cache:
        save_cities_to_cache(cities, country_name, num_cities)

    return cities


def load_cities_from_file(filepath: str) -> list[City]:
    """
    بارگذاری مستقیم شهرها از یک فایل JSON کَش (برای GUI آفلاین).

    Args:
        filepath: مسیر کامل فایل JSON

    Returns:
        list[City]

    Raises:
        FileNotFoundError: اگه فایل وجود نداشته باشه
        ValueError: اگه فرمت فایل اشتباه باشه
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"فایل یافت نشد: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        cities = [
            City(
                index = item["index"],
                name  = item["name"],
                lat   = float(item["lat"]),
                lon   = float(item["lon"])
            )
            for item in payload["cities"]
        ]
        logger.info(f"✅ {len(cities)} شهر از فایل بارگذاری شد: {filepath}")
        return cities

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"فرمت فایل JSON نامعتبر است: {e}")


# =============================================================================
# بخش ۳: فرمول هاورساین و ساخت ماتریس فواصل (Haversine & Distance Matrix)
# =============================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    محاسبه فاصله کرویِ بین دو نقطه روی زمین با فرمول هاورساین.

    ══════════════════════════════════════════════════════════════
    فرمول هاورساین (Haversine Formula):

      a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
      c = 2 × atan2( √a, √(1−a) )
      d = R × c

    جایی که R شعاع زمین است (6371 کیلومتر).
    ══════════════════════════════════════════════════════════════
    Args:
        lat1, lon1: مختصات نقطه اول (درجه)
        lat2, lon2: مختصات نقطه دوم (درجه)

    Returns:
        float: فاصله به کیلومتر
    """
    # تبدیل از درجه به رادیان 
    phi1    = math.radians(lat1)   # عرض جغرافیایی نقطه اول به رادیان
    phi2    = math.radians(lat2)   # عرض جغرافیایی نقطه دوم به رادیان
    dphi    = math.radians(lat2 - lat1)  # اختلاف عرض جغرافیایی
    dlambda = math.radians(lon2 - lon1)  # اختلاف طول جغرافیایی

    # محاسبه جزء اصلی فرمول (a)
    a = (
        math.sin(dphi / 2.0) ** 2 +          # sin²(Δlat/2)
        math.cos(phi1) * math.cos(phi2) *     # cos(lat1) × cos(lat2)
        math.sin(dlambda / 2.0) ** 2          # × sin²(Δlon/2)
    )

    # زاویه مرکزی (c) بر حسب رادیان
    # atan2 نسبت به atan پایدارتره (از تقسیم بر صفر جلوگیری می‌کنه)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    # فاصله نهایی بر حسب کیلومتر
    distance_km = EARTH_RADIUS_KM * c

    return round(distance_km, 3)  # گرد کردن به ۳ رقم اعشار


def build_distance_matrix(cities: list[City]) -> list[list[float]]:  # بهینه سازی دسترسی به فاصله دو شهر در حد O(1)
    """
    ساخت ماتریس فواصل N×N برای لیست شهرهای داده‌شده.

    این ماتریس خوراک اصلی تمام الگوریتم‌های حل مسئله‌ست:
      - ACO (مورچگان): از این ماتریس برای محاسبه احتمالات استفاده می‌کنه
      - Genetic (ژنتیک): طول هر مسیر رو با جمع فواصل از این ماتریس می‌گیره
      - Prolog CLP(FD): این ماتریس به صورت facts به Prolog پاس داده میشه

    ویژگی‌های ماتریس خروجی:
      - matrix[i][i] = 0.0  (فاصله شهر با خودش صفره)
      - matrix[i][j] = matrix[j][i]  (ماتریس متقارن — جاده دوطرفه‌ست)
      - مقادیر به کیلومتر و با ۳ رقم اعشار

    طراحی برای Prolog:
      اندیس هر ردیف/ستون دقیقاً با City.index مطابقت داره.
      بعداً در تبدیل به Prolog facts می‌نویسیم:
        distance(0, 1, 1234.5).
        distance(1, 0, 1234.5).

    Args: cities

    Returns:
        list[list[float]]: ماتریس N×N فواصل
    """
    n = len(cities)

    if n < 2:
        raise ValueError("برای ساخت ماتریس، حداقل ۲ شهر لازم است.")

    logger.info(f"📐 در حال ساخت ماتریس {n}×{n} فواصل...")

    # مقداردهی اولیه ماتریس با صفر
    # از list comprehension استفاده می‌کنیم (نه numpy) تا وابستگی خارجی نداشته باشیم
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

    # محاسبه فواصل — فقط نیمه بالای مثلثی (چون ماتریس متقارنه)
    # این بهینه‌سازی تعداد محاسبات رو از n² به n(n-1)/2 کاهش میده
    for i in range(n):
        for j in range(i + 1, n):  # j > i → فقط نیمه بالا
            dist = haversine_km(
                cities[i].lat, cities[i].lon,
                cities[j].lat, cities[j].lon
            )
            # استفاده از تقارن برای پر کردن هر دو خانه
            matrix[i][j] = dist  # فاصله از شهر i به شهر j
            matrix[j][i] = dist  # فاصله از شهر j به شهر i (یکسانه)

    # آمار خلاصه برای دیباگ
    all_dists = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
    logger.info(
        f"✅ ماتریس فواصل آماده | "
        f"حداقل: {min(all_dists):.1f} km | "
        f"حداکثر: {max(all_dists):.1f} km | "
        f"میانگین: {sum(all_dists)/len(all_dists):.1f} km"
    )

    return matrix


def print_matrix_preview(cities: list[City], matrix: list[list[float]], max_show: int = 6) -> None:
    """
    نمایش  خلاصه ماتریس فواصل در ترمینال 
    """
    n   = len(cities)
    cap = min(n, max_show)

    header = "شهر".ljust(18) + "".join(cities[j].name[:10].ljust(12) for j in range(cap))
    print("\n" + "═" * len(header))
    print(f"  ماتریس فواصل (کیلومتر) — {n} شهر")
    print("═" * len(header))
    print("  " + header)
    print("  " + "─" * len(header))

    for i in range(cap):
        row_name = cities[i].name[:15].ljust(18)
        row_vals = "".join(f"{matrix[i][j]:<12.1f}" for j in range(cap))
        print(f"  {row_name}{row_vals}")

    if n > max_show:
        print(f"  ... (و {n - max_show} شهر دیگه)")

    print("═" * len(header) + "\n")


# =============================================================================
# بخش ۴: تابع اصلی ادغام‌شده (Main Entry for Data Layer)
# =============================================================================

def get_tsp_data(
    country_name: str,
    num_cities:   int,
    use_cache:    bool = True
) -> tuple[list[City], list[list[float]]]:
    """
    تابع واحد high-level که کل فرایند لایه داده رو مدیریت می‌کنه.

    این تابع توسط GUI، الگوریتم‌ها و ماژول ML فراخوانی میشه.
    """
    # گرفتن شهرها (آنلاین یا آفلاین)
    cities = fetch_cities_online(country_name, num_cities, use_cache=use_cache)

    # ساخت ماتریس فواصل
    matrix = build_distance_matrix(cities)

    return cities, matrix


# =============================================================================
# تست مستقیم این ماژول 
# =============================================================================
if __name__ == "__main__":
    """

    """
    import sys

    print("\n" + "★" * 60)
    print("  تست لایه داده — فاز ۱ پروژه TSP")
    print("★" * 60)

    # --- تست ۱: فرمول هاورساین ---
    print("\n[تست ۱] فرمول هاورساین:")
    # تهران ↔ تبریز (فاصله واقعی حدود ۵۵۰ کیلومتر)
    tehran_lat, tehran_lon = 35.6892, 51.3890
    tabriz_lat, tabriz_lon = 38.0800, 46.2919
    dist = haversine_km(tehran_lat, tehran_lon, tabriz_lat, tabriz_lon)
    print(f"  فاصله تهران ← تبریز: {dist:.1f} کیلومتر (انتظار: ~۵۵۰ km)")

    # پاریس ↔ لندن (فاصله واقعی حدود ۳۴۰ کیلومتر)
    paris_lat, paris_lon     = 48.8566,  2.3522
    london_lat, london_lon   = 51.5074, -0.1278
    dist2 = haversine_km(paris_lat, paris_lon, london_lat, london_lon)
    print(f"  فاصله پاریس ← لندن: {dist2:.1f} کیلومتر (انتظار: ~۳۴۰ km)")

    # --- تست ۲: کَش و API ---
    print("\n[تست ۲] دریافت داده (ایران، ۵ شهر):")
    try:
        cities, matrix = get_tsp_data("Iran", 5, use_cache=True)

        print(f"\n  شهرهای انتخاب‌شده:")
        for city in cities:
            print(f"    [{city.index}] {city.name:<20} lat={city.lat:.4f}  lon={city.lon:.4f}")

        print_matrix_preview(cities, matrix)

    except Exception as e:
        print(f"\n  ⚠️  خطا (احتمالاً عدم دسترسی به اینترنت): {e}")
        print("  → حالت آفلاین: لیست dataset های کَش‌شده:")
        for ds in list_cached_datasets():
            print(f"    • {ds['filename']}")

    # --- تست ۳: لیست کَش‌ها ---
    print("\n[تست ۳] dataset های موجود در کَش:")
    for ds in list_cached_datasets():
        print(f"  • {ds['country']} — {ds['num_cities']} شهر → {ds['filename']}")

    print("\n" + "★" * 60 + "\n")
