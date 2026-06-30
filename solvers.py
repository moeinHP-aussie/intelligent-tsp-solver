# =============================================================================
# فایل: src/solvers.py
#  هسته الگوریتم‌های حل مسئله 
#

#
#   ۱. کلونی مورچگان — Ant Colony Optimization (ACO)
#      فلسفه: هوش جمعی
#
#   ۲. الگوریتم ژنتیک — Genetic Algorithm (GA)
#      فلسفه: تکامل 
#      عملگر Crossover: Edge Recombination Crossover (ERX) — حرفه‌ای‌ترین
#      روش برای حفظ ساختار یال‌ها بین نسل‌ها
#
# طراحی مشترک مهم (برای QThread در فاز ۴):
#   هر دو الگوریتم یک پارامتر callback می‌پذیرند.
#   callback(iteration, best_path, best_cost) هر چند iteration فراخوانی می‌شود
#   تا GUI بتواند انیمیشن زنده نمایش دهد — بدون هیچ وابستگی به PyQt6 در این فایل.
# =============================================================================

import math       
import random     
import time       # برای اندازه‌گیری زمان اجرا
import logging   
import copy       # برای کپی عمیق لیست‌ها
from typing import Optional, Callable  # برای type hint‌های خوانا
from dataclasses import dataclass, field  # برای ساختار نتایج

# لاگر اختصاصی این ماژول
logger = logging.getLogger("TSP.Solvers")

# نوع تابع callback که GUI ارسال می‌کند
# امضا: callback(iteration: int, best_path: list[int], best_cost: float) -> None
CallbackFn = Callable[[int, list[int], float], None]


# =============================================================================
# ساختار داده نتیجه الگوریتم (SolverResult)
# =============================================================================
@dataclass
class SolverResult:
    """
    ساختار یکپارچه خروجی هر دو الگوریتم.

    این ساختار مشترک باعث می‌شود GUI و بنچمارک بتوانند
    نتایج ACO و GA را به‌صورت یکسان مقایسه کنند.

    فیلدها:
      algorithm:    نام الگوریتم ("ACO" یا "Genetic")
      best_path:    بهترین مسیر پیداشده (لیست اندیس‌های شهرها)
                    مثال: [0, 3, 1, 4, 2] — از شهر ۰ شروع و به ۰ برمی‌گردد
      best_cost:    هزینه (فاصله کل) بهترین مسیر به کیلومتر
      history:      تاریخچه بهبود: لیست (iteration, cost) برای رسم نمودار همگرایی
      elapsed_sec:  زمان اجرای کل الگوریتم به ثانیه
      iterations
    """
    algorithm:   str
    best_path:   list[int]
    best_cost:   float
    history:     list[tuple[int, float]] = field(default_factory=list)
    elapsed_sec: float = 0.0
    iterations:  int   = 0


# =============================================================================
# تابع کمکی: محاسبه هزینه یک مسیر کامل
# =============================================================================
def calculate_path_cost(path: list[int], matrix: list[list[float]]) -> float:
    """
    هزینه (فاصله کل) یک مسیر دوری را محاسبه می‌کند.

    مثال: path=[0,2,1,3] → d(0→2) + d(2→1) + d(1→3) + d(3→0)

    Args:
        path:   لیست اندیس شهرها (بدون تکرار شهر اول در آخر)
        matrix: ماتریس N×N فواصل

    Returns:
        float: مجموع فواصل به کیلومتر
    """
    n    = len(path)
    cost = 0.0

    for i in range(n):
        # شهر فعلی و شهر بعدی (برای آخرین شهر، بعدی همان اولی است)
        from_city = path[i]
        to_city   = path[(i + 1) % n]  # % n برای بسته شدن حلقه
        cost += matrix[from_city][to_city]

    return cost


# =============================================================================
# الگوریتم ۱: کلونی مورچگان (ACO — Ant Colony Optimization)
# =============================================================================
#
#  ┌─────────────────────────────────────────────────────────┐
#  │             منطق کامل ACO در یک نگاه                     │
#  │                                                         │
#  │  مقداردهی اولیه:                                          │
#  │    • ماتریس فرومون τ[i][j] = τ₀ (مقدار اولیه یکسان)         │
#  │    • ماتریس دید η[i][j] = 1 / d[i][j] (معکوس فاصله)      │
#  │                                                         │
#  │  هر Iteration:                                          │
#  │    ۱. هر مورچه یک تور کامل می‌سازد                        │
#  │       P(i→j) = [τ[i][j]^α × η[i][j]^β] / Σ(...)         │
#  │    ۲. فرومون‌ها تبخیر می‌شوند: τ *= (1-ρ)                  │
#  │    ۳. مورچه‌های موفق فرومون اضافه می‌کنند: τ += Q/L        │
#  │                                                        │
#  │  پارامترها:                                               │
#  │    α (alpha): وزن فرومون      — پیش‌فرض: 1.0            │
#  │    β (beta):  وزن فاصله       — پیش‌فرض: 2.0            │
#  │    ρ (rho):   نرخ تبخیر       — پیش‌فرض: 0.1             │
#  │    Q:         ثابت فرومون     — پیش‌فرض: 100             │
#  └─────────────────────────────────────────────────────────┘
#
class AntColonyOptimizer:
    """
    این پیاده‌سازی از مدل کلاسیک Ant System (AS) استفاده می‌کند
    که توسط Dorigo, Maniezzo, و Colorni در ۱۹۹۶ معرفی شد.

    ویژگی‌های این پیاده‌سازی:
      - ماتریس فرومون با مقداردهی اولیه heuristic (بر اساس greedy tour)
      - به‌روزرسانی فرومون elitist: فقط بهترین مورچه هر iteration فرومون می‌گذارد
      - پشتیبانی از callback برای انیمیشن زنده در GUI
    """

    def __init__(
        self,
        matrix:     list[list[float]],
        n_ants:     int   = 20,    # تعداد مورچه‌ها در هر تکرار
        n_iter:     int   = 200,   # تعداد کل تکرارها
        alpha:      float = 1.0,   # وزن فرومون (τ^α) — بالاتر = پیروی بیشتر از فرومون
        beta:       float = 2.0,   # وزن فاصله (η^β)  — بالاتر = ترجیح شهرهای نزدیک‌تر
        rho:        float = 0.1,   # نرخ تبخیر فرومون — بالاتر = فراموشی سریع‌تر
        q_constant: float = 100.0, # ثابت مقدار فرومون اضافه‌شده
        seed:       Optional[int] = 42  # seed برای تکرارپذیری نتایج
    ):
        """
        مقداردهی اولیه الگوریتم و تنظیم پارامترها.

        Args:
            matrix:     ماتریس N×N فواصل (خروجی core.py)
            n_ants:     تعداد مورچه در هر iteration
            n_iter:     تعداد کل تکرارها
            alpha:      توان فرومون در فرمول احتمال
            beta:       توان دید (معکوس فاصله) در فرمول احتمال
            rho:        نرخ تبخیر (بین ۰ و ۱)
            q_constant: ثابت تقویت فرومون
            seed:       عدد تصادفی برای نتایج قابل تکرار
        """
        # ذخیره ماتریس فواصل و ابعاد مسئله
        self.matrix  = matrix
        self.n       = len(matrix)          # تعداد شهرها
        self.n_ants  = n_ants
        self.n_iter  = n_iter
        self.alpha   = alpha
        self.beta    = beta
        self.rho     = rho
        self.q       = q_constant

        # تنظیم seed برای تکرارپذیری
        if seed is not None:
            random.seed(seed)

        # --- ساخت ماتریس دید (Visibility): η[i][j] = 1 / d[i][j] ---
        # مورچه‌ها به شهرهای نزدیک‌تر «دید بهتری» دارند
        # برای جلوگیری از تقسیم بر صفر (شهر با خودش)، مقدار ۰ می‌گذاریم
        self.eta: list[list[float]] = [
            [
                (1.0 / matrix[i][j]) if (i != j and matrix[i][j] > 0) else 0.0
                for j in range(self.n)
            ]
            for i in range(self.n)
        ]

        # --- مقداردهی اولیه ماتریس فرومون (τ₀) ---
        # روش heuristic: τ₀ = 1 / (n × L_greedy)
        # L_greedy = طول مسیر greedy (تخمین اولیه از طول مسیر بهینه)
        # این مقداردهی بهتر از عدد ثابت کار می‌کند
        greedy_cost = self._greedy_tour_cost()
        tau_initial = 1.0 / (self.n * greedy_cost) if greedy_cost > 0 else 1.0

        # ماتریس فرومون — مقدار اولیه یکسان برای همه یال‌ها
        self.tau: list[list[float]] = [
            [tau_initial] * self.n
            for _ in range(self.n)
        ]

        logger.info(
            f"ACO آماده | شهرها={self.n} | مورچه={n_ants} | "
            f"iter={n_iter} | α={alpha} β={beta} ρ={rho} | "
            f"τ₀={tau_initial:.6f}"
        )

    def _greedy_tour_cost(self) -> float:
        """
        محاسبه هزینه یک مسیر greedy ساده (نزدیک‌ترین همسایه).
        فقط برای مقداردهی اولیه فرومون استفاده می‌شود.
        """
        visited  = [False] * self.n
        path     = [0]          # از شهر ۰ شروع می‌کنیم
        visited[0] = True
        cost = 0.0

        for _ in range(self.n - 1):
            current   = path[-1]
            best_next = -1
            best_dist = math.inf

            # پیدا کردن نزدیک‌ترین شهر نرفته
            for j in range(self.n):
                if not visited[j] and self.matrix[current][j] < best_dist:
                    best_dist = self.matrix[current][j]
                    best_next = j

            if best_next == -1:
                break  # اگه شهری نمانده باشد

            path.append(best_next)
            visited[best_next] = True
            cost += best_dist

        # برگشت به شهر اول
        if path:
            cost += self.matrix[path[-1]][path[0]]

        return cost

    def _build_ant_tour(self) -> list[int]:
        """
        یک مورچه از شهر تصادفی شروع می‌کند و یک تور کامل می‌سازد.

        فرمول احتمال انتخاب شهر بعدی:
          P(i → j) = [τ(i,j)^α × η(i,j)^β] / Σ_{k ∈ allowed} [τ(i,k)^α × η(i,k)^β]

        این فرمول یک توازن بین:
          - Exploitation: رفتن به شهری که فرومون زیاد دارد (مسیر شناخته‌شده)
          - Exploration: رفتن به شهر نزدیک‌تر حتی اگه فرومونش کم باشد

        Returns:
            list[int]: تور کامل — ترتیب اندیس شهرها
        """
        # شهر شروع تصادفی (هر مورچه از جای متفاوتی شروع می‌کند)
        start = random.randint(0, self.n - 1)

        visited = [False] * self.n   # آیا شهر i بازدید شده؟
        tour    = [start]            # مسیر ساخته‌شده
        visited[start] = True

        current = start  # موقعیت فعلی مورچه

        # تا وقتی همه شهرها رفته نشده‌اند
        for _ in range(self.n - 1):

            # محاسبه وزن احتمالی هر شهر نرفته (صورت کسر فرمول)
            weights = []
            candidates = []

            for j in range(self.n):
                if not visited[j]:
                    # τ[current][j]^α × η[current][j]^β
                    pheromone = (self.tau[current][j] ** self.alpha)
                    visibility = (self.eta[current][j] ** self.beta)
                    weight = pheromone * visibility
                    weights.append(weight)
                    candidates.append(j)

            if not candidates:
                break  # نباید بیفتد ولی محافظت در برابر edge case

            # اگه همه وزن‌ها صفر باشند (edge case)، انتخاب تصادفی می‌کنیم
            total_weight = sum(weights)
            if total_weight == 0.0:
                next_city = random.choice(candidates)
            else:
                # انتخاب احتمالی با روش Roulette Wheel
                # (بدون numpy — با حلقه ساده)
                threshold = random.uniform(0.0, total_weight)
                cumulative = 0.0
                next_city  = candidates[-1]  # fallback به آخرین گزینه

                for city, weight in zip(candidates, weights):
                    cumulative += weight
                    if cumulative >= threshold:
                        next_city = city
                        break

            # حرکت مورچه به شهر انتخابی
            tour.append(next_city)
            visited[next_city] = True
            current = next_city

        return tour

    def _update_pheromones(self, all_tours: list[list[int]], all_costs: list[float]) -> None:
        """
        به‌روزرسانی ماتریس فرومون بعد از یک iteration.

        دو مرحله:
          ۱. تبخیر: همه فرومون‌ها ضرب در (1-ρ) می‌شوند → فراموشی تدریجی
          ۲. تقویت: فقط بهترین مورچه این iteration فرومون می‌گذارد (Elitist AS)

        فرمول تبخیر: τ[i][j] ← (1-ρ) × τ[i][j]
        فرمول تقویت: τ[i][j] ← τ[i][j] + Q / L_best   (برای یال‌های مسیر بهترین مورچه)

        روش Elitist (نه همه مورچه‌ها بلکه فقط بهترین):
          - همگرایی سریع‌تر
          - نتایج بهتر در TSP نسبت به Ant System کلاسیک

        Args:
            all_tours: لیست تورهای همه مورچه‌ها در این iteration
            all_costs: هزینه متناظر هر تور
        """
        # مرحله ۱: تبخیر همه فرومون‌ها
        for i in range(self.n):
            for j in range(self.n):
                self.tau[i][j] *= (1.0 - self.rho)
                # جلوگیری از رسیدن فرومون به صفر مطلق (حداقل مقدار)
                if self.tau[i][j] < 1e-10:
                    self.tau[i][j] = 1e-10

        # مرحله ۲: پیدا کردن بهترین مورچه این iteration
        best_idx  = min(range(len(all_costs)), key=lambda k: all_costs[k])
        best_tour = all_tours[best_idx]
        best_cost = all_costs[best_idx]

        # مقدار فرومون اضافه‌شده توسط بهترین مورچه
        delta_tau = self.q / best_cost if best_cost > 0 else 0.0

        # اضافه کردن فرومون روی یال‌های مسیر بهترین مورچه
        n_cities = len(best_tour)
        for k in range(n_cities):
            i = best_tour[k]
            j = best_tour[(k + 1) % n_cities]  # یال بین شهر k و k+1 (و آخر به اول)
            self.tau[i][j] += delta_tau
            self.tau[j][i] += delta_tau  # گراف غیر جهت‌دار — هر دو طرف

    def solve(
        self,
        callback:          Optional[CallbackFn] = None,
        callback_interval: int = 10   # هر چند iteration به GUI سیگنال بده
    ) -> SolverResult:
        """
        اجرای کامل الگوریتم ACO و برگرداندن بهترین نتیجه.

        Args:
            callback:          تابع callback برای GUI (انیمیشن زنده)
                               امضا: callback(iter, best_path, best_cost)
            callback_interval: هر چند iteration یک بار callback فراخوانی شود

        Returns:
            SolverResult: نتیجه کامل شامل بهترین مسیر، هزینه، و تاریخچه
        """
        start_time = time.perf_counter()

        # بهترین مسیر و هزینه‌ای که تاکنون دیده‌ایم
        global_best_path: list[int] = []
        global_best_cost: float     = math.inf
        history: list[tuple[int, float]] = []

        logger.info(f"▶ ACO شروع شد | {self.n_iter} iteration × {self.n_ants} مورچه")

        for iteration in range(1, self.n_iter + 1):

            # ساخت تور توسط هر مورچه
            all_tours: list[list[int]] = []
            all_costs: list[float]     = []

            for _ in range(self.n_ants):
                tour = self._build_ant_tour()
                cost = calculate_path_cost(tour, self.matrix)
                all_tours.append(tour)
                all_costs.append(cost)

            # به‌روزرسانی فرومون‌ها
            self._update_pheromones(all_tours, all_costs)

            # بهترین تور این iteration
            iter_best_idx  = min(range(len(all_costs)), key=lambda k: all_costs[k])
            iter_best_cost = all_costs[iter_best_idx]
            iter_best_tour = all_tours[iter_best_idx]

            # آیا بهبود جهانی داشتیم؟
            if iter_best_cost < global_best_cost:
                global_best_cost = iter_best_cost
                global_best_path = iter_best_tour[:]
                logger.debug(f"  iter {iteration}: بهبود جدید ← {global_best_cost:.2f} km")

            # ثبت در تاریخچه (برای نمودار همگرایی در فاز ۴)
            history.append((iteration, global_best_cost))

            # ارسال سیگنال به GUI (اگه callback داده شده)
            if callback and (iteration % callback_interval == 0 or iteration == self.n_iter):
                callback(iteration, global_best_path, global_best_cost)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"✅ ACO تمام شد | بهترین: {global_best_cost:.2f} km | "
            f"زمان: {elapsed:.2f} ثانیه"
        )

        return SolverResult(
            algorithm   = "ACO",
            best_path   = global_best_path,
            best_cost   = global_best_cost,
            history     = history,
            elapsed_sec = elapsed,
            iterations  = self.n_iter
        )


# =============================================================================
# الگوریتم ۲: الگوریتم ژنتیک (GA با ERX Crossover)
# =============================================================================
#
#  ┌─────────────────────────────────────────────────────────┐
#  │        Edge Recombination Crossover (ERX)               │
#  │                                                         │
#  │  ایده اصلی:                                              │
#  │    برای هر شهر، جدول یال‌های همسایه در هر دو والد           │
#  │    می‌سازیم. فرزند شهری را انتخاب می‌کند که کمترین          │
#  │    همسایه "استفاده‌نشده" داشته باشد (حریصانه).             │
#  │                                                         │
#  │  مثال:                                                   │
#  │    Parent1: [0,1,2,3,4]                                 │
#  │    Parent2: [0,2,4,1,3]                                 │
#  │                                                         │
#  │    Edge-list شهر 0: {1, 3, 2, 4} (همسایه‌ها در هر دو)    │
#  │    Edge-list شهر 1: {0, 2, 4, 3}                       │
#  │    ...                                                  │
#  │                                                         │
#  │  چرا ERX بهتر از Order Crossover (OX) است؟                │
#  │    OX ترتیب نسبی شهرها را حفظ می‌کند.                    │
#  │    ERX یال‌های (لبه‌های) خوب را حفظ می‌کند.                 │
#  │    در TSP، یال‌های کوتاه مهم‌تر از ترتیب نسبی هستند.          │
#  └─────────────────────────────────────────────────────────┘
#
class GeneticAlgorithmSolver:
    """
    پیاده‌سازی الگوریتم ژنتیک با ERX برای TSP.

    مراحل اصلی هر نسل:
      ۱. ارزیابی fitness همه کروموزوم‌ها (محاسبه هزینه)
      ۲. انتخاب والدین با Tournament Selection
      ۳. تولید فرزند با ERX Crossover
      ۴. جهش (Mutation) با روش 2-opt swap
      ۵. جایگزینی نسل قدیم با نسل جدید (Elitist)
    """

    def __init__(
        self,
        matrix:          list[list[float]],
        pop_size:         int   = 80,   # اندازه جمعیت
        n_generations:    int   = 300,  # تعداد نسل‌ها
        mutation_rate:    float = 0.02, # احتمال جهش هر کروموزوم
        tournament_size:  int   = 5,    # اندازه تورنامنت برای انتخاب والدین
        elite_count:      int   = 2,    # تعداد بهترین‌ها که مستقیماً به نسل بعد می‌روند
        seed:             Optional[int] = 42
    ):
        """
        مقداردهی اولیه الگوریتم ژنتیک.

        Args:
            matrix:         ماتریس N×N فواصل
            pop_size:       تعداد کروموزوم در هر نسل
            n_generations:  تعداد نسل‌ها
            mutation_rate:  احتمال جهش (بین ۰ و ۱)
            tournament_size: تعداد رقبا در هر تورنمنت انتخاب
            elite_count:    تعداد نخبگانی که بدون تغییر به نسل بعد می‌روند
            seed:           seed تصادفی
        """
        self.matrix         = matrix
        self.n              = len(matrix)   # تعداد شهرها
        self.pop_size       = pop_size
        self.n_generations  = n_generations
        self.mutation_rate  = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count    = elite_count

        if seed is not None:
            random.seed(seed)

        logger.info(
            f"GA آماده | شهرها={self.n} | جمعیت={pop_size} | "
            f"نسل={n_generations} | mutation={mutation_rate} | elite={elite_count}"
        )

    # ─────────────────────────────────────────────────────────────────
    # بخش ۱: مقداردهی اولیه جمعیت
    # ─────────────────────────────────────────────────────────────────
    def _create_initial_population(self) -> list[list[int]]:
        """
        ساخت جمعیت اولیه از کروموزوم‌های تصادفی.

        هر کروموزوم یک permutation از [0, 1, ..., n-1] است.
        یک کروموزوم Greedy هم اضافه می‌کنیم تا جمعیت از نقطه
        معقولی شروع کند (نه کاملاً تصادفی).

        Returns:
            list[list[int]]: جمعیت اولیه
        """
        population = []

        # یک مسیر greedy به عنوان اولین عضو (خوراک خوب برای شروع)
        greedy = self._create_greedy_chromosome()
        population.append(greedy)

        # بقیه کروموزوم‌ها تصادفی
        base = list(range(self.n))
        for _ in range(self.pop_size - 1):
            chromosome = base[:]     # کپی از [0,1,...,n-1]
            random.shuffle(chromosome)  # ترتیب تصادفی
            population.append(chromosome)

        return population

    def _create_greedy_chromosome(self) -> list[int]:
        """
        یک کروموزوم با استراتژی نزدیک‌ترین همسایه (Greedy/NN Heuristic) می‌سازد.
        معمولاً خیلی بهتر از تصادفی است و نقطه شروع خوبی برای GA فراهم می‌کند.
        """
        visited = [False] * self.n
        start   = random.randint(0, self.n - 1)  # شروع تصادفی
        path    = [start]
        visited[start] = True
        current = start

        for _ in range(self.n - 1):
            best_next = -1
            best_dist = math.inf

            for j in range(self.n):
                if not visited[j] and self.matrix[current][j] < best_dist:
                    best_dist = self.matrix[current][j]
                    best_next = j

            if best_next == -1:
                # اگه شهری نماند (نباید رخ دهد)، اولی نرفته را اضافه کن
                best_next = next(j for j in range(self.n) if not visited[j])

            path.append(best_next)
            visited[best_next] = True
            current = best_next

        return path

    # ─────────────────────────────────────────────────────────────────
    # بخش ۲: ارزیابی Fitness
    # ─────────────────────────────────────────────────────────────────
    def _evaluate_population(self, population: list[list[int]]) -> list[float]:
        """
        هزینه هر کروموزوم را محاسبه می‌کند.
        در TSP، fitness = 1/cost (هزینه کمتر = fitness بهتر).
        اما ما مستقیماً با cost کار می‌کنیم (min cost).

        Returns:
            list[float]: لیست هزینه‌ها (یک مقدار به ازای هر کروموزوم)
        """
        return [calculate_path_cost(chromo, self.matrix) for chromo in population]

    # ─────────────────────────────────────────────────────────────────
    # بخش ۳: انتخاب والدین (Tournament Selection)
    # ─────────────────────────────────────────────────────────────────
    def _tournament_select(
        self,
        population: list[list[int]],
        costs:      list[float]
    ) -> list[int]:
        """
        یک والد با Tournament Selection انتخاب می‌کند.

        روش: k کروموزوم تصادفی از جمعیت انتخاب کن، بهترین‌شان را برگردان.
        این روش:
          - ساده و کارا است
          - فشار انتخابی قابل تنظیم دارد (tournament_size بزرگ‌تر = فشار بیشتر)
          - از Roulette Wheel پایدارتر است (به scaling حساس نیست)

        Returns:
            list[int]: کروموزوم برنده تورنامنت
        """
        # k نمونه تصادفی از جمعیت (بدون تکرار)
        k         = min(self.tournament_size, len(population))
        indices   = random.sample(range(len(population)), k)

        # پیدا کردن بهترین (کمترین هزینه) در بین k نمونه
        winner_idx = min(indices, key=lambda i: costs[i])
        return population[winner_idx][:]  # کپی برمی‌گردانیم

    # ─────────────────────────────────────────────────────────────────
    # بخش ۴: Edge Recombination Crossover (ERX)
    # ─────────────────────────────────────────────────────────────────
    def _build_edge_table(self, p1: list[int], p2: list[int]) -> dict[int, set[int]]:
        """
        جدول یال‌ها (Edge Table) را برای ERX می‌سازد.

        برای هر شهر، مجموعه‌ای از همسایه‌هایش در هر دو والد را جمع می‌کنیم.

        مثال:
          Parent1: [0, 1, 2, 3, 4]
          Parent2: [0, 3, 1, 4, 2]

          edge_table[0] = {1, 4, 3}    ← همسایه‌های ۰ در P1: {1,4}، در P2: {3,4}
          edge_table[1] = {0, 2, 3, 4} ← همسایه‌های ۱ در P1: {0,2}، در P2: {3,4} ← اشتراک یعنی shared edge
          ...

        یال‌های مشترک بین دو والد با علامت خاص مشخص می‌شوند (ارزش بیشتر دارند).
        برای سادگی پیاده‌سازی، ما از set استفاده می‌کنیم (بدون وزن‌گذاری shared).

        Args:
            p1: کروموزوم والد اول
            p2: کروموزوم والد دوم

        Returns:
            dict[int, set[int]]: edge_table[city] = مجموعه همسایه‌ها
        """
        n = len(p1)
        edge_table: dict[int, set[int]] = {city: set() for city in range(n)}

        # ساخت جدول برای هر والد
        for parent in [p1, p2]:
            for idx in range(n):
                city = parent[idx]
                # همسایه چپ (با wrapping)
                left_neighbor  = parent[(idx - 1) % n]
                # همسایه راست (با wrapping)
                right_neighbor = parent[(idx + 1) % n]

                edge_table[city].add(left_neighbor)
                edge_table[city].add(right_neighbor)

        return edge_table

    def _erx_crossover(self, parent1: list[int], parent2: list[int]) -> list[int]:
        """
        تولید فرزند با Edge Recombination Crossover (ERX).

        الگوریتم ERX:
          ۱. ساخت edge_table از هر دو والد
          ۲. شهر شروع: شهر اول یکی از والدین (تصادفی)
          ۳. حلقه تا همه شهرها انتخاب شوند:
             a. شهر فعلی را به فرزند اضافه کن
             b. آن را از edge_table همه همسایه‌ها حذف کن
             c. بعدی: همسایه‌ای با کمترین تعداد همسایه باقی‌مانده انتخاب کن
                (اگه تساوی بود: تصادفی)
             d. اگه لیست همسایه خالی بود: یک شهر نرفته تصادفی انتخاب کن

        Args:
            parent1: کروموزوم والد اول
            parent2: کروموزوم والد دوم

        Returns:
            list[int]: کروموزوم فرزند
        """
        n          = len(parent1)
        edge_table = self._build_edge_table(parent1, parent2)

        # مجموعه شهرهای نرفته
        unvisited: set[int] = set(range(n))

        # شهر شروع: تصادفی بین شهر اول هر دو والد
        current = random.choice([parent1[0], parent2[0]])
        child   = [current]
        unvisited.remove(current)

        while len(child) < n:
            # حذف شهر فعلی از لیست همسایه همه شهرهای دیگر
            for neighbors in edge_table.values():
                neighbors.discard(current)  # discard بر خلاف remove خطا نمی‌دهد

            # انتخاب شهر بعدی از بین همسایه‌های شهر فعلی
            available_neighbors = edge_table[current] & unvisited  # اشتراک با نرفته‌ها

            if available_neighbors:
                # انتخاب همسایه با کمترین تعداد همسایه باقی‌مانده (حریصانه)
                next_city = min(
                    available_neighbors,
                    key=lambda c: len(edge_table[c] & unvisited)
                )
            else:
                # اگه هیچ همسایه‌ای نمانده: انتخاب تصادفی از شهرهای نرفته
                next_city = random.choice(list(unvisited))

            child.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        return child

    # ─────────────────────────────────────────────────────────────────
    # بخش ۵: جهش (Mutation) — روش 2-opt Swap
    # ─────────────────────────────────────────────────────────────────
    def _mutate(self, chromosome: list[int]) -> list[int]:
        """
        جهش با روش 2-opt swap (Swap Mutation).

        دو ایندکس تصادفی انتخاب می‌کنیم و بخش بینشان را معکوس می‌کنیم.
        این جهش ساختار مسیر را کمی تغییر می‌دهد بدون اینکه شهری تکرار شود.

        مثال:
          chromosome = [0, 1, 2, 3, 4, 5]
          i=1, j=4 → [0, 4, 3, 2, 1, 5]  (بخش ۱ تا ۴ معکوس شد)

        Args:
            chromosome: کروموزوم ورودی

        Returns:
            list[int]: کروموزوم بعد از جهش (یا بدون تغییر اگه احتمال جهش نگرفت)
        """
        # با احتمال mutation_rate جهش انجام می‌شود
        if random.random() > self.mutation_rate:
            return chromosome  # بدون تغییر

        n = len(chromosome)
        if n < 4:
            return chromosome  # برای مسیرهای خیلی کوچک جهش معنی ندارد

        # انتخاب دو نقطه تصادفی (i < j)
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)

        # معکوس کردن بخش chromosome[i:j+1]
        mutated = chromosome[:]
        mutated[i:j + 1] = reversed(mutated[i:j + 1])

        return mutated

    # ─────────────────────────────────────────────────────────────────
    # بخش ۶: حلقه اصلی الگوریتم
    # ─────────────────────────────────────────────────────────────────
    def solve(
        self,
        callback:          Optional[CallbackFn] = None,
        callback_interval: int = 10
    ) -> SolverResult:
        """
        اجرای کامل الگوریتم ژنتیک.

        منطق نسل‌به‌نسل:
          ۱. ارزیابی جمعیت فعلی
          ۲. کپی نخبگان مستقیم به نسل بعدی
          ۳. پر کردن بقیه جای با: انتخاب + ERX + جهش
          ۴. جایگزینی جمعیت

        Args:
            callback:          تابع callback برای GUI (انیمیشن زنده)
            callback_interval: هر چند نسل یک بار callback فراخوانی شود

        Returns:
            SolverResult: نتیجه کامل
        """
        start_time = time.perf_counter()

        # ساخت جمعیت اولیه
        population = self._create_initial_population()

        # بهترین جهانی
        global_best_path: list[int] = []
        global_best_cost: float     = math.inf
        history: list[tuple[int, float]] = []

        logger.info(
            f"▶ GA شروع شد | {self.n_generations} نسل × جمعیت {self.pop_size}"
        )

        for generation in range(1, self.n_generations + 1):

            # ─── مرحله ۱: ارزیابی جمعیت ───
            costs = self._evaluate_population(population)

            # ─── مرحله ۲: به‌روزرسانی بهترین جهانی ───
            gen_best_idx  = min(range(len(costs)), key=lambda k: costs[k])
            gen_best_cost = costs[gen_best_idx]
            gen_best_path = population[gen_best_idx]

            if gen_best_cost < global_best_cost:
                global_best_cost = gen_best_cost
                global_best_path = gen_best_path[:]
                logger.debug(f"  نسل {generation}: بهبود ← {global_best_cost:.2f} km")

            history.append((generation, global_best_cost))

            # ─── ارسال سیگنال به GUI ───
            if callback and (generation % callback_interval == 0 or generation == self.n_generations):
                callback(generation, global_best_path, global_best_cost)

            # ─── مرحله ۳: ساخت نسل جدید ───
            new_population: list[list[int]] = []

            # الیتیسم: مستقیم کپی کردن بهترین‌ها
            # ابتدا جمعیت را بر اساس هزینه مرتب می‌کنیم
            sorted_pop = sorted(zip(costs, population), key=lambda x: x[0])
            for _, elite_chromo in sorted_pop[:self.elite_count]:
                new_population.append(elite_chromo[:])

            # پر کردن بقیه با crossover و mutation
            while len(new_population) < self.pop_size:
                # انتخاب دو والد با tournament selection
                parent1 = self._tournament_select(population, costs)
                parent2 = self._tournament_select(population, costs)

                # تولید فرزند با ERX
                child = self._erx_crossover(parent1, parent2)

                # اعمال جهش
                child = self._mutate(child)

                new_population.append(child)

            # جایگزینی نسل قدیم با نسل جدید
            population = new_population

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"✅ GA تمام شد | بهترین: {global_best_cost:.2f} km | "
            f"زمان: {elapsed:.2f} ثانیه"
        )

        return SolverResult(
            algorithm   = "Genetic",
            best_path   = global_best_path,
            best_cost   = global_best_cost,
            history     = history,
            elapsed_sec = elapsed,
            iterations  = self.n_generations
        )


# =============================================================================
# تابع کمکی: نمایش نتیجه در ترمینال
# =============================================================================
def print_result(result: SolverResult, cities_names: Optional[list[str]] = None) -> None:
    """
    Args:
        result:       خروجی solve()
        cities_names: اگه داده شود، اندیس‌ها را به نام تبدیل می‌کند
    """
    print("\n" + "─" * 55)
    print(f"  الگوریتم: {result.algorithm}")
    print(f"  بهترین هزینه: {result.best_cost:,.2f} کیلومتر")
    print(f"  زمان اجرا:   {result.elapsed_sec:.3f} ثانیه")
    print(f"  تکرارها:     {result.iterations}")
    print("  بهترین مسیر:")

    if cities_names:
        path_str = " → ".join(cities_names[i] for i in result.best_path)
        # بازگشت به شهر اول
        path_str += f" → {cities_names[result.best_path[0]]}"
    else:
        path_str = " → ".join(str(i) for i in result.best_path)
        path_str += f" → {result.best_path[0]}"

    # نمایش در چند خط اگه مسیر بلند بود
    max_width = 60
    if len(path_str) > max_width:
        words = path_str.split(" → ")
        line = "    "
        for w in words:
            if len(line) + len(w) + 4 > max_width:
                print(line)
                line = "    " + w + " → "
            else:
                line += w + " → "
        if line.strip():
            print(line.rstrip(" → "))
    else:
        print(f"    {path_str}")

    print("─" * 55)


# =============================================================================
# تست مستقیم این ماژول
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from core import City, build_distance_matrix

    print("\n" + "★" * 60)
    print("  تست فاز ۲ — الگوریتم‌های ACO و GA")
    print("★" * 60)

    # شهرهای ایران برای تست
    cities = [
        City(0, "Tehran",   35.6892, 51.3890),
        City(1, "Isfahan",  32.6546, 51.6680),
        City(2, "Shiraz",   29.5918, 52.5836),
        City(3, "Tabriz",   38.0800, 46.2919),
        City(4, "Mashhad",  36.2605, 59.6168),
        City(5, "Ahvaz",    31.3183, 48.6706),
        City(6, "Kerman",   30.2839, 57.0834),
        City(7, "Rasht",    37.2809, 49.5832),
    ]
    names  = [c.name for c in cities]
    matrix = build_distance_matrix(cities)

    print(f"\n  ۸ شهر | ماتریس {len(cities)}×{len(cities)} آماده\n")

    # ─── تست ACO ───
    print("[تست ۱] الگوریتم کلونی مورچگان (ACO):")
    aco = AntColonyOptimizer(
        matrix   = matrix,
        n_ants   = 15,
        n_iter   = 100,
        alpha    = 1.0,
        beta     = 2.0,
        rho      = 0.1,
        seed     = 42
    )
    aco_result = aco.solve(callback=None)
    print_result(aco_result, names)

    # ─── تست GA ───
    print("\n[تست ۲] الگوریتم ژنتیک با ERX:")
    ga = GeneticAlgorithmSolver(
        matrix         = matrix,
        pop_size       = 60,
        n_generations  = 200,
        mutation_rate  = 0.02,
        tournament_size= 5,
        elite_count    = 2,
        seed           = 42
    )
    ga_result = ga.solve(callback=None)
    print_result(ga_result, names)

    # ─── مقایسه سریع ───
    print("\n[مقایسه نتایج]")
    print(f"  ACO:  {aco_result.best_cost:,.2f} km در {aco_result.elapsed_sec:.3f}s")
    print(f"  GA:   {ga_result.best_cost:,.2f} km در {ga_result.elapsed_sec:.3f}s")

    winner = "ACO" if aco_result.best_cost <= ga_result.best_cost else "GA"
    print(f"  برنده در این تست: {winner}")

    # ─── تست صحت مسیر ───
    print("\n[تست صحت مسیر]")
    for name, result in [("ACO", aco_result), ("GA", ga_result)]:
        path = result.best_path
        valid = (
            len(path) == len(cities) and
            set(path) == set(range(len(cities)))
        )
        cost_check = abs(calculate_path_cost(path, matrix) - result.best_cost) < 0.01
        print(f"  {name}: شهرها کامل و بدون تکرار = {'✅' if valid else '❌'} | هزینه صحیح = {'✅' if cost_check else '❌'}")

    print("\n" + "★" * 60 + "\n")
