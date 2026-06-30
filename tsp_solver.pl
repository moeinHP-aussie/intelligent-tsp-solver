% =============================================================================
% فایل: prolog_core/tsp_solver.pl
% حلگر دقیق (Exact Solver) مسئله TSP با الگوریتم Held-Karp
%
%
%   تصمیم گرفتیم به‌جای heuristic یا constraint-propagation صرف، از روش
%   "برنامه‌نویسی پویا" (Dynamic Programming) به نام Held-Karp استفاده کنیم.
%   این الگوریتم ۱۰۰٪ دقیق (Exact) است — یعنی همیشه بهینه‌ی مطلق را پیدا
%   می‌کند — اما پیچیدگی زمانی آن را از O(N!) فاکتوریل به O(N² × 2^N)
%   نمایی کاهش می‌دهد. این یعنی هنوز برای N بزرگ منفجر می‌شود، اما خیلی
%   دیرتر از روش brute-force منفجر می‌شود.
%
% ✦ ترفند کلیدی: Mode-Directed Tabling + Answer Subsumption
%   به‌جای محاسبه‌ی مجدد زیرمسئله‌های تکراری (که در بازگشت معمولی Prolog
%   پیش می‌آید)، از تابعیت (tabling) داخلی SWI-Prolog استفاده می‌کنیم.
%   با دستور:
%       :- table held_karp(+, +, min).
%   به Prolog می‌گوییم: «دو آرگومان اول ورودی (+) هستند و آرگومان سوم
%   خروجی‌ای است که باید با عملگر min جمع‌سپاری (subsume) شود — یعنی
%   اگر held_karp/3 با همان (Mask,Last) چند بار صدا زده شد، فقط جوابی
%   با کمترین Cost را در جدول نگه‌دار». این دقیقاً همان نقشی است که در
%   پیاده‌سازی کلاسیک Held-Karp با آرایه‌ی dp[mask][i] انجام می‌شود —
%   اما این‌جا خودِ موتور Prolog آن آرایه‌ی memoization را به‌صورت
%   خودکار مدیریت می‌کند.
%
% ✦ نمایش زیرمجموعه‌ی شهرها (Subset Representation)
%   هر زیرمجموعه از شهرها را با یک عدد صحیح (bitmask) نشان می‌دهیم.
%   مثلاً اگر بیت i روشن باشد، یعنی شهر i در آن زیرمجموعه حضور دارد.
%   این روش بسیار سریع‌تر از لیست است، چون عملیات اشتراک/عضویت با
%   عملگرهای بیتی (>>، /\، \/) در زمان O(1) انجام می‌شود.
% =============================================================================

:- module(tsp_solver, [
    solve_tsp/4,            % نقطه ورود اصلی برای پایتون (pyswip)
    solve_tsp_safe/5,       % نسخه‌ی امن با محدودیت زمانی/اندازه
    set_distance_matrix/1   % بارگذاری ماتریس فاصله از پایتون
]).

% --- بارگذاری کتابخانه‌های مورد نیاز ---
:- use_module(library(lists)).     % برای nth0، length، msort، min_member و غیره
:- use_module(library(apply)).     % برای maplist، foldl


% =============================================================================
% بخش ۱: حافظه‌ی موقت برای نگهداری ماتریس فاصله
% =============================================================================
:- dynamic(dist_fact/3).   % dist_fact(ShahrI, ShahrJ, Fasele)
:- dynamic(num_cities_fact/1).

%% set_distance_matrix(+Matrix) is det.
%
% Converts the distance matrix (received from Python as a list of lists) into
% Prolog facts. This predicate must be called before each solver execution 
% to clear previous data and load the new dataset.
%
% Matrix: A list of lists — Matrix[I][J] = distance from city I to J (in km).
%         Zero-based indexing is used (aligned with core.py).

set_distance_matrix(Matrix) :-
    % پاک‌سازی کامل فکت‌های قبلی — جلوگیری از تداخل بین اجراهای مختلف بنچمارک
    retractall(dist_fact(_, _, _)),
    retractall(num_cities_fact(_)),

    length(Matrix, N),
    assertz(num_cities_fact(N)),

    % پر کردن فکت‌ها با اندیس صفر-پایه (zero-based) دقیقاً مثل core.py
    forall(
        ( nth0(I, Matrix, Row), nth0(J, Row, Dist) ),
        assertz(dist_fact(I, J, Dist))
    ),

    % Delete tabling Cache
    abolish_all_tables.    

% =============================================================================
% بخش ۲: هسته‌ی الگوریتم Held-Karp با Tabling + Answer Subsumption
% =============================================================================

% --- اعلام جدولی بودن (Mode-Directed Tabling) ---
% امضای حالت: held_karp(+Mask, +Last, -Cost)
%   Mask: بیت‌مپ زیرمجموعه‌ی شهرهای بازدید‌شده (ورودی)
%   Last: آخرین شهری که در این زیرمسیر بازدید شده (ورودی)
%   Cost: کمترین هزینه‌ی ممکن برای رسیدن به این حالت (خروجی — min)
%
% عبارت "min" بعد از "as" یعنی: «اگر چند جواب برای یک (Mask,Last) پیدا
% شد، فقط جوابی با کمترین Cost را در جدول نگه‌دار». این دقیقاً معادل
% فرمول بازگشتی کلاسیک Held-Karp است:
%
%   dp[{S}, k] = min over m in S\{k} of ( dp[S\{k}, m] + dist(m, k) )
%
% اما اینجا بدون نوشتن آرایه‌ی صریح — خود tabling این نقش را بازی می‌کند.

:- table held_karp(+, +, min).

%% held_karp(+Mask, +Last, -Cost) is nondet.

%  حالت پایه: وقتی فقط یک شهر در Mask روشن باشد و آن شهر همان Last باشد،
%  یعنی مسیر را تازه از شهر شروع (همیشه شهر ۰) به Last رسانده‌ایم.

held_karp(Mask, Last, Cost) :-

    % حالت پایه — Mask فقط بیت شهر ۰ (مبدا) و بیت Last را دارد
    Mask =:= (1 << 0) \/ (1 << Last),
    Last =:= 0,
    !,
    Cost = 0.

held_karp(Mask, Last, Cost) :-

    % حالت پایه‌ی دوم: مسیر مستقیم از شهر ۰ به Last (بدون واسطه)
    Mask =:= (1 << 0) \/ (1 << Last),
    Last =\= 0,
    !,
    dist_fact(0, Last, Cost).

held_karp(Mask, Last, Cost) :-
    % حالت بازگشتی اصلی:
    % باید Last در Mask روشن باشد، و حداقل یک شهر دیگر (Mid) هم در
    % Mask باشد که از آن به Last آمده باشیم.
    Mask /\ (1 << Last) =:= (1 << Last),   % Last عضو Mask است
    PrevMask is Mask /\ \(1 << Last),       % زیرمجموعه‌ی قبل از رسیدن به Last

    % باید حداقل بیت شهر ۰ در PrevMask باقی بماند (مبدا همیشه حاضر است)
    PrevMask /\ (1 << 0) =:= (1 << 0),

    % Mid باید عضو PrevMask باشد و Last نباشد — یعنی شهر قبلی در مسیر
    % توجه: اندیس شهرها از ۰ تا N-1 است، پس کران بالای between باید N-1 باشد
    num_cities_fact(N),
    MaxIdx is N - 1,
    between(0, MaxIdx, Mid),                % Mid یکی از همه‌ی شهرهای ممکن (۰ تا N-1)
    Mid =\= Last,
    PrevMask /\ (1 << Mid) =:= (1 << Mid),  % Mid باید واقعاً بازدید شده باشد

    % فراخوانی بازگشتی روی زیرمسئله‌ی کوچکتر (همین جاست که tabling
    % جلوی محاسبه‌ی تکراری زیرمسئله‌های مشترک را می‌گیرد)
    held_karp(PrevMask, Mid, SubCost),

    dist_fact(Mid, Last, EdgeCost),
    Cost is SubCost + EdgeCost.

% =============================================================================
% بخش ۳: بازسازی مسیر بهینه (Path Reconstruction)
% =============================================================================
% خودِ held_karp/3 فقط کمترین هزینه را برمی‌گرداند، نه مسیر را.



% ─────────────────────────────────────────────────────────────────
reconstruct_path(Mask, Last, [Last, 0]) :-
    Mask =:= (1 << 0) \/ (1 << Last),
    !.

reconstruct_path(Mask, Last, [Last | RestPath]) :-
    held_karp(Mask, Last, CostHere),
    PrevMask is Mask /\ \(1 << Last),
    num_cities_fact(N),
    MaxIdx is N - 1,

    % پیدا کردن آن Mid که واقعاً روی مسیر بهینه قرار دارد —
    % یعنی همانی که SubCost + EdgeCost == CostHere را برآورده می‌کند.
    between(0, MaxIdx, Mid),
    Mid =\= Last,
    PrevMask /\ (1 << Mid) =:= (1 << Mid),
    held_karp(PrevMask, Mid, SubCost),
    dist_fact(Mid, Last, EdgeCost),
    CostHere =:= SubCost + EdgeCost,
    !,

    reconstruct_path(PrevMask, Mid, RestPath).

% =============================================================================
% بخش ۴: نقطه‌ی ورود اصلی — solve_tsp/4
% =============================================================================

%% solve_tsp(+NumCities, -BestPath, -BestCost, -ElapsedMs) is det.
%
%  این تابع همان چیزی است که از پایتون (pyswip) صدا زده می‌شود.
%%
%  Args:
%    NumCities: تعداد شهرها (N) — باید با ماتریس بارگذاری‌شده هم‌خوان باشد
%
%  Returns:
%    BestPath:   لیست اندیس شهرها به ترتیب بازدید — همیشه با شهر ۰ شروع می‌شود
%    BestCost:   کمترین هزینه‌ی ممکن (مجموع فاصله‌ی دوری کامل)
%    ElapsedMs:  زمان اجرای محض الگوریتم به میلی‌ثانیه (برای بنچمارک فاز ۴)
solve_tsp(NumCities, BestPath, BestCost, ElapsedMs) :-
    get_time(StartTime),

    LastCityMax is NumCities - 1,
    findall(
        TotalCost-LastCity,
        (
            between(1, LastCityMax, LastCity),   % شهرهای ۱ تا N-1 (شهر ۰ همیشه مبدا است)
            held_karp(FullMask, LastCity, PathCost),
            dist_fact(LastCity, 0, ReturnCost),
            TotalCost is PathCost + ReturnCost
        ),
        AllResults
    ),

    (   AllResults == []
    ->  BestPath = [0], BestCost = 0.0
    ;   % پیدا کردن کمینه‌ی مطلق در میان همه‌ی شهرهای پایانی ممکن
        min_member(BestCost-BestLast, AllResults),

        % بازسازی کامل مسیر از انتها به ابتدا، سپس معکوس کردن آن
        reconstruct_path(FullMask, BestLast, ReversedPath),
        reverse(ReversedPath, BestPath)
    ),

    get_time(EndTime),
    ElapsedMs is (EndTime - StartTime) * 1000.0.

% =============================================================================
% بخش ۵: نسخه‌ی امن با آستانه‌ی اندازه (Safety Threshold Guard)
% =============================================================================

max_safe_cities(15).

%% solve_tsp_safe(+NumCities, -BestPath, -BestCost, -ElapsedMs, -Status) is det.
%
%
%  Status یکی از این مقادیر است:
%    'ok'      → حل شد و BestPath/BestCost معتبرند
%    'skipped' → به‌خاطر بزرگ بودن N، اجرا نشد (BestPath=[], BestCost=0.0)
solve_tsp_safe(NumCities, BestPath, BestCost, ElapsedMs, Status) :-
    max_safe_cities(MaxN),
    (   NumCities =< MaxN
    ->  solve_tsp(NumCities, BestPath, BestCost, ElapsedMs),
        Status = ok
    ;   % رد کردن اجرا — دقیقاً طبق سیاست UX پروژه (پیغام informative در UI)
        BestPath = [],
        BestCost = 0.0,
        ElapsedMs = 0.0,
        Status = skipped
    ).

% =============================================================================
% بخش ۶: تست مستقیم این فایل (اجرای دستی با swipl برای دیباگ)
% =============================================================================

%
%   ?- consult('tsp_solver.pl').
%   ?- test_small_example.
%
% این مثال همان نمونه‌ی استاندارد ۴ شهره‌ی داده‌شده در مستند پروژه است:
%   A→B=10, A→C=15, B→C=20, B→D=25, C→D=30
%   انتظار: مسیر A→B→D→C→A با هزینه‌ی کل ۸۰

test_small_example :-
    Matrix = [
        [0,  10, 15, 1000],   % A
        [10, 0,  20, 25],     % B
        [15, 20, 0,  30],     % C
        [1000, 25, 30, 0]     % D
    ],
    set_distance_matrix(Matrix),
    solve_tsp(4, Path, Cost, ElapsedMs),
    format("~n~`─t~60|~n", []),
    format("  مسیر بهینه (اندیس‌ها): ~w~n", [Path]),
    format("  هزینه‌ی کل: ~w~n", [Cost]),
    format("  زمان اجرا: ~3f میلی‌ثانیه~n", [ElapsedMs]),
    format("~`─t~60|~n", []).
