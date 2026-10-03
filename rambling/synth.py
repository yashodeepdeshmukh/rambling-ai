"""Synthetic training data with exact labels.

Each family has 4 phrasing variants. Splits are by variant, not by sample:
    train = variants 0, 1     calib = variant 2     test = variant 3
so calibration and test both measure generalization to wordings the model never saw.
scripts/train.py uses "v0".."v2" for cross-phrasing calibration and "trainval" for the final fit.
Every sample is checked by building it with the oracle (and solving / validating);
samples the template can't represent are dropped and counted.
"""

from __future__ import annotations

import random

from .decide import NotRepresentable
from .deciders import OracleDecider
from .nodered import build as nr_build, validate as nr_validate
from .solvers import linear

SPLITS = {"train": (0, 1), "calib": (2,), "test": (3,), "trainval": (0, 1, 2),
          "v0": (0,), "v1": (1,), "v2": (2,)}
NUMWORD = {2: "twice", 3: "three times", 4: "four times", 5: "five times"}


def _cap(s):
    return s[0].upper() + s[1:]


# ================================================================ solver families
# Each returns (text, vars, domain, nonneg, constraints, goal, target|objective)
# Constraints are listed in order of their key number's first mention in the text.

def f_legs(r, v):
    birds = ["chickens", "ducks", "geese", "turkeys"]
    beasts = ["cows", "pigs", "goats", "horses"]
    A, B = r.choice(birds), r.choice(beasts)
    if r.random() < 0.5:
        A, B = B, A
    la, lb = (2 if A in birds else 4), (2 if B in birds else 4)
    a, b = r.randint(3, 60), r.randint(3, 60)
    heads, legs = a + b, la * a + lb * b
    Q = r.choice([A, B])
    text = [
        f"A farm has {A} and {B}. There are {heads} heads and {legs} legs in total. How many {Q} are there?",
        f"On a farm there are {A} and {B} with {heads} heads and {legs} legs altogether. How many {Q} are on the farm?",
        f"Counting the {A} and {B} in a barn gives {heads} heads and {legs} legs. Find the number of {Q}.",
        f"A pen holds some {A} and some {B}: {heads} animals and {legs} legs. How many {Q} does it hold?",
    ][v]
    return text, [A, B], "integers", True, [([1, 1], 0, "=", heads), ([la, lb], 0, "=", legs)], "value", Q


def f_prices(r, v):
    A, B = r.sample(["pens", "pencils", "apples", "pears", "notebooks", "folders", "mugs", "plates", "roses", "tulips"], 2)
    p, q = r.sample(range(2, 21), 2)
    a, b = r.randint(1, 60), r.randint(1, 60)
    n, T = a + b, p * a + q * b
    who = r.choice(["Sam", "Lena", "Ravi", "Mia", "Omar"])
    Q = r.choice([A, B])
    cnt, cost = ([1, 1], 0, "=", n), ([p, q], 0, "=", T)
    text, cons = [
        (f"{_cap(A)} cost ${p} each and {B} cost ${q} each. {who} bought {n} items for ${T}. How many {Q} did {who} buy?", [cnt, cost]),
        (f"A shop sold {n} {A} and {B} for a total of ${T}. Each of the {A} costs ${p} and each of the {B} costs ${q}. How many {Q} were sold?", [cnt, cost]),
        (f"{who} spent ${T} on {n} {A} and {B}; {A} are ${p} each and {B} are ${q} each. How many {Q} did {who} get?", [cost, cnt]),
        (f"Together, {n} {A} and {B} cost ${T}. One of the {A} is ${p} and one of the {B} is ${q}. Find how many {Q} there are.", [cnt, cost]),
    ][v]
    if v == 0:  # prices are mentioned first, but constraints follow their right-hand sides
        pass
    return text, [A, B], "integers", True, cons, "value", Q


def f_coins(r, v):
    (A, a), (B, b) = r.sample([("nickels", 5), ("dimes", 10), ("quarters", 25)], 2)
    x, y = r.randint(2, 50), r.randint(2, 50)
    n, T = x + y, a * x + b * y
    who = r.choice(["Sam", "Lena", "Ravi", "Mia", "Omar"])
    Q = r.choice([A, B])
    cnt, val = ([1, 1], 0, "=", n), ([a, b], 0, "=", T)
    text, cons = [
        (f"A jar has {n} coins, all {A} and {B}. {_cap(A)} are worth {a} cents and {B} are worth {b} cents. The coins total {T} cents. How many {Q} are there?", [cnt, val]),
        (f"{who} has {T} cents in {A} and {B}, {n} coins in all. Each of the {A} is worth {a} cents and each of the {B} {b} cents. How many {Q} does {who} have?", [val, cnt]),
        (f"There are {n} {A} and {B} in a purse worth {T} cents altogether ({A} {a} cents each, {B} {b} cents each). How many {Q} are in the purse?", [cnt, val]),
        (f"Some {A} ({a} cents) and {B} ({b} cents) add up to {T} cents using {n} coins. How many {Q} are used?", [val, cnt]),
    ][v]
    return text, [A, B], "integers", True, cons, "value", Q


def f_sum_diff(r, v):
    S = r.randint(1, 80)
    d = r.randint(1, 40)
    L = S + d
    s = L + S
    Q = r.choice(["larger", "smaller"])
    add, sub = ([1, 1], 0, "=", s), ([1, -1], 0, "=", d)
    text, cons = [
        (f"The sum of a larger number and a smaller number is {s}, and their difference is {d}. What is the {Q} number?", [add, sub]),
        (f"Two numbers add up to {s}. The larger number exceeds the smaller number by {d}. Find the {Q} number.", [add, sub]),
        (f"A larger number and a smaller number have a sum of {s} and a difference of {d}. What is the {Q} one?", [add, sub]),
        (f"If the larger of two numbers is {d} more than the smaller and they total {s}, what is the {Q} number?", [sub, add]),
    ][v]
    return text, ["larger", "smaller"], "integers", False, cons, "value", Q


PEOPLE = ["Ann", "Ben", "Cal", "Dina", "Eli", "Fay", "Gus", "Hana", "Ivan", "Joy", "Kim", "Leo"]
ITEMS = ["marbles", "stickers", "cards", "books", "stamps", "shells"]


def f_times(r, v):
    A, B = r.sample(PEOPLE, 2)
    X = r.choice(ITEMS)
    k = r.randint(2, 5)
    kw = NUMWORD[k] if r.random() < 0.5 else f"{k} times"
    b = r.randint(2, 40)
    a, N = k * b, k * b + b
    Q = r.choice([A, B])
    if v == 2:  # B mentioned first
        return (f"{B} owns some {X}; {A} owns {kw} as many. In total they own {N}. How many {X} does {Q} own?",
                [B.lower(), A.lower()], "integers", True, [([-k, 1], 0, "=", 0), ([1, 1], 0, "=", N)], "value", Q.lower())
    rel, tot = ([1, -k], 0, "=", 0), ([1, 1], 0, "=", N)
    text, cons = [
        (f"{A} has {kw} as many {X} as {B}. Together they have {N} {X}. How many {X} does {Q} have?", [rel, tot]),
        (f"{A} and {B} have {N} {X} between them, and {A} has {kw} as many as {B}. How many {X} does {Q} have?", [tot, rel]),
        None,
        (f"Between {A} and {B} there are {N} {X}. {A} holds {kw} the number {B} holds. How many does {Q} hold?", [tot, rel]),
    ][v]
    return text, [A.lower(), B.lower()], "integers", True, cons, "value", Q.lower()


def f_more(r, v):
    A, B = r.sample(PEOPLE, 2)
    X = r.choice(ITEMS)
    b, d = r.randint(1, 50), r.randint(1, 30)
    a, N = b + d, 2 * b + d
    Q = r.choice([A, B])
    if v == 2:
        return (f"{B} has {d} fewer {X} than {A}, and they have {N} {X} combined. How many {X} does {Q} have?",
                [B.lower(), A.lower()], "integers", True, [([-1, 1], 0, "=", d), ([1, 1], 0, "=", N)], "value", Q.lower())
    rel, tot = ([1, -1], 0, "=", d), ([1, 1], 0, "=", N)
    text, cons = [
        (f"{A} has {d} more {X} than {B}. Together they have {N} {X}. How many {X} does {Q} have?", [rel, tot]),
        (f"{A} and {B} share {N} {X}, and {A} gets {d} more than {B}. How many {X} does {Q} get?", [tot, rel]),
        None,
        (f"Together {A} and {B} collect {N} {X}; {A} collects {d} more than {B}. How many {X} did {Q} collect?", [tot, rel]),
    ][v]
    return text, [A.lower(), B.lower()], "integers", True, cons, "value", Q.lower()


def f_fee(r, v):
    worker = r.choice(["plumber", "electrician", "tutor", "driver", "cleaner"])
    one, many, domain = r.choice([("hour", "hours", "real numbers"), ("day", "days", "integers"),
                                  ("visit", "visits", "integers"), ("mile", "miles", "real numbers")])
    F, rate, h = r.randint(5, 90), r.randint(3, 60), r.randint(1, 40)
    T = F + rate * h
    text = [
        f"A {worker} charges a ${F} fee plus ${rate} per {one}. A job cost ${T}. How many {many} did the job take?",
        f"The {worker} charges ${rate} per {one} plus a flat ${F}. If the bill was ${T}, how many {many} were charged?",
        f"The bill of ${T} included a ${F} base fee and ${rate} for each {one}. How many {many} were billed?",
        f"After a fixed ${F} charge, every {one} costs ${rate}. Total paid: ${T}. How many {many}?",
    ][v]
    return text, [many], domain, True, [([rate], F, "=", T)], "value", many


def _lp_numbers(r):
    while True:
        a1, b1, a2, b2 = (r.randint(1, 6) for _ in range(4))
        C1, C2 = r.randint(20, 120), r.randint(20, 120)
        p, q = r.sample(range(2, 30), 2)
        if len({a1, b1, a2, b2, C1, C2, p, q}) >= 6:
            return a1, b1, a2, b2, C1, C2, p, q


def f_lp_max(r, v):
    A, B = r.sample(["cakes", "pies", "chairs", "tables", "shirts", "jackets", "lamps", "shelves"], 2)
    a1, b1, a2, b2, C1, C2, p, q = _lp_numbers(r)
    text = [
        f"A workshop makes {A} and {B}. Each of the {A} needs {a1} hours of labor and each of the {B} needs {b1}; {C1} labor hours are available. "
        f"Each of the {A} uses {a2} units of material and each of the {B} uses {b2}; there are {C2} units of material. "
        f"Profit is ${p} for each of the {A} and ${q} for each of the {B}. What is the maximum profit?",
        f"Making one of the {A} takes {a1} hours of labor and {a2} units of material; making one of the {B} takes {b1} hours and {b2} units. "
        f"There are {C1} hours and {C2} units available. Each of the {A} earns ${p} and each of the {B} earns ${q}. Find the maximum profit.",
        f"With {C1} labor hours and {C2} units of material, a shop builds {A} ({a1} hours, {a2} units, ${p} profit each) and "
        f"{B} ({b1} hours, {b2} units, ${q} profit each). What is the largest possible profit?",
        f"Profit per item is ${p} for {A} and ${q} for {B}. Labor: {a1} hours per item of {A}, {b1} per item of {B}, at most {C1} hours. "
        f"Material: {a2} units per item of {A}, {b2} per item of {B}, at most {C2} units. Maximize the profit.",
    ][v]
    return text, [A, B], "integers", True, [([a1, b1], 0, "<=", C1), ([a2, b2], 0, "<=", C2)], "max", [p, q]


def f_lp_min(r, v):
    A, B = r.sample(["oats", "beans", "corn", "barley", "peas", "rice"], 2)
    a1, b1, a2, b2, R1, R2, p, q = _lp_numbers(r)
    text = [
        f"A feed mix uses {A} and {B}. Each unit of {A} gives {a1} grams of protein and each unit of {B} gives {b1} grams. "
        f"Each unit of {A} gives {a2} grams of fiber and each unit of {B} gives {b2} grams. "
        f"The mix needs at least {R1} grams of protein and at least {R2} grams of fiber. {_cap(A)} cost ${p} per unit and {B} cost ${q} per unit. What is the minimum cost?",
        f"One unit of {A} has {a1} grams of protein and {a2} grams of fiber; one unit of {B} has {b1} grams of protein and {b2} grams of fiber. "
        f"A diet requires at least {R1} grams of protein and at least {R2} grams of fiber. {_cap(A)} costs ${p} a unit and {B} costs ${q} a unit. Find the minimum cost.",
        f"To get at least {R1} grams of protein and at least {R2} grams of fiber, a farmer blends {A} ({a1} g protein, {a2} g fiber, ${p} per unit) "
        f"and {B} ({b1} g protein, {b2} g fiber, ${q} per unit). What is the cheapest possible cost?",
        f"Costs are ${p} per unit of {A} and ${q} per unit of {B}. Protein: {a1} grams per unit of {A}, {b1} per unit of {B}, needing at least {R1}. "
        f"Fiber: {a2} grams per unit of {A}, {b2} per unit of {B}, needing at least {R2}. Minimize the cost.",
    ][v]
    return text, [A, B], "integers", True, [([a1, b1], 0, ">=", R1), ([a2, b2], 0, ">=", R2)], "min", [p, q]


def f_capacity(r, v):
    X = r.choice(["crates", "boxes", "bags", "barrels", "sacks"])
    c, k = r.randint(5, 90), r.randint(40, 200)
    C = k + c * r.randint(2, 40) + r.randint(0, c - 1)
    text = [
        f"A van can carry at most {C} kg. Each of the {X} weighs {c} kg and the driver weighs {k} kg. What is the largest number of {X} the van can carry?",
        f"An elevator holds up to {C} kg. With a {k} kg operator inside, how many {X} of {c} kg each can it take at most?",
        f"A boat's limit is {C} kg. It already carries {k} kg of gear. Each of the {X} weighs {c} kg. Find the maximum number of {X} it can carry.",
        f"The trailer is rated for {C} kg and its frame counts {k} kg toward that. {_cap(X)} weigh {c} kg each. How many {X} fit at most?",
    ][v]
    return text, [X], "integers", True, [([c], k, "<=", C)], "max", [1]


def f_ages(r, v):
    A, B = r.sample(PEOPLE, 2)
    k = r.randint(3, 5)
    b = r.randint(2, 15)
    a, y = k * b, (k - 2) * b
    Q = r.choice([A, B])
    now, later = ([1, -k], 0, "=", 0), ([1, -2], 0, "=", y)
    text, cons = [
        (f"{A} is {k} times as old as {B}. In {y} years {A} will be twice as old as {B}. How old is {Q} now?", [now, later]),
        (f"Right now {A} is {k} times {B}'s age. After {y} years, {A} will be twice {B}'s age. What is {Q}'s age now?", [now, later]),
        (f"{A}'s age is {k} times {B}'s. {y} years from now, {A}'s age will be double {B}'s. How old is {Q}?", [now, later]),
        (f"In {y} years, {A} will be twice as old as {B}, though today {A} is {k} times as old. How old is {Q} today?", [later, now]),
    ][v]
    return text, [A.lower(), B.lower()], "integers", True, cons, "value", Q.lower()


def f_perimeter(r, v):
    W, d = r.randint(2, 60), r.randint(1, 30)
    L = W + d
    P = 2 * (L + W)
    Q = r.choice(["length", "width"])
    if v == 3:  # width mentioned first
        return (f"For a rectangle with perimeter {P} cm, the width is {d} cm less than the length. What is the {Q}?",
                ["width", "length"], "real numbers", True, [([2, 2], 0, "=", P), ([-1, 1], 0, "=", d)], "value", Q)
    rel, per = ([1, -1], 0, "=", d), ([2, 2], 0, "=", P)
    text, cons = [
        (f"A rectangle's length is {d} meters more than its width, and its perimeter is {P} meters. What is the {Q}?", [rel, per]),
        (f"A garden has a perimeter of {P} meters, and its length exceeds its width by {d} meters. Find the {Q}.", [per, rel]),
        (f"The length of a field is {d} feet greater than the width. If the perimeter is {P} feet, what is the {Q}?", [rel, per]),
    ][v]
    return text, ["length", "width"], "real numbers", True, cons, "value", Q


SOLVER_FAMILIES = [f_legs, f_prices, f_coins, f_sum_diff, f_times, f_more, f_fee,
                   f_lp_max, f_lp_min, f_capacity, f_ages, f_perimeter]


def solver_example(r, family, variant, idx):
    text, names, domain, nonneg, cons, goal, last = family(r, variant)
    ex = {"id": f"{family.__name__[2:]}-v{variant}-{idx}", "family": family.__name__[2:], "variant": variant,
          "text": text, "vars": names, "domain": domain, "nonneg": nonneg, "goal": goal,
          "constraints": [{"coeffs": c, "const": k, "rel": rel, "rhs": b} for c, k, rel, b in cons]}
    if goal == "value":
        ex["target"] = last
    else:
        ex["objective"] = last
    try:
        model = linear.build(text, OracleDecider(linear.gold_answers(ex)))
    except (NotRepresentable, ValueError):
        return None
    result = linear.solve(model)
    if result["status"] != "ok":
        return None
    ex["answer"] = str(result["value"])
    return ex


def solver_split(split: str, per_family_variant: int, seed: int = 0):
    r = random.Random(f"solver/{split}/{seed}")
    out, dropped = [], 0
    for fam in SOLVER_FAMILIES:
        for v in SPLITS[split]:
            for i in range(per_family_variant):
                ex = solver_example(r, fam, v, i)
                if ex is None:
                    dropped += 1
                else:
                    out.append(ex)
    return out, dropped


# ================================================================ Node-RED families

AREAS = ["home", "factory", "office", "garden", "plant", "lab", "farm", "warehouse"]
ROOMS = ["kitchen", "garage", "line1", "zone3", "floor2", "greenhouse", "dock", "attic"]
MEASURES = ["temperature", "humidity", "pressure", "level", "power", "voltage", "speed", "flow"]
HOSTS = ["broker.local", "mqtt.example.com", "iot.example.net", "10.0.0.5", "192.168.1.20", "mqtt.factory.local", "hub.lan"]
URL_HOSTS = ["hooks.example.com", "api.example.com", "ops.example.org", "alerts.example.net", "data.example.io"]
URL_PATHS = ["alert", "notify", "v1/events", "ingest", "door", "motion", "prices", "health", "status"]
PATHS = ["check", "alert", "echo", "ping", "events", "log", "status", "battery", "ingest", "hook", "data", "report"]
FILES = ["requests", "events", "audit", "orders", "hooks", "trace"]
INTERVALS = [("every second", 1), ("every 5 seconds", 5), ("every 10 seconds", 10), ("every 30 seconds", 30),
             ("every minute", 60), ("every 5 minutes", 300), ("every 10 minutes", 600), ("every hour", 3600),
             ("every 2 hours", 7200)]
RATES = [("every 5 seconds", 5), ("every 10 seconds", 10), ("every 30 seconds", 30), ("per minute", 60),
         ("every 2 minutes", 120), ("per hour", 3600)]
OPS = {"greater than": ["goes above", "is above", "exceeds", "is over", "rises above", "is greater than"],
       "less than": ["drops below", "is below", "falls under", "is less than", "goes under"],
       "at least": ["is at least", "reaches"],
       "at most": ["is at most", "does not exceed"],
       "equal to": ["equals", "is exactly"],
       "not equal to": ["is not", "differs from"]}
METHOD_PHRASES = {"GET": ["send a GET request to", "do a GET on"], "POST": ["POST it to", "send a POST to"],
                  "PUT": ["PUT it to", "send a PUT request to"], "DELETE": ["send a DELETE request to"]}


def _topic(r):
    parts = [r.choice(AREAS), r.choice(ROOMS)] + ([r.choice(MEASURES)] if r.random() < 0.5 else [])
    return "/".join(parts)


def _url(r):
    return f"https://{r.choice(URL_HOSTS)}/{r.choice(URL_PATHS)}"


def _broker(r):
    if r.random() < 0.5:
        return "", "localhost"
    h = r.choice(HOSTS)
    return r.choice([f" on broker {h}", f" via {h}", f" using the broker at {h}"]), h


def nr_sample(r, template, v):
    topic, topic2 = _topic(r), _topic(r)
    while topic2 == topic:
        topic2 = _topic(r)
    url, path = _url(r), "/" + r.choice(PATHS)
    file = f"/tmp/rambling-nr/{r.choice(FILES)}.{r.choice(['log', 'txt', 'csv'])}"
    measure = r.choice(MEASURES)
    op = r.choice(list(OPS))
    op_text = r.choice(OPS[op])
    thr = str(r.randint(1, 120))
    method = r.choice(list(METHOD_PHRASES))
    m_text = r.choice(METHOD_PHRASES[method])
    ivl_text, ivl = r.choice(INTERVALS)
    rate_text, rate = r.choice(RATES)
    bc, host = _broker(r)

    if template == "mqtt_threshold_webhook":
        text = [f"When the {measure} on MQTT topic {topic} {op_text} {thr}, {m_text} {url}{bc}.",
                f"Subscribe to {topic}{bc} and if {measure} {op_text} {thr} {m_text} {url}.",
                f"Watch {topic}{bc}: whenever the {measure} {op_text} {thr}, call {url} using {method}.",
                f"Trigger a {method} to {url} if the {measure} reported on {topic} {op_text} {thr}{bc}."][v]
        slots = dict(broker=host, topic=topic, property=f"payload.{measure}", operator=op, threshold=thr, method=method, url=url)
    elif template == "mqtt_monitor":
        text = [f"Show me everything published on {topic}{bc} in the debug panel.",
                f"Monitor the MQTT topic {topic}{bc} and print the messages.",
                f"I want to see the messages arriving on {topic}{bc} in the debug sidebar.",
                f"Log incoming MQTT traffic from {topic}{bc} to the debug view."][v]
        slots = dict(broker=host, topic=topic)
    elif template == "mqtt_bridge":
        text = [f"Bridge messages from {topic} to {topic2}{bc}.",
                f"Copy every message on {topic} over to {topic2}{bc}.",
                f"Republish anything that arrives on {topic} onto {topic2}{bc}.",
                f"Mirror {topic} into {topic2}{bc}."][v]
        slots = dict(broker=host, topic_in=topic, topic_out=topic2)
    elif template == "poll_http_debug":
        text = [f"{_cap(ivl_text)} fetch {url} and show the JSON in debug.",
                f"Poll {url} {ivl_text} so I can see it in the debug sidebar.",
                f"Request {url} {ivl_text} and print the result.",
                f"Check {url} {ivl_text} and display the response in the debug panel."][v]
        slots = dict(interval=f"{ivl} seconds", url=url)
    elif template == "poll_http_to_mqtt":
        text = [f"{_cap(ivl_text)}, download {url} and publish the response to MQTT topic {topic}{bc}.",
                f"Grab {url} {ivl_text} and push it to {topic}{bc}.",
                f"Fetch {url} {ivl_text}, then send the body to {topic}{bc}.",
                f"Mirror the API at {url} to the MQTT topic {topic}{bc}, refreshing {ivl_text}."][v]
        slots = dict(interval=f"{ivl} seconds", url=url, broker=host, topic=topic)
    elif template == "http_echo":
        text = [f"Make a {method} endpoint at {path} that sends back whatever it receives.",
                f"I need a {method} {path} endpoint that returns what it was sent.",
                f"Create an HTTP {method} route {path} that echoes the request.",
                f"Expose {path} for {method} requests and reply with the incoming data."][v]
        slots = dict(method=method, path=path)
    elif template == "http_threshold":
        text = [f"Create an endpoint {path} that answers ALERT when the posted {measure} {op_text} {thr} and OK otherwise.",
                f"Expose {path}: reply ALERT if the {measure} in the request {op_text} {thr}, else OK.",
                f"Build a POST API at {path} returning ALERT whenever {measure} {op_text} {thr}; otherwise return OK.",
                f"When someone posts to {path}, respond with ALERT if their {measure} {op_text} {thr} and with OK if not."][v]
        slots = dict(path=path, property=f"payload.{measure}", operator=op, threshold=thr)
    elif template == "webhook_to_file":
        text = [f"Accept POSTs on {path} and append each body to {file}.",
                f"Save every webhook hitting {path} into the file {file}.",
                f"Write each request posted to {path} at the end of {file}.",
                f"Record incoming posts on {path} in {file}, one per line."][v]
        slots = dict(path=path, file=file)
    elif template == "rate_limit_forward":
        text = [f"Forward {topic} messages to {url} but at most once {rate_text}{bc}.",
                f"Throttle {topic} to one webhook call {rate_text} to {url}{bc}.",
                f"Send messages from {topic} to {url}, limited to one {rate_text}{bc}.",
                f"Pass {topic} through to {url} without exceeding one message {rate_text}{bc}."][v]
        slots = dict(broker=host, topic=topic, interval=f"{rate} seconds", url=url)
    elif template == "timer_publish":
        text = [f"Publish the current time to {topic} {ivl_text}{bc}.",
                f"Send a heartbeat timestamp to {topic} {ivl_text}{bc}.",
                f"{_cap(ivl_text)}, post a timestamp on {topic}{bc}.",
                f"Emit a tick to the MQTT topic {topic} {ivl_text}{bc}."][v]
        slots = dict(interval=f"{ivl} seconds", broker=host, topic=topic)
    else:
        raise KeyError(template)
    return text, slots


def nodered_example(r, template, variant, idx):
    text, slots = nr_sample(r, template, variant)
    ex = {"id": f"{template}-v{variant}-{idx}", "family": template, "variant": variant,
          "text": text, "template": template, "slots": slots}
    try:
        t, got, nodes = nr_build.build(text, OracleDecider(nr_build.gold_answers(ex)))
    except NotRepresentable:
        return None
    if got != slots or nr_validate.validate(nodes):
        return None
    return ex


def nodered_split(split: str, per_family_variant: int, seed: int = 0):
    from .nodered.templates import TEMPLATES
    r = random.Random(f"nodered/{split}/{seed}")
    out, dropped = [], 0
    for t in TEMPLATES:
        for v in SPLITS[split]:
            for i in range(per_family_variant):
                ex = nodered_example(r, t.name, v, i)
                if ex is None:
                    dropped += 1
                else:
                    out.append(ex)
    return out, dropped
