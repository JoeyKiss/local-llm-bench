"""Benchmark local Ollama models on this PC: speed, VRAM fit and a small auto-graded
German/English quality set.

Usage:
    python bench.py                       # all default models
    python bench.py qwen3.6:35b-a3b ...   # only these models
    python bench.py --think ...           # let thinking models think (slower, usually smarter)
    python bench.py --set extra ...       # task set: base (13, default), extra (20) or all

Writes results/<timestamp>.json and results/<timestamp>.md. Stdlib only.
"""
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

OLLAMA = "http://localhost:11434"
NUM_CTX = 8192
DEFAULT_MODELS = [
    "qwen3.6:35b-a3b",
    "qwen3.8:27b",
    "qwen3.5:9b",
]
ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- ollama api

def api(path, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(OLLAMA + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(model, prompt, think, num_predict=2048, temperature=0.0):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
        "options": {"num_ctx": NUM_CTX, "temperature": temperature,
                    "num_predict": num_predict, "seed": 42},
    }
    try:
        r = api("/api/chat", payload)
    except urllib.error.HTTPError as e:
        # Non-thinking models reject the "think" field; retry without it.
        if e.code == 400 and "think" in e.read().decode(errors="ignore").lower():
            payload.pop("think")
            r = api("/api/chat", payload)
        else:
            raise
    text = r["message"].get("content", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text, r


def unload(model):
    try:
        api("/api/generate", {"model": model, "keep_alive": 0})
    except Exception:
        pass


def gpu_share(model):
    for m in api("/api/ps").get("models", []):
        if m["name"] == model or m["model"] == model:
            return m.get("size", 0), m.get("size_vram", 0)
    return 0, 0


# ---------------------------------------------------------------- graders

def last_answer(text):
    m = re.findall(r"ANTWORT\s*:\s*(.+)|ANSWER\s*:\s*(.+)", text, flags=re.I)
    if not m:
        return None
    a, b = m[-1]
    return (a or b).strip().strip("*").strip()


def number_of(s):
    if s is None:
        return None
    s = s.replace("€", "").replace(" ", "")
    m = re.search(r"-?\d[\d.,]*", s)
    if not m:
        return None
    t = m.group().rstrip(".,")
    if "." in t and "," in t:
        # Whichever separator comes last is the decimal one: 2.251,02 or 2,251.02
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        t = t.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", t):
        t = re.sub(r"[.,]", "", t)  # thousands only: 2,318 / 2.251
    else:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def num_grader(expected, tol=0.01):
    def g(text):
        v = number_of(last_answer(text))
        return v is not None and abs(v - expected) <= tol
    return g


def word_grader(*accepted, reject=()):
    def g(text):
        a = last_answer(text) or text.strip().splitlines()[-1] if text.strip() else ""
        a = a.lower().strip(" .!\"'*")
        return any(x.lower() in a for x in accepted) and not any(x.lower() in a for x in reject)
    return g


def extract_json(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def grade_order(text):
    d = extract_json(text)
    if not isinstance(d, dict):
        return False
    try:
        return (int(d.get("menge")) == 3
                and str(d.get("groesse", "")).replace(" ", "") == "205/55R16"
                and abs(float(d.get("stueckpreis")) - 89.90) < 0.01
                and abs(float(d.get("gesamtpreis")) - 269.70) < 0.01
                and str(d.get("lieferdatum")) == "2026-11-14"
                and str(d.get("plz")) == "73728"
                and str(d.get("stadt", "")).lower() == "esslingen")
    except (TypeError, ValueError):
        return False


def grade_sentences(text):
    t = text.strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    return (len(sentences) == 3 and "," not in t
            and all(s[0] in "Ss" for s in sentences))


def code_grader(tests):
    def g(text):
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
        code = blocks[0] if blocks else text
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.py"
            p.write_text(code + "\n\n" + tests, encoding="utf-8")
            try:
                r = subprocess.run([sys.executable, "-I", str(p)], capture_output=True,
                                   timeout=20, cwd=d)
            except subprocess.TimeoutExpired:
                return False
            return r.returncode == 0
    return g


def needle_prompt():
    filler = ("Die Stadtverwaltung hat in ihrer letzten Sitzung über den Ausbau der Radwege, "
              "die Sanierung der Grundschule und die neuen Öffnungszeiten der Bibliothek beraten. ")
    parts = [filler] * 120
    parts.insert(83, "Wichtiger Hinweis: Der Zugangscode für Tresor B lautet 7-4-1-9. ")
    return ("".join(parts) + "\n\nFrage: Wie lautet der Zugangscode für Tresor B? "
            "Antworte am Ende mit 'ANTWORT: <Code>'.")


ANS = " Antworte am Ende in einer eigenen Zeile mit 'ANTWORT: <Ergebnis>'."
ANS_EN = " End with a final line 'ANSWER: <result>'."

TASKS = [
    ("de_rechnen_wechselgeld", "Ein Laden verkauft Äpfel für 0,45 € pro Stück. Anna kauft 17 Äpfel "
     "und bezahlt mit einem 20-€-Schein. Wie viel Wechselgeld bekommt sie (in €)?" + ANS,
     num_grader(12.35)),
    ("de_rechnen_zug", "Ein Zug fährt um 14:40 Uhr in Stuttgart ab und kommt um 17:05 Uhr in "
     "München an. Die Strecke ist 232 km lang. Wie hoch ist die Durchschnittsgeschwindigkeit in "
     "km/h, gerundet auf eine ganze Zahl?" + ANS, num_grader(96, tol=0.5)),
    ("en_math_sum", "What is the sum of all positive integers less than 100 that are divisible "
     "by 3 or 5?" + ANS_EN, num_grader(2318)),
    ("en_logic_pen", "A notebook and a pen cost €3.40 together. The notebook costs €2.90 more "
     "than the pen. How many cents does the pen cost?" + ANS_EN, num_grader(25)),
    ("de_logik_alter", "Lisa ist älter als Tom. Tom ist älter als Mia. Ben ist jünger als Mia. "
     "Wer ist am zweitjüngsten?" + ANS, word_grader("mia")),
    ("de_buchstaben", "Wie oft kommt der Buchstabe 'e' im Wort 'Erdbeermarmelade' vor "
     "(Groß-/Kleinschreibung ignorieren)?" + ANS, num_grader(5)),
    ("de_grammatik_genitiv", "Setze die richtige Form ein: 'Wegen ___ (das schlechte Wetter) fiel "
     "das Spiel aus.' Gib nur die eingesetzten Wörter an." + ANS,
     word_grader("des schlechten wetters")),
    ("de_wissen_wiesbaden", "Welches Bundesland hat die Landeshauptstadt Wiesbaden?" + ANS,
     word_grader("hessen")),
    ("de_extraktion_json", "Extrahiere aus dieser E-Mail ein JSON-Objekt mit den Schlüsseln "
     "menge (Zahl), groesse (Text), stueckpreis (Zahl), gesamtpreis (Zahl), lieferdatum "
     "(Format JJJJ-MM-TT), plz (Text) und stadt (Text). Gib nur das JSON aus.\n\n"
     "\"Sehr geehrte Damen und Herren, hiermit bestelle ich 3 Winterreifen der Größe 205/55 R16 "
     "zum Stückpreis von 89,90 €. Lieferung bitte bis zum 14.11.2026 an Müller GmbH, "
     "Hauptstraße 12, 73728 Esslingen. Mit freundlichen Grüßen, Petra Müller\"",
     grade_order),
    ("de_anweisung_saetze", "Schreibe genau drei Sätze über den Herbst. Jeder Satz muss mit dem "
     "Buchstaben S beginnen. Verwende kein einziges Komma. Gib nur die drei Sätze aus.",
     grade_sentences),
    ("code_roman", "Write a Python function `roman_to_int(s: str) -> int` that converts a Roman "
     "numeral (I, V, X, L, C, D, M, including subtractive forms like IV and CM) to an integer. "
     "Reply with only one Python code block.",
     code_grader("assert roman_to_int('III') == 3\nassert roman_to_int('LVIII') == 58\n"
                 "assert roman_to_int('MCMXCIV') == 1994\nassert roman_to_int('CDXLIV') == 444\n")),
    ("code_intervals", "Write a Python function `merge_intervals(intervals)` that takes a list of "
     "[start, end] pairs (unsorted, inclusive) and returns the merged, sorted list of intervals "
     "where overlapping or touching intervals ([1,3] and [3,5]) are merged. "
     "Reply with only one Python code block.",
     code_grader("assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
                 "assert merge_intervals([[5,7],[1,3],[3,4]]) == [[1,4],[5,7]]\n"
                 "assert merge_intervals([]) == []\n"
                 "assert merge_intervals([[1,10],[2,3]]) == [[1,10]]\n")),
    ("de_langtext_nadel", needle_prompt(), word_grader("7-4-1-9", "7419")),
]

def grade_termin(text):
    d = extract_json(text)
    if not isinstance(d, dict):
        return False
    norm = lambda s: re.sub(r"[^0-9a-z]", "", str(s).lower())
    arbeiten = d.get("arbeiten")
    return ("yılmaz" in str(d.get("name", "")).lower() or "yilmaz" in str(d.get("name", "")).lower()) \
        and norm(d.get("kennzeichen")) == "esak482" \
        and str(d.get("datum")) == "2026-11-03" \
        and str(d.get("uhrzeit")) in ("08:30", "8:30") \
        and norm(d.get("telefon")) == "0711123456" \
        and isinstance(arbeiten, list) and len(arbeiten) == 2


def grade_fruit_list(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    items = [re.match(r"^(\d)\.\s*(.+)$", l) for l in lines]
    if len(lines) != 5 or not all(items):
        return False
    if [int(m.group(1)) for m in items] != [1, 2, 3, 4, 5]:
        return False
    key = lambda s: s.lower().replace("ä", "a").replace("ö", "o").replace("ü", "u").strip("* ")
    names = [key(m.group(2)) for m in items]
    return names == sorted(names) and len(set(names)) == 5


def grade_b_words(text):
    m = re.search(r"\[.*\]", text, flags=re.S)
    try:
        words = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return False
    return (isinstance(words, list) and len(words) == 4 and len(set(words)) == 4
            and all(isinstance(w, str) and len(w) == 5 and w[0] == "B" and w.isalpha()
                    for w in words))


def multi_needle_prompt():
    filler = [
        "Das Lager in Halle 3 wurde im Frühjahr neu gestrichen und mit LED-Leuchten ausgestattet. ",
        "Die Kantine bietet montags und donnerstags ein vegetarisches Tagesgericht an. ",
        "Für die Gabelstapler gilt auf dem gesamten Gelände eine Höchstgeschwindigkeit von 10 km/h. ",
        "Die Betriebsversammlung findet in diesem Jahr im großen Besprechungsraum statt. ",
        "Neue Mitarbeitende erhalten in der ersten Woche eine Sicherheitsunterweisung. ",
        "Der Parkplatz an der Nordseite ist wegen Bauarbeiten bis Ende des Monats gesperrt. ",
    ]
    parts = [filler[i % len(filler)] for i in range(150)]
    parts.insert(31, "Lieferant A liefert jede Woche 120 Schrauben an das Lager. ")
    parts.insert(117, "Lieferant B liefert jede Woche 85 Schrauben an das Lager. ")
    return ("".join(parts) + "\n\nFrage: Wie viele Schrauben liefern Lieferant A und Lieferant B "
            "zusammen in 3 Wochen?" + ANS)


BUGGY_FIND_FIRST = '''def find_first(nums, target):
    """Return the index of the first occurrence of target in sorted nums, or -1."""
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid
        else:
            hi = mid
    return lo if nums[lo] == target else -1'''

TASKS_EXTRA = [
    ("de_prozent_fahrrad", "Ein Fahrrad kostet 840 €. Der Preis wird erst um 15 % erhöht und danach "
     "um 15 % gesenkt. Was kostet das Fahrrad danach (in €, zwei Nachkommastellen)?" + ANS,
     num_grader(821.10)),
    ("de_zinseszins", "Du legst 2.000 € zu 3 % Zinsen pro Jahr mit Zinseszins an. Wie viel Geld hast "
     "du nach 4 Jahren (in €, auf zwei Nachkommastellen gerundet)?" + ANS, num_grader(2251.02)),
    ("en_pipes", "Pipe A fills a tank in 6 hours, pipe B fills it in 4 hours, and drain C empties a "
     "full tank in 12 hours. With all three open, how many hours does it take to fill the empty "
     "tank?" + ANS_EN, num_grader(3)),
    ("de_wochentag", "Der 1. März 2026 ist ein Sonntag. Welcher Wochentag ist der 24. Dezember 2026?"
     + ANS, word_grader("donnerstag")),
    ("de_runder_tisch", "Wie viele Möglichkeiten gibt es, 8 Personen an einen runden Tisch zu setzen, "
     "wenn Drehungen als gleich gelten und Anna und Ben nicht nebeneinander sitzen dürfen?" + ANS,
     num_grader(3600)),
    ("de_luegner", "Auf einer Insel lügen Lügner immer, Ehrliche sagen immer die Wahrheit. A sagt: "
     "'B ist ein Lügner.' B sagt: 'A und C sind vom gleichen Typ.' A ist ehrlich. Ist C ehrlich oder "
     "ein Lügner?" + ANS, word_grader("lügner", "luegner", reject=("kein",))),
    ("de_buchstaben_r", "Wie oft kommt der Buchstabe 'r' im Wort 'Strohhalmverkäuferin' vor?" + ANS,
     num_grader(3)),
    ("de_rueckwaerts", "Schreibe das Wort 'Donaudampfschiff' rückwärts, komplett in Kleinbuchstaben."
     + ANS, word_grader("ffihcsfpmaduanod")),
    ("de_grammatik_dativ", "Setze das richtige Wort ein: 'Ich gebe ___ (der Hund) einen Knochen.' "
     "Gib nur die eingesetzten Wörter an." + ANS, word_grader("dem", reject=("den ", "der "))),
    ("de_konjunktiv", "Wandle in indirekte Rede mit Konjunktiv I um: Er sagt: 'Ich habe keine Zeit.'"
     + ANS, word_grader("er habe keine zeit", "er keine zeit habe")),
    ("de_rechtschreibung", "Welches dieser Wörter ist falsch geschrieben: Rhythmus, Standart, "
     "Reparatur, Maschine? Gib nur das falsche Wort an." + ANS, word_grader("standart")),
    ("de_wissen_magdeburg", "Wie heißt die Landeshauptstadt von Sachsen-Anhalt?" + ANS,
     word_grader("magdeburg")),
    ("en_knowledge_w", "Which chemical element has the symbol W?" + ANS_EN,
     word_grader("tungsten", "wolfram")),
    ("de_extraktion_termin", "Extrahiere aus dieser Nachricht ein JSON-Objekt mit den Schlüsseln "
     "name (Nachname), kennzeichen, datum (JJJJ-MM-TT), uhrzeit (HH:MM), telefon und arbeiten "
     "(Liste der gewünschten Arbeiten). Gib nur das JSON aus.\n\n\"Hallo, hier ist Herr Yılmaz. Ich "
     "möchte meinen BMW 320d (Kennzeichen ES-AK 482) am Dienstag, den 3. November 2026 um 8:30 Uhr "
     "zur Inspektion bringen. Zusätzlich sollen die Bremsbeläge vorne getauscht werden. Rückruf "
     "bitte unter 0711 123456.\"", grade_termin),
    ("code_isbn10", "Write a Python function `is_valid_isbn10(s: str) -> bool`. Hyphens are ignored. "
     "After removing them the string must have exactly 10 characters: 9 digits followed by a digit "
     "or 'X' (worth 10). It is valid if the sum of (10 - i) * digit_i over positions i = 0..9 is "
     "divisible by 11. Reply with only one Python code block.",
     code_grader("assert is_valid_isbn10('0306406152')\nassert not is_valid_isbn10('0306406153')\n"
                 "assert is_valid_isbn10('123456789X')\nassert is_valid_isbn10('0-306-40615-2')\n"
                 "assert not is_valid_isbn10('X234567890')\nassert not is_valid_isbn10('12345')\n")),
    ("code_spiral", "Write a Python function `spiral_order(matrix)` that returns the elements of a "
     "2D list in clockwise spiral order, starting top-left. Reply with only one Python code block.",
     code_grader("assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\n"
                 "assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == "
                 "[1,2,3,4,8,12,11,10,9,5,6,7]\n"
                 "assert spiral_order([]) == []\nassert spiral_order([[1],[2],[3]]) == [1,2,3]\n")),
    ("code_bugfix", "This function has bugs. Fix it and reply with only the corrected function in "
     "one Python code block.\n\n```python\n" + BUGGY_FIND_FIRST + "\n```",
     code_grader("assert find_first([1,2,2,2,3], 2) == 1\nassert find_first([1,3], 2) == -1\n"
                 "assert find_first([], 5) == -1\nassert find_first([1,2,3], 4) == -1\n"
                 "assert find_first([5,5,5], 5) == 0\n")),
    ("de_anweisung_liste", "Nenne genau fünf Obstsorten als nummerierte Liste (1. bis 5.), "
     "alphabetisch sortiert, ohne weiteren Text.", grade_fruit_list),
    ("de_anweisung_json_woerter", "Antworte ausschließlich mit einem JSON-Array aus genau 4 "
     "verschiedenen deutschen Substantiven, die genau 5 Buchstaben haben und mit 'B' beginnen.",
     grade_b_words),
    ("de_langtext_zwei_fakten", multi_needle_prompt(), num_grader(615)),
]

TASK_SETS = {"base": TASKS, "extra": TASKS_EXTRA, "all": TASKS + TASKS_EXTRA}

SPEED_PROMPT = ("Schreibe einen ausführlichen Aufsatz (etwa 400 Wörter) über die Geschichte der "
                "Eisenbahn in Deutschland.")


# ---------------------------------------------------------------- runner

def bench_model(model, think, task_list):
    print(f"\n=== {model} (think={think})", flush=True)
    unload(model)
    t0 = time.time()
    _, r = chat(model, "Sag nur: OK", think=False, num_predict=8)
    load_s = time.time() - t0
    size, vram = gpu_share(model)
    gpu_pct = round(100 * vram / size) if size else None
    print(f"  load {load_s:.1f}s, {size/1e9:.1f} GB, {gpu_pct}% on GPU", flush=True)

    speeds, prompt_speeds = [], []
    for _ in range(2):
        _, r = chat(model, SPEED_PROMPT, think=False, num_predict=512, temperature=0.7)
        speeds.append(r["eval_count"] / (r["eval_duration"] / 1e9))
    _, r = chat(model, needle_prompt(), think=False, num_predict=32)
    pps = r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)
    gen_tps = sum(speeds) / len(speeds)
    print(f"  gen {gen_tps:.1f} tok/s, prompt {pps:.0f} tok/s", flush=True)

    tasks, passed, task_time = {}, 0, 0.0
    for name, prompt, grader in task_list:
        t = time.time()
        try:
            text, _ = chat(model, prompt, think=think, num_predict=4096 if think else 1536)
            ok = bool(grader(text))
        except Exception as e:  # noqa: BLE001 - one broken task must not kill the run
            text, ok = f"ERROR: {e}", False
        dt = time.time() - t
        task_time += dt
        passed += ok
        tasks[name] = {"pass": ok, "seconds": round(dt, 1), "output": text[-1500:]}
        print(f"  {'PASS' if ok else 'fail'}  {name}  ({dt:.1f}s)", flush=True)

    unload(model)
    return {"model": model, "think": think, "load_s": round(load_s, 1),
            "size_gb": round(size / 1e9, 1), "gpu_pct": gpu_pct,
            "gen_tps": round(gen_tps, 1), "prompt_tps": round(pps),
            "score": passed, "total": len(task_list), "task_seconds": round(task_time, 1),
            "tasks": tasks}


def write_report(results, stamp, tasks):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{stamp}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    lines = [f"# Local LLM bench {stamp}", "",
             f"num_ctx={NUM_CTX}, temperature 0 for quality tasks", "",
             "| Model | Think | Score | Gen tok/s | Prompt tok/s | Size GB | % on GPU | Load s | Quality-set time s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['model']} | | ERROR: {r['error']} | | | | | | |")
            continue
        lines.append(f"| {r['model']} | {r['think']} | {r['score']}/{r['total']} | {r['gen_tps']} | "
                     f"{r['prompt_tps']} | {r['size_gb']} | {r['gpu_pct']} | {r['load_s']} | "
                     f"{r['task_seconds']} |")
    lines += ["", "## Per task", ""]
    ok_results = [r for r in results if "error" not in r]
    names = [t[0] for t in tasks]
    lines.append("| Task | " + " | ".join(r["model"].split("/")[-1] for r in ok_results) + " |")
    lines.append("|---|" + "---|" * len(ok_results))
    for n in names:
        lines.append(f"| {n} | " + " | ".join("✅" if r["tasks"][n]["pass"] else "❌"
                                             for r in ok_results) + " |")
    (out / f"{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    args = sys.argv[1:]
    think = "--think" in args
    set_name = "base"
    if "--set" in args:
        i = args.index("--set")
        set_name = args[i + 1]
        del args[i:i + 2]
    tasks = TASK_SETS[set_name]
    models = [a for a in args if not a.startswith("--")] or DEFAULT_MODELS
    stamp = (datetime.now().strftime("%Y-%m-%d_%H%M") + ("" if set_name == "base" else f"_{set_name}")
             + ("_think" if think else ""))
    results = []
    for m in models:
        try:
            results.append(bench_model(m, think, tasks))
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {m}: {e}", flush=True)
            results.append({"model": m, "error": str(e)})
        write_report(results, stamp, tasks)  # keep partial results if a later model hangs


if __name__ == "__main__":
    main()
