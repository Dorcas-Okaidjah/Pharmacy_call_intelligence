"""Generate stand-in calls in the INSURE-Dial file format, for testing the code only.

These are NOT the dataset. They let you check that the whole pipeline runs before
you download the real data. Results computed on them are meaningless.

    python -m tests.make_fixtures --out tests/fixtures
    DATA_DIR override: run_all reads data/, so point it at fixtures with --use-fixtures in tests/smoke_test.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DRUGS = ["Ozempic", "Humira", "Eliquis", "Jardiance", "Dupixent", "Trulicity", "Xarelto", "Entresto"]


def make_call(rng: random.Random, style: str) -> dict:
    alt = style == "real"  # "real" fixtures use different phrasing, to mimic domain shift
    script, ann = [], {}

    def say(role, text):
        script.append({"role": role, "content": text, "turn": len(script) + 1})
        return len(script)

    s = say("user", "Welcome to ABC Pharmacy Solutions. Your call may be recorded. Press 1 for providers.")
    say("assistant", "*presses 1*")
    e = say("user", "Please hold while we connect you.")
    ann["ivr"] = {"turn_range": [s, e]}
    s = say("user", "Thanks for holding, this is Dana, how can I help?" if alt else "Thank you for calling, my name is Dana. How may I help you?")
    e = say("assistant", "Hi Dana, I'm calling to verify pharmacy benefits for a patient.")
    ann["greeting"] = {"turn_range": [s, e]}

    s = say("user", "Sure, can I get the member ID and date of birth?")
    say("assistant", f"The member ID is W{rng.randint(10**7, 10**8)} and DOB is 03/14/1962.")
    found = rng.random() > 0.15
    e = say("user", ("Okay, I've got them pulled up." if alt else "I found the record.") if found
            else ("Hmm, nothing's coming up under that ID." if alt else "I'm unable to locate that member."))
    ann["patient_info"] = {"turn_range": [s, e], "record_found": found}
    ann["drugs"] = []

    if found:
        s = say("assistant", "Is the patient's plan currently active?" if alt else "Can you confirm the coverage status, is the plan active?")
        status = rng.choices(["ACTIVE", "INACTIVE"], [0.85, 0.15])[0]
        e = say("user", {"ACTIVE": "Yep, it's in effect." if alt else "Yes, the plan is active.",
                         "INACTIVE": "That plan was terminated last month." if alt else "The coverage is inactive."}[status])
        ann["coverage_check"] = {"turn_range": [s, e], "plan_status": status}
        if status == "ACTIVE":
            for name in rng.sample(DRUGS, 2):
                d = {}
                s = say("assistant", f"Is {name} on the formulary?" if alt else f"Is {name} covered under the plan?")
                form = rng.choices(["YES", "NO"], [0.75, 0.25])[0]
                e = say("user", (f"{name} is tier 3, preferred." if alt else f"Yes, {name} is covered.") if form == "YES"
                        else (f"{name} is non-formulary." if alt else f"No, {name} is not covered."))
                d["formulary"] = {"turn_range": [s, e], "value": form}
                if form == "YES":
                    s = say("assistant", "Any prior auth or step therapy?" if alt else "Are there any restrictions like prior authorization?")
                    rest = rng.choices(["YES", "NO"], [0.5, 0.5])[0]
                    e = say("user", ("Yes, it requires prior authorization." if not alt else "It needs a PA and has a quantity limit.")
                            if rest == "YES" else ("No restrictions." if not alt else "Nope, doesn't need anything."))
                    d["restrictions"] = {"turn_range": [s, e], "value": rest}
                    if rest == "NO":
                        s = say("assistant", "What is the copay?")
                        e = say("user", f"The copay is ${rng.choice([10, 25, 40, 60])}.")
                        d["copay"] = {"turn_range": [s, e], "value": "stated"}
                ann["drugs"].append(d)
    s = say("assistant", "Could I get your full name and a reference number?")
    e = say("user", f"Dana Reyes, reference {rng.randint(10**5, 10**6)}.")
    ann["agent_interaction"] = {"turn_range": [s, e]}
    say("assistant", "Thank you, have a great day.")
    return {"script": script, "annotation": ann}


def write(out: Path, n_real: int = 20, n_synth: int = 120, seed: int = 7) -> None:
    rng = random.Random(seed)
    for split, n in (("real", n_real), ("synthetic", n_synth)):
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{split}_{i:04d}.json").write_text(json.dumps(make_call(rng, split)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures")
    write(Path(ap.parse_args().out))
