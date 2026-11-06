"""Run the full evaluation and write everything to reports/.

Examples:
    python run_all.py                                   # rules baseline + phase classifier
    python run_all.py --llm anthropic                   # + LLM extraction (needs ANTHROPIC_API_KEY)
    python run_all.py --llm anthropic --llm openai      # compare two providers
    python run_all.py --llm anthropic --limit 10        # cheap smoke test on 10 calls
"""
from __future__ import annotations

import argparse
import statistics
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from src import baseline_rules, llm_extract
from src.config import LLM_MODELS, REPORT_DIR
from src.data import gold_outcomes, load_calls, turn_frame
from src.evaluate import long_table, summarize
from src.phase_classifier import grouped_cv_f1, train_and_test


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-split", default="real", help="split to score outcome extraction on")
    ap.add_argument("--llm", action="append", default=[], choices=["anthropic", "openai", "mock"])
    ap.add_argument("--limit", type=int, default=None, help="max calls to score (cost control)")
    args = ap.parse_args()
    REPORT_DIR.mkdir(exist_ok=True)

    real, synth = load_calls("real"), load_calls("synthetic")
    evals = load_calls(args.eval_split, args.limit)
    golds = {c["call_id"]: gold_outcomes(c["annotation"]) for c in evals}
    print(f"loaded {len(real)} real, {len(synth)} synthetic; scoring outcomes on {len(evals)} {args.eval_split} calls")

    # ---------------------------------------------------------------- 1. outcome extraction
    tables = [long_table(golds, {c["call_id"]: baseline_rules.extract(c["script"]) for c in evals}, "rules")]
    usage = []
    for provider in args.llm:
        if not llm_extract.provider_available(provider):
            print(f"skipping {provider}: API key not set")
            continue
        model = LLM_MODELS.get(provider, "mock")
        preds = {}
        for i, c in enumerate(evals, 1):
            rec = llm_extract.extract(c, provider, gold=golds[c["call_id"]])
            preds[c["call_id"]] = rec["prediction"]
            usage.append({"method": model, **{k: rec[k] for k in
                          ("parse_ok", "input_tokens", "output_tokens", "latency_s", "cost_usd")}})
            print(f"  {model}: {i}/{len(evals)}", end="\r")
        print()
        tables.append(long_table(golds, preds, model))

    detail = pd.concat(tables)
    per_field, overall = summarize(detail)
    if usage:
        u = pd.DataFrame(usage)
        agg = u.groupby("method").agg(parse_rate=("parse_ok", "mean"),
                                      median_latency_s=("latency_s", "median"),
                                      mean_input_tokens=("input_tokens", "mean"),
                                      cost_per_1k_calls=("cost_usd", lambda s: s.mean() * 1000 if s.notna().all() else None))
        overall = overall.merge(agg, left_on="method", right_index=True, how="left")
    detail.to_csv(REPORT_DIR / "outcome_predictions.csv", index=False)
    per_field.to_csv(REPORT_DIR / "outcome_accuracy_by_field.csv", index=False)
    overall.to_csv(REPORT_DIR / "outcome_overall.csv", index=False)
    errors = detail[(detail["applicable"]) & (detail["correct"] == False)]  # noqa: E712
    errors.to_csv(REPORT_DIR / "outcome_errors.csv", index=False)

    # ---------------------------------------------------------------- 2. phase classifier
    synth_turns, real_turns = turn_frame(synth), turn_frame(real)
    cv_mean, cv_std = grouped_cv_f1(synth_turns)
    res = train_and_test(synth_turns, real_turns)
    pd.DataFrame(res["report"]).T.to_csv(REPORT_DIR / "phase_report_real.csv")
    res["confusion"].to_csv(REPORT_DIR / "phase_confusion_real.csv")

    # ---------------------------------------------------------------- 3. operational insight
    all_gold = pd.DataFrame([gold_outcomes(c["annotation"]) for c in real + synth])
    drugs = pd.concat([all_gold[["drug1_formulary", "drug1_restrictions"]].set_axis(["formulary", "restrictions"], axis=1),
                       all_gold[["drug2_formulary", "drug2_restrictions"]].set_axis(["formulary", "restrictions"], axis=1)])
    drugs = drugs[drugs["formulary"].notna()]
    ops = {
        "calls": len(all_gold),
        "record_not_found_rate": (all_gold["record_found"] == False).mean(),  # noqa: E712
        "inactive_coverage_rate": (all_gold["coverage_status"] == "INACTIVE").mean(),
        "drugs_checked": len(drugs),
        "not_on_formulary_rate": (drugs["formulary"] == "NO").mean(),
        "covered_drugs_needing_PA_or_restriction": (drugs.loc[drugs["formulary"] == "YES", "restrictions"] == "YES").mean(),
    }

    # ---------------------------------------------------------------- 4. charts
    pivot = per_field.pivot(index="field", columns="method", values="accuracy")
    ax = pivot.plot.barh(figsize=(8, 4.5))
    ax.set_xlabel("Accuracy on applicable fields"); ax.set_xlim(0, 1); ax.set_ylabel("")
    ax.set_title("Outcome extraction accuracy by field")
    ax.legend(title="method", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    plt.tight_layout(); plt.savefig(REPORT_DIR / "outcome_accuracy_by_field.png", dpi=150); plt.close()

    cm = res["confusion"]
    cmn = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.index)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm.iat[i, j]:
                ax.text(j, i, cm.iat[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cmn.iat[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
    ax.set_title("Phase classifier on real calls (trained on synthetic)")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "phase_confusion_real.png", dpi=150); plt.close()

    # ---------------------------------------------------------------- 5. summary
    fmt = lambda x: f"{x:.1%}" if isinstance(x, float) else x  # noqa: E731
    md = [f"# Pharmacy Benefits-Call Intelligence: results ({date.today()})", "",
          f"Data: {len(real)} real + {len(synth)} synthetic INSURE-Dial calls. "
          f"Outcome extraction scored on {len(evals)} `{args.eval_split}` calls.", "",
          "## 1. Outcome extraction", "",
          overall.to_markdown(index=False, floatfmt=".3f"), "",
          "Accuracy is over fields that apply to the call; `hallucination_rate` is the share of "
          "non-applicable fields where the method still produced a value. CIs resample whole calls.", "",
          pivot.to_markdown(floatfmt=".3f"), "",
          "## 2. Turn-level phase classifier (TF-IDF + logistic regression)", "",
          f"- Grouped 5-fold CV within synthetic calls: macro-F1 {cv_mean:.3f} ± {cv_std:.3f}",
          f"- Trained on synthetic, tested on real calls: macro-F1 {res['macro_f1']:.3f}",
          f"- Synthetic-to-real gap: {cv_mean - res['macro_f1']:+.3f}", "",
          "## 3. Operational picture (gold labels, all calls)", ""]
    md += [f"- {k.replace('_', ' ')}: {fmt(v)}" for k, v in ops.items()]
    md += ["", f"Most frequent extraction errors: see `outcome_errors.csv` ({len(errors)} rows)."]
    (REPORT_DIR / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
