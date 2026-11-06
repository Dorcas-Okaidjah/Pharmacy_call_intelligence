"""Keyword-rule baseline for outcome extraction.

This is deliberately simple: it is the bar a stakeholder could build in an afternoon.
The LLM has to beat it by enough to justify its cost and complexity.
"""
from __future__ import annotations

import re

NOT_FOUND = re.compile(r"(unable to (find|locate)|can(no|')t (find|locate)|no record|not (finding|locating|able to find)|doesn'?t (show|come up))", re.I)
FOUND = re.compile(r"(found (the|him|her|it|that)|i (have|see|located|pulled up) (the|him|her|it|that|their)|record (is|was) (found|located)|got (the|it|him|her))", re.I)
INACTIVE = re.compile(r"\b(inactive|terminated|not active|termed|expired|cancel+ed)\b", re.I)
ACTIVE = re.compile(r"\b(active|in effect|currently covered)\b", re.I)
NOT_COVERED = re.compile(r"(not (covered|on (the )?formulary)|non-?formulary|isn'?t covered|excluded)", re.I)
COVERED = re.compile(r"(\bis covered|on (the )?formulary|\bcovered\b|preferred|tier \d)", re.I)
NO_RESTRICT = re.compile(r"(no (restrictions?|prior auth\w*|pa\b|step therapy|quantity limits?)|doesn'?t (need|require)|not required|none)", re.I)
RESTRICT = re.compile(r"(prior auth\w*|\bpa\b|step therapy|quantity limit|requires? (a |an )?(authorization|approval))", re.I)

COVERAGE_Q = re.compile(r"(coverage|plan|policy|benefits?).{0,40}(active|status)|(active|status).{0,40}(coverage|plan|policy)", re.I)
FORMULARY_Q = re.compile(r"(covered|formulary)", re.I)
RESTRICT_Q = re.compile(r"(prior auth|restriction|step therapy|quantity limit|\bpa\b)", re.I)


def _user_answer_after(script, start, question_re, answer_res):
    """Find the first assistant turn at/after `start` matching question_re, then return
    the label of the first following user turn that matches one of answer_res."""
    for i in range(start, len(script)):
        t = script[i]
        if t.get("role") == "assistant" and question_re.search(t.get("content", "")):
            for j in range(i + 1, min(i + 5, len(script))):
                u = script[j]
                if u.get("role") != "user":
                    continue
                for label, rx in answer_res:
                    if rx.search(u.get("content", "")):
                        return label, j
            return None, i
    return None, start


def extract(script: list[dict]) -> dict:
    out = {k: None for k in ("record_found", "coverage_status", "drug1_formulary",
                             "drug1_restrictions", "drug2_formulary", "drug2_restrictions")}

    user_text = " ".join(t.get("content", "") for t in script if t.get("role") == "user")
    if NOT_FOUND.search(user_text):
        out["record_found"] = False
    elif FOUND.search(user_text) or COVERAGE_Q.search(" ".join(t["content"] for t in script)):
        out["record_found"] = True
    if out["record_found"] is False:
        return out

    status, cursor = _user_answer_after(script, 0, COVERAGE_Q,
                                        [("INACTIVE", INACTIVE), ("ACTIVE", ACTIVE)])
    out["coverage_status"] = status or "NOT_ANSWERED"
    if status != "ACTIVE":
        return out

    for d in (1, 2):
        form, cursor = _user_answer_after(script, cursor + 1, FORMULARY_Q,
                                          [("NO", NOT_COVERED), ("YES", COVERED)])
        out[f"drug{d}_formulary"] = form
        if form == "YES":
            rest, cursor = _user_answer_after(script, cursor + 1, RESTRICT_Q,
                                              [("NO", NO_RESTRICT), ("YES", RESTRICT)])
            out[f"drug{d}_restrictions"] = rest
    return out
