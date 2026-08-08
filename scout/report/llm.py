"""LLM scouting notes, strictly grounded in computed metrics.

Privacy: this involves minors — only the jersey number and anonymized numeric
metrics are sent to the API. Names and video never leave the machine.
Fallback: template-based note when no API key / API failure. Ratings unaffected.
"""
from __future__ import annotations

import json
import logging

from scout.config import get_settings

log = logging.getLogger(__name__)

PROMPT = """You are a youth football scout writing a short development note for a coach.
Player: jersey #{jersey}, role {role}, overall rating {overall}/100 (percentile-based within squad).
Sub-scores: {subs}
Metrics: {evidence}
Low sample size: {low_sample}
Limited measurement (camera-relative footage — no distance, speed or pitch-position data): {limited}

Write ~120 words: 2-3 strengths, 1-2 areas to develop, one concrete training suggestion.
Use ONLY the numbers provided. If a metric is missing, do not mention it.
Never comment on fitness, work rate, distance covered, speed or positioning unless
those exact metrics appear above.
If low sample size is true, note the rating is based on limited minutes.
If limited measurement is true, note the rating covers on-ball actions only.
Encouraging, constructive tone — this is a child's development report, not a transfer assessment."""


def template_note(rating: dict) -> str:
    subs = rating.get("sub_scores", {})
    best = max(subs, key=subs.get) if subs else "overall play"
    worst = min(subs, key=subs.get) if subs else "consistency"
    note = (f"Rated {rating['overall']}/100 as a {rating['role']} (squad percentile scoring). "
            f"Strongest area: {best} ({subs.get(best, '?')}). "
            f"Development focus: {worst} ({subs.get(worst, '?')}). ")
    if rating.get("low_sample"):
        note += "Note: limited minutes played — treat this rating as indicative only. "
    if rating.get("limited"):
        note += ("Footage allowed on-ball actions only — no running, speed or positioning "
                 "data was measurable.")
    return note


def scouting_note(rating: dict, jersey: int | str = "?") -> str:
    s = get_settings()
    if not s.anthropic_api_key:
        return template_note(rating)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=s.anthropic_api_key)
        prompt = PROMPT.format(jersey=jersey, role=rating["role"], overall=rating["overall"],
                               subs=json.dumps(rating.get("sub_scores", {})),
                               evidence=json.dumps(rating.get("evidence", {})),
                               low_sample=rating.get("low_sample", False),
                               limited=rating.get("limited", False))
        log.info("LLM prompt for #%s: %s", jersey, prompt)
        msg = client.messages.create(model=s.llm_model, max_tokens=400,
                                     messages=[{"role": "user", "content": prompt}])
        text = msg.content[0].text.strip()
        log.info("LLM response for #%s: %s", jersey, text)
        return text
    except Exception as e:  # noqa: BLE001 — any API failure -> template fallback
        log.warning("LLM note failed (%s); using template", e)
        return template_note(rating)
