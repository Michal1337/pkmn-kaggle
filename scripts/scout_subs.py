"""Submission-resolved top-deck scout -- THE meta scout (replaced name-level scout_meta.py,
which blended a team's two concurrent submissions into one bucket, 2026-07-12).

Kaggle episodes are played by SUBMISSIONS: each team has up to 2 active subs, each with its
own fixed deck.csv and its OWN rating. For the top-N leaderboard teams this emits the full
portfolio: per sub its latest live score, games seen, 60-card deck (pulled from a replay's
step-1 deck action), an archetype tag, and whether the team is all-in on one deck or
running a main + hedge.

  python scripts/scout_subs.py [top_n] [--out notes/sub_portfolios.json]

Anonymous internal API (no login): session GET -> XSRF cookie -> POST /api/i/...
  GetLeaderboard {competitionId} -> rows {teamId, submissionId, rank, displayScore}
  ListEpisodes {submissionId}    -> episodes (agents: submissionId/teamId/updatedScore)
                                    + submissions (dateSubmitted, teamId) + teams (names)
  replay: kaggleusercontent.com/episodes/{id}.json -> steps -> 60-int action per seat
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
import time

import requests

COMP_SLUG = "pokemon-tcg-ai-battle"
COMP_ID = 116727
BASE = "https://www.kaggle.com/api/i/"


def api_session():
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    s.get(f"https://www.kaggle.com/competitions/{COMP_SLUG}", timeout=30)
    s.headers["x-xsrf-token"] = s.cookies.get("XSRF-TOKEN", "")
    s.headers["content-type"] = "application/json"
    return s


def post(s, endpoint, body):
    for attempt in range(6):
        r = s.post(BASE + endpoint, json=body, timeout=60)
        if r.status_code == 429:                      # rate limit: back off and retry
            wait = 20 * (attempt + 1)
            print(f"  [429] backing off {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def card_names():
    """cid -> (name, is_key_attacker, stage_rank) from EN_Card_Data.csv (repo root).
    stage_rank: 0 = not a Pokemon (tool/trainer/energy), 1 basic, 2 stage1, 3 stage2."""
    out = {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "EN_Card_Data.csv")
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        cid = r.get("Card ID") or r.get("﻿Card ID")
        try:
            cid = int(re.search(r"\d+", cid).group())
        except Exception:
            continue
        nm = r["Card Name"]
        st = (r.get("Stage (Pokémon)/Type (Energy and Trainer)") or "").lower()
        if "tool" in st or "pok" not in st:
            stage = 0
        elif "stage 2" in st:
            stage = 3
        elif "stage 1" in st:
            stage = 2
        else:
            stage = 1
        prev = (r.get("Previous stage") or "").strip()
        out[cid] = (nm, nm.endswith(" ex") or nm.startswith("Mega "), stage, prev)
    return out


def deck_tag(deck, names):
    """Archetype tag = the deck's dominant Pokemon LINE(S), not just ex/Mega cards.
    (v1 keyed on ex/Mega only, which mislabeled the whole non-ex Alakazam archetype as
    'fezandipiti_ex' after its 1-of tech, 2026-07-12.) Score each Pokemon by copies,
    evolution stage and ex/Mega-ness; secondary name only if it's a real second engine."""
    score = collections.Counter()
    meta = {}
    for c in deck:
        nm, is_key, stage, prev = names.get(c, (str(c), False, 0, ""))
        if stage == 0:
            continue
        score[nm] += 2                                   # copies dominate
        meta[nm] = (is_key, stage, prev)
    for nm in score:
        is_key, stage, _ = meta[nm]
        score[nm] += stage + (3 if is_key else 0)
    if not score:
        return "no_pokemon"
    ranked = score.most_common()
    prim = ranked[0][0]
    parts = [re.sub(r"\W+", "_", prim.lower()).strip("_")]
    line = {prim}                                         # primary's whole evolution chain
    p = meta[prim][2]
    while p and p not in line:
        line.add(p)
        p = meta[p][2] if p in meta else ""
    for nm, sc in ranked[1:]:
        if nm in line or meta[nm][2] in line:             # same evolution line -> not a co-engine
            line.add(nm)
            continue
        if sc >= max(8, ranked[0][1] - 4):                # genuine co-engine, not a tech
            parts.append(re.sub(r"\W+", "_", nm.lower()).strip("_"))
        break
    return "+".join(parts)


def fetch_replay_deck(episode_id, agent_index, cache_dir):
    """The 60-int deck action for one seat of one episode (cached; replays are ~4MB)."""
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"{episode_id}.json")
    if os.path.exists(p):
        j = json.load(open(p, encoding="utf-8"))
    else:
        r = requests.get(f"https://www.kaggleusercontent.com/episodes/{episode_id}.json",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        r.raise_for_status()
        j = r.json()
        json.dump(j, open(p, "w", encoding="utf-8"))
    for step in j.get("steps", []):
        ag = step[agent_index]
        a = ag.get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
            return a
    return None


def main():
    try:                                              # Windows cp1250 console vs unicode team names
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("top_n", nargs="?", type=int, default=15)
    ap.add_argument("--out", default="notes/sub_portfolios.json")
    ap.add_argument("--cache", default="notes/_replay_cache")
    args = ap.parse_args()

    s = api_session()
    names = card_names()
    lb = post(s, "competitions.LeaderboardService/GetLeaderboard", {"competitionId": COMP_ID})
    rows = lb["publicLeaderboard"][: args.top_n]
    team_name = {t["teamId"]: t["teamName"] for t in lb.get("teams", [])}

    # bootstrap sweep: one ListEpisodes per top team's LB sub; the responses' submissions/agents
    # metadata cover the whole active top band (opponents included), which is how we discover
    # each team's SECOND active sub without a team-level query (the API only filters by sub).
    submeta = {}                    # sid -> {"teamId", "date"}
    latest = {}                     # sid -> (endTime, updatedScore, episodeId, agentIndex)
    games = collections.Counter()   # sid -> episodes seen in the sweep

    def ingest(j):
        for sub in j.get("submissions", []):
            submeta[sub["id"]] = {"teamId": sub["teamId"], "date": sub.get("dateSubmitted", "")}
        for t in j.get("teams", []):
            team_name.setdefault(t["id"], t.get("teamName", "?"))
        for ep in j.get("episodes", []):
            et = ep.get("endTime") or ep.get("createTime") or ""
            for ai, ag in enumerate(ep.get("agents", [])):
                sid = ag.get("submissionId")
                if sid is None:
                    continue
                games[sid] += 1
                if et >= latest.get(sid, ("",))[0]:
                    latest[sid] = (et, ag.get("updatedScore"), ep["id"], ag.get("index", ai))

    for row in rows:
        ingest(post(s, "competitions.EpisodeService/ListEpisodes", {"submissionId": row["submissionId"]}))
        time.sleep(2.0)

    report = []
    for row in rows:
        tid = row["teamId"]
        subs = sorted((sid for sid, m in submeta.items() if m["teamId"] == tid),
                      key=lambda x: submeta[x]["date"], reverse=True)[:2]
        entries = []
        for sid in subs:
            if sid not in latest:   # discovered sub with no episode in the sweep: query it directly
                try:
                    ingest(post(s, "competitions.EpisodeService/ListEpisodes", {"submissionId": sid}))
                    time.sleep(2.0)
                except Exception:
                    pass
            deck = None
            if sid in latest:
                _, score, ep_id, ai = latest[sid]
                try:
                    deck = fetch_replay_deck(ep_id, ai, args.cache)
                except Exception as e:
                    print(f"  [warn] replay {ep_id} failed: {e}", file=sys.stderr)
            entries.append({
                "submissionId": sid,
                "submitted": submeta[sid]["date"],
                "score": latest.get(sid, (None, None))[1],
                "games_seen": games.get(sid, 0),
                "deck": sorted(deck) if deck else None,
                "tag": deck_tag(deck, names) if deck else "?",
            })
        decks = [tuple(e["deck"]) for e in entries if e["deck"]]
        report.append({
            "rank": row["rank"],
            "team": team_name.get(tid, "?"),
            "teamId": tid,
            "lb_score": row.get("displayScore"),
            "subs": entries,
            "same_deck": len(decks) == 2 and decks[0] == decks[1],
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"competition": COMP_SLUG, "top_n": args.top_n, "portfolios": report},
              open(args.out, "w", encoding="utf-8"), indent=1)

    w = max((len(r["team"]) for r in report), default=4)
    print(f"{'rk':>3} {'team':<{w}} {'lb':>7}  portfolio")
    for r in report:
        parts = []
        for e in r["subs"]:
            sc = f"{e['score']:.0f}" if isinstance(e["score"], (int, float)) else "?"
            parts.append(f"[{e['tag']} @{sc} g{e['games_seen']}]")
        flag = "SAME-DECK" if r["same_deck"] else ("main+hedge" if len(r["subs"]) == 2 else "single")
        print(f"{r['rank']:>3} {r['team']:<{w}} {r['lb_score']:>7}  {' '.join(parts)}  {flag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
