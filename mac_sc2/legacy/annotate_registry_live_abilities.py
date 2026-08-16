#!/usr/bin/env python3
"""Attach verified/ambiguous live ability candidates to a replay registry."""
import argparse, json, re
from pathlib import Path


STOP = {"effect", "ability", "research", "train", "build", "morph", "behavior", "upgrade", "level"}


def words(value):
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value or "")
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    return {x.lower() for x in re.findall(r"[A-Za-z]+|\d+", value) if x.lower() not in STOP}


def target_kind(mode):
    return {1: "none", 2: "point", 3: "unit", 4: "either", 5: "point"}.get(mode, "unknown")


def candidates(name, replay_target, catalog):
    query = words(name)
    scored = []
    for row in catalog:
        fields = words(row["button_name"]) | words(row["link_name"]) | words(row["friendly_name"])
        overlap = len(query & fields)
        if not overlap:
            continue
        # Exact target mode is evidence, but never overrides name evidence.
        target_bonus = 1 if target_kind(row["target"]) in (replay_target, "either") else 0
        score = overlap * 10 - len(query ^ fields) * .15 + target_bonus
        scored.append((score, row))
    return sorted(scored, key=lambda x: (-x[0], x[1]["id"]))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--registry",required=True);p.add_argument("--catalog",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    registry=json.loads(Path(a.registry).read_text()); catalog=json.loads(Path(a.catalog).read_text())
    resolved=ambiguous=unresolved=0; cache={}
    for rows in registry["tasks"].values():
        for row in rows:
            cache_key=(row["ability_name"], row["target_kind"])
            options=cache.get(cache_key)
            if options is None:
                options=candidates(*cache_key, catalog); cache[cache_key]=options
            best=options[0] if options else None
            next_score=options[1][0] if len(options)>1 else float("-inf")
            if best and best[0] >= 9 and best[0] > next_score + .5:
                row["live_4_9_2"]={"status":"resolved","ability_id":best[1]["id"],"target_mode":target_kind(best[1]["target"]),"matched_name":best[1]["friendly_name"]}; resolved+=1
            elif options:
                row["live_4_9_2"]={"status":"ambiguous","candidates":[{"ability_id":v[1]["id"],"name":v[1]["friendly_name"]} for v in options[:5]]}; ambiguous+=1
            else:
                row["live_4_9_2"]={"status":"unresolved"}; unresolved+=1
    registry["live_catalog"]="installed SC2 4.9.2"; registry["live_mapping_counts"]={"resolved":resolved,"ambiguous":ambiguous,"unresolved":unresolved}
    Path(a.output).write_text(json.dumps(registry)); print(json.dumps(registry["live_mapping_counts"]))
if __name__=="__main__":main()
