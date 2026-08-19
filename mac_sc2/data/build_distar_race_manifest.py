#!/usr/bin/env python3
"""Route current-patch replay players to explicit per-race DI-star manifests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sc2reader

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay-manifest',type=Path,required=True)
    parser.add_argument('--race',choices=('zerg','terran','protoss'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--winners-only', action='store_true')
    args=parser.parse_args()
    desired=args.race.title()
    source=json.loads(args.replay_manifest.read_text())['rows']
    paths=sorted({Path(row['path']).resolve() for row in source})
    selected=[]
    for path in paths:
        replay=sc2reader.load_replay(str(path),load_level=2)
        for index,player in enumerate(replay.players):
            if str(player.play_race)==desired and (not args.winners_only or str(player.result) == 'Win'):
                selected.append((path,index,int(getattr(player,'mmr',0) or 0),str(player.result)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(f'{path}\t{index}\n' for path,index,_,_ in selected))
    report={'race':desired,'trajectories':len(selected),'games':len({path for path,_,_,_ in selected}),
            'mmr_min':min((mmr for _,_,mmr,_ in selected),default=None),'mmr_max':max((mmr for _,_,mmr,_ in selected),default=None),
            'winner_trajectories':sum(result=='Win' for *_,result in selected),'manifest':str(args.output.resolve())}
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__': main()
