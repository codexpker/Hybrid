"""Execute the fixed public batch. Score mode requires all twelve target model seals."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from threadpoolctl import threadpool_limits
from hybrid_tsad.trajectory_experiment import OUT, NORMAL, MIXED, ROLES, source, train, score, target, dump
from hybrid_tsad.trajectory_checks import check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace",type=Path)
    parser.add_argument("phase",choices=("prepare","train","score"))
    parser.add_argument("--domain",choices=ROLES)
    parser.add_argument("--workers",type=int,default=12)
    args=parser.parse_args()
    w=args.workspace
    out=w/OUT
    with threadpool_limits(1):
        if args.phase=="prepare":
            out.mkdir(parents=True,exist_ok=True)
            dump(out/"checks.json",check(target(w,"PSM","FIT")))
            source(w,out/"source")
            return
        if not args.domain:
            parser.error("train and score require --domain")
        arms=tuple(dict.fromkeys(NORMAL+MIXED))
        if args.phase=="score" and not all((out/d/a/"train.json").is_file() for d in ROLES for a in arms):
            raise RuntimeError("All model seals required")
        with ProcessPoolExecutor(args.workers) as pool:
            for arm in arms:
                print(args.phase,args.domain,arm,flush=True)
                if args.phase=="train":
                    train(w,out,args.domain,arm,pool)
                else:
                    roles=(("TEST",) if arm in NORMAL else ())+(("CAL","QUERY") if arm in MIXED else ())
                    for role in roles:
                        score(w,out,args.domain,arm,role,pool)


if __name__=="__main__":
    main()
