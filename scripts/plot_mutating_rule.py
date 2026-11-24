"""Plot three-seed mean and SD of joint eight-frame NLL."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    path=Path(__file__).resolve().parents[1]/"results/mutating_rule"
    rows=[json.loads(l) for f in ["results.jsonl","off_grid.jsonl"] for l in (path/f).read_text().splitlines()]
    assert len(rows)==63 and len({(r["seed"],r["p"],r["q"],r["arm"]) for r in rows})==63
    cases=sorted({(r["p"],r["q"]) for r in rows})
    x=np.arange(len(cases))
    fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
    for i,arm in enumerate(("static","dynamic","known_rates")):
        vals=[np.array([r["eight_nll"] for r in rows if (r["p"],r["q"])==c and r["arm"]==arm]) for c in cases]
        assert all(len(v)==3 and np.isfinite(v).all() for v in vals)
        means=[v.mean() for v in vals]
        ax.bar(x+(i-1)*.24,means,.23,yerr=[v.std(ddof=1) for v in vals],label=arm)
        print(arm,[(c,round(float(v.mean()),6),round(float(v.std(ddof=1)),6)) for c,v in zip(cases,vals)])
    ax.set(xticks=x,xticklabels=[f"p={p:g}\nq={q:g}" for p,q in cases],ylabel="Joint 8-frame NLL (nats / pixel)",xlabel="Pixel flip p; rule-bit mutation q",title="Changing CA rules: infer both rates from observed history")
    ax.axhline(np.log(2),ls=":",color="gray",label="fair output noise")
    ax.legend(frameon=False);ax.grid(axis="y",alpha=.2)
    fig.savefig(path/"comparison.png",dpi=180)
if __name__=="__main__":main()
