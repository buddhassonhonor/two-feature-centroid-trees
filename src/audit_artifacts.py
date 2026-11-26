"""Independent artifact checks; verifies saved predictions and split integrity."""
from pathlib import Path
import json,re,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from run_experiments import ROOT,metrics,atomic_json

def main():
    counts={};max_error=0.;group_checks=0;probability_error=0.
    for suite,expected in [('main',3500),('ablation',120)]:
        files=list((ROOT/f'exp/runs/{suite}').rglob('result.json'));assert len(files)==expected
        counts[suite]=len(files)
        assert not list((ROOT/f'exp/runs/{suite}').rglob('error.txt'))
        for p in files:
            r=json.loads(p.read_text());pred=np.load(p.with_name('predictions.npz'))
            assert np.isfinite(pred['proba']).all() and (pred['proba']>=-1e-10).all()
            probability_error=max(probability_error,float(np.max(abs(pred['proba'].sum(axis=1)-1))))
            dtype_tolerance=max(1e-12,2*np.finfo(pred['proba'].dtype).eps*pred['proba'].shape[1])
            assert np.max(abs(pred['proba'].sum(axis=1)-1))<=dtype_tolerance,(p,dtype_tolerance)
            recalculated=metrics(pred['y'],pred['pred'],pred['proba'])
            for m,value in recalculated.items():max_error=max(max_error,abs(value-r[m]))
            y=np.load(ROOT/f'exp/data/{r["dataset"]}_y.npz')['y']
            assert np.array_equal(y[pred['test_rows']],pred['y'])
    assert max_error<1e-12
    for p in (ROOT/'exp/splits').glob('*.npz'):
        did=int(p.stem.split('_')[0]);X=pd.read_pickle(ROOT/f'exp/data/{did}_X.pkl')
        groups=pd.util.hash_pandas_object(X,index=False).to_numpy();split=np.load(p)
        sets=[set(groups[split[k]]) for k in ('train','val','test')]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        rows=np.concatenate([split[k] for k in ('train','val','test')]);assert len(rows)==len(X) and len(np.unique(rows))==len(X)
        group_checks+=1
    matched=pd.read_csv(ROOT/'exp/matched_depth.csv');assert len(matched)==3500
    rep=pd.read_csv(ROOT/'exp/replication.csv');assert len(rep)==210 and set(rep.seed)==set(range(10,15))
    primary=pd.read_csv(ROOT/'exp/supplement.csv');assert len(primary)==630 and set(primary.seed)==set(range(5))
    tex=(ROOT/'main.tex').read_text(encoding='utf-8');bib=(ROOT/'ref.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\{([^,]+),',bib));cited=set()
    for cluster in re.findall(r'\\cite\w*\{([^}]+)\}',tex):cited.update(cluster.split(','))
    assert keys==cited,(keys-cited,cited-keys)
    verified={r['key']:r for r in json.loads((ROOT/'research/reference_audit.json').read_text()) if r['status']=='verified'}
    assert keys<=set(verified)
    recent=[key for key in keys if int(verified[key]['year'])>=2025];assert len(recent)>=5
    for entry in re.split(r'@\w+\{',bib)[1:]:assert not ('doi =' in entry and 'url =' in entry)
    status=re.findall(r'(?i)\bTODO\b|\bTBD\b|\bXXX\b|final pipeline|zero failures|all tasks succeeded|dataset-seed runs|final result files',tex)
    assert not status,status
    log=(ROOT/'main.log').read_text(encoding='utf-8',errors='replace')
    assert 'undefined' not in log.lower() and 'Overfull' not in log
    plot_inputs=['results.csv','statistics.csv','supplement.csv','perturbation.csv','matched_depth.csv','stability.csv']
    provenance=dict(inputs={f'exp/{n}':hashlib.sha256((ROOT/'exp'/n).read_bytes()).hexdigest() for n in plot_inputs},
                    script_sha256=hashlib.sha256((ROOT/'src/plot_results.py').read_bytes()).hexdigest(),
                    figures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'figures').glob('*.pdf')})
    atomic_json(ROOT/'exp/figure_provenance.json',provenance)
    output=dict(status='verified',run_counts=counts,matched_depth=3500,controlled=630,independent_replication=210,
                group_disjoint_splits=group_checks,max_metric_discrepancy=max_error,max_probability_sum_error=probability_error,
                verified_cited_references=len(keys),recent_cited_references=len(recent),
                manuscript_sha256=hashlib.sha256((ROOT/'main.tex').read_bytes()).hexdigest(),
                pdf_sha256=hashlib.sha256((ROOT/'main.pdf').read_bytes()).hexdigest())
    atomic_json(ROOT/'exp/artifact_audit.json',output);print(json.dumps(output,indent=2))

if __name__=='__main__':main()
