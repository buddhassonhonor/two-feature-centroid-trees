from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, spearmanr
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from obliquetree.utils import load_tree
import joblib
from run_experiments import ROOT,METHODS,atomic_json

def augment_predictions():
    perclass=[];extra=[]
    for p in (ROOT/'exp/runs/main').rglob('result.json'):
        r=json.loads(p.read_text()); a=np.load(p.with_name('predictions.npz'));y,pr,prob=a['y'],a['pred'],a['proba']
        precision,recall,f1,support=precision_recall_fscore_support(y,pr,labels=np.arange(prob.shape[1]),zero_division=0)
        for c in range(prob.shape[1]): perclass.append(dict(dataset=r['dataset'],seed=r['seed'],method=r['method'],label=c,
                    precision=precision[c],recall=recall[c],f1=f1[c],support=int(support[c])))
        apr=float(average_precision_score(y,prob[:,1])) if prob.shape[1]==2 else float(average_precision_score(np.eye(prob.shape[1])[y],prob,average='macro'))
        extra.append(dict(dataset=r['dataset'],seed=r['seed'],method=r['method'],average_precision=apr))
    pd.DataFrame(perclass).to_csv(ROOT/'exp/class_metrics.csv',index=False)
    return pd.DataFrame(extra)

def perturb():
    rows=[];cases=[];cert=[]
    for p in (ROOT/'exp/runs/main').rglob('result.json'):
        r=json.loads(p.read_text())
        if r['method'] not in ['P2','P2-H','CART','Oblique-2','Oblique-Dense']:continue
        path=p.parent;meta=json.loads((ROOT/f'exp/data/{r["dataset"]}.json').read_text());X=pd.read_pickle(ROOT/f'exp/data/{r["dataset"]}_X.pkl')
        a=np.load(path/'predictions.npz');raw=X.iloc[a['test_rows']]
        prep=joblib.load(path/'preprocessor.joblib');xe=np.asarray(prep.transform(raw),float)
        model=load_tree(str(path/'model.json')) if r['method'].startswith('Oblique') else joblib.load(path/'model.joblib')
        nnum=len(meta['numerical']);rng=np.random.default_rng(91400+r['seed']);z=rng.normal(size=(len(xe),nnum))
        clean=model.predict(xe)
        for sigma in [0,.02,.05,.1,.2]:
            xp=xe.copy();xp[:,:nnum]+=sigma*z
            pred=model.predict(xp)
            rows.append(dict(dataset=r['dataset'],seed=r['seed'],method=r['method'],sigma=sigma,n_numeric=nnum,
                             balanced_accuracy=float(balanced_accuracy_score(a['y'],pred)),change_rate=float(np.mean(pred!=clean))))
        if r['method'] in ['P2','P2-H']:
            radii=[]
            for x in xe:
                idx=0;rad=np.inf
                while not model.nodes[idx]['leaf']:
                    node=model.nodes[idx];w=np.asarray(node['w']);z0=x@w-node['b'];den=np.abs(w[:nnum]).sum()
                    if den>1e-12:rad=min(rad,abs(z0)/den)
                    idx=node['left'] if z0<=0 else node['right']
                radii.append(rad)
            finite=np.isfinite(radii)
            cert.append(dict(dataset=r['dataset'],seed=r['seed'],method=r['method'],numeric_dimension=nnum,
                             median_radius=float(np.median(np.array(radii)[finite])) if finite.any() else None,
                             numeric_invariant_fraction=float(np.mean(~finite)),
                             coverage_005=float(np.mean(np.array(radii)>.05)),coverage_01=float(np.mean(np.array(radii)>.1))))
            if r['seed']==0:
                root=model.nodes[0];names=prep.get_feature_names_out()
                terms=[]
                if not root['leaf']:
                    for i,w in enumerate(root['w']):
                        if abs(w)>1e-12:terms.append(dict(feature=names[i],weight=w))
                mistakes=np.flatnonzero(clean!=a['y'])[:3]
                cases.append(dict(dataset=r['dataset'],name=r['name'],method=r['method'],
                                  root_terms=terms,threshold=root.get('b'),
                                  error_rows=a['test_rows'][mistakes].tolist(),true=a['y'][mistakes].tolist(),
                                  predicted=clean[mistakes].tolist(),complexity=model.complexity(xe)))
    pd.DataFrame(rows).to_csv(ROOT/'exp/perturbation.csv',index=False)
    pd.DataFrame(cert).to_csv(ROOT/'exp/certificates.csv',index=False)
    atomic_json(ROOT/'exp/case_studies.json',cases)

def main():
    if '--skip-perturb' not in sys.argv:perturb()
    d=pd.DataFrame([json.loads(p.read_text()) for p in (ROOT/'exp/runs/main').rglob('result.json')])
    assert len(d)==3500 and not d.duplicated(['dataset','seed','method']).any()
    d=d.merge(augment_predictions(),on=['dataset','seed','method'],validate='one_to_one')
    d.to_csv(ROOT/'exp/results.csv',index=False)
    metrics=['accuracy','balanced_accuracy','macro_f1','auc','average_precision','brier','leaves','depth',
             'path_tests','path_burden','fit_seconds','search_seconds','prediction_seconds']
    summary=d.groupby(['dataset','name','method'])[metrics].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(ROOT/'exp/summary.csv',index=False)
    means=d.groupby(['dataset','method'])[metrics].mean().reset_index()
    overall=means.groupby('method')[metrics].mean()
    piv=means.pivot(index='dataset',columns='method',values='balanced_accuracy')
    overall['mean_rank']=piv.rank(axis=1,ascending=False,method='average').mean()
    overall.to_csv(ROOT/'exp/overall.csv')
    rows=[];rng=np.random.default_rng(7182)
    for target in ['P2','P2-H']:
        tests=[]
        for other in METHODS:
            if other==target:continue
            diff=(piv[target]-piv[other]).to_numpy()
            pvalue=1. if np.max(np.abs(diff))<1e-12 else float(wilcoxon(diff,zero_method='wilcox').pvalue)
            boots=rng.choice(diff,size=(10000,len(diff)),replace=True).mean(axis=1)
            row=dict(target=target,other=other,n_datasets=len(diff),mean_difference=float(diff.mean()),
                     median_difference=float(np.median(diff)),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),
                     wins=int(np.sum(diff>1e-12)),ties=int(np.sum(np.abs(diff)<=1e-12)),losses=int(np.sum(diff< -1e-12)),pvalue=pvalue)
            rows.append(row);tests.append(row)
        ordered=sorted(tests,key=lambda r:r['pvalue']);previous=0
        for i,row in enumerate(ordered):
            previous=max(previous,min(1,row['pvalue']*(len(ordered)-i)));row['holm_pvalue']=previous
    pd.DataFrame(rows).to_csv(ROOT/'exp/statistics.csv',index=False)
    f=friedmanchisquare(*[piv[m] for m in METHODS]);atomic_json(ROOT/'exp/friedman.json',dict(statistic=float(f.statistic),pvalue=float(f.pvalue)))
    supplement=pd.DataFrame([json.loads(p.read_text()) for p in (ROOT/'exp/supplement').rglob('result.json')
                             if json.loads(p.read_text()).get('factor')!='replication'])
    supplement.to_csv(ROOT/'exp/supplement.csv',index=False)
    ab=pd.DataFrame([json.loads(p.read_text()) for p in (ROOT/'exp/runs/ablation').rglob('result.json')])
    ab.to_csv(ROOT/'exp/ablation.csv',index=False)
    stability=[]
    for (did,method),part in d[d.method.isin(['P2','P2-H'])].groupby(['dataset','method']):
        supports=[]
        for seed in part.seed:
            tree=json.loads((ROOT/f'exp/runs/main/{did}/{seed}/{method}/tree.json').read_text())
            node=tree['nodes'][0];names=tree['features']
            supports.append(tuple(names[i] for i,w in enumerate(node.get('w',[])) if abs(w)>1e-12))
        agree=[supports[i]==supports[j] for i in range(5) for j in range(i+1,5)]
        stability.append(dict(dataset=did,method=method,root_support_agreement=float(np.mean(agree))))
    pd.DataFrame(stability).to_csv(ROOT/'exp/stability.csv',index=False)
    validations=[]
    for p in (ROOT/'exp/runs/main').rglob('validation.json'):
        parts=p.relative_to(ROOT/'exp/runs/main').parts
        for v in json.loads(p.read_text()):
            if 'depth' in v['params']:
                validations.append(dict(dataset=int(parts[0]),seed=int(parts[1]),method=parts[2],
                                        depth_limit=v['params']['depth'],score=v['score'],leaves=v['leaves']))
    pd.DataFrame(validations).to_csv(ROOT/'exp/validation_frontier.csv',index=False)
    atomic_json(ROOT/'exp/analysis_design.json',dict(bootstrap_repeats=10000,bootstrap_seed=7182,
                 statistical_unit='dataset mean over 5 grouped splits',holm_family='19 comparisons separately for each target',
                 dependence_caveat='Some benchmark datasets share domains; dataset bootstrap is descriptive, not a population guarantee.',
                 perturbation='Numeric standardized columns only; categorical encodings remain fixed.'))
    print(overall.round(4).to_string());print(pd.DataFrame(rows).query("other in ['CART','Oblique-2','Oblique-Dense']").to_string(index=False))

if __name__=='__main__':main()
