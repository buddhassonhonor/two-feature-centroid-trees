"""Official header with bounded subprocess execution; not mixed into primary search."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,subprocess,time,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from run_experiments import ROOT,preprocessor,metrics,atomic_json
from threadpoolctl import threadpool_limits

def files(path,xt,y,xe):
    with (path/'train.txt').open('w') as f:
        f.write(f'{len(xt)} {xt.shape[1]} {len(np.unique(y))}\n');np.savetxt(f,np.c_[xt,y],fmt='%.17g')
    with (path/'eval.txt').open('w') as f:
        f.write(f'{len(xe)} {xe.shape[1]}\n');np.savetxt(f,xe,fmt='%.17g')

def fit_predict(path,xt,y,xe,leaf,seed):
    files(path,xt,y,xe)
    command=[str(ROOT/'exp/lht/lht_adapter.exe'),str(path/'train.txt'),str(path/'eval.txt'),
             str(path/'scores.csv'),str(leaf),str(seed)]
    start=time.perf_counter()
    proc=subprocess.run(command,capture_output=True,text=True,timeout=30)
    seconds=time.perf_counter()-start
    (path/'stdout.log').write_text(proc.stdout,encoding='utf-8');(path/'stderr.log').write_text(proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError(f'Official header returned {proc.returncode}: '+proc.stderr)
    score=np.loadtxt(path/'scores.csv',delimiter=',',ndmin=2)
    if not np.isfinite(score).all():raise RuntimeError('Nonfinite output scores')
    pred=score.argmax(axis=1)
    # Normalize the nonnegative weighted one-vs-rest scores for descriptive probability metrics.
    totals=score.sum(axis=1,keepdims=True);prob=np.divide(score,totals,out=np.full_like(score,1/score.shape[1]),where=totals>1e-12)
    return pred,prob,seconds

def evaluate(did,seed):
    threadpool_limits(1)
    output=ROOT/f'exp/lht/results/{did}/{seed}';output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists() or (output/'failure.json').exists():return 'cached'
    try:
        meta=json.loads((ROOT/f'exp/data/{did}.json').read_text());X=pd.read_pickle(ROOT/f'exp/data/{did}_X.pkl');y=np.load(ROOT/f'exp/data/{did}_y.npz')['y']
        split=np.load(ROOT/f'exp/splits/{did}_{seed}.npz');tr,va,te=split['train'],split['val'],split['test']
        prep=preprocessor(meta);xt=np.asarray(prep.fit_transform(X.iloc[tr]),float);xv=np.asarray(prep.transform(X.iloc[va]),float)
        search=[]
        for leaf in [8,16,32,64]:
            path=output/f'validation_{leaf}';path.mkdir(exist_ok=True)
            pred,_,seconds=fit_predict(path,xt,y[tr],xv,leaf,seed)
            search.append(dict(min_split=leaf,score=float(balanced_accuracy_score(y[va],pred)),seconds=seconds))
        best=max(search,key=lambda r:(r['score'],r['min_split']))
        full=np.r_[tr,va];prep=preprocessor(meta);xf=np.asarray(prep.fit_transform(X.iloc[full]),float);xe=np.asarray(prep.transform(X.iloc[te]),float)
        path=output/'test';path.mkdir(exist_ok=True);pred,prob,seconds=fit_predict(path,xf,y[full],xe,best['min_split'],seed)
        np.savez_compressed(output/'predictions.npz',y=y[te],pred=pred,proba=prob,test_rows=te)
        atomic_json(output/'result.json',dict(dataset=did,name=meta['name'],seed=seed,method='LHT-official-adapter',
                        selected=best,search=search,total_subprocess_seconds=seconds,**metrics(y[te],pred,prob)))
        return 'ok'
    except Exception as e:
        atomic_json(output/'failure.json',dict(dataset=did,seed=seed,error=repr(e)));return 'failed'

def main():
    ids=pd.read_csv(ROOT/'exp/data/manifest.csv').id.tolist()
    with ProcessPoolExecutor(max_workers=2) as ex:
        fs={ex.submit(evaluate,d,s):(d,s) for d in ids for s in range(5)}
        for i,f in enumerate(as_completed(fs),1):print(i,len(fs),fs[f],f.result(),flush=True)
    rows=[json.loads(p.read_text()) for p in (ROOT/'exp/lht/results').rglob('result.json')]
    pd.DataFrame(rows).to_csv(ROOT/'exp/lht/results.csv',index=False)
    failures=[json.loads(p.read_text()) for p in (ROOT/'exp/lht/results').rglob('failure.json')]
    atomic_json(ROOT/'exp/lht/reproducibility.json',dict(header_sha256=hashlib.sha256((ROOT/'research/lht_source/LHT.h').read_bytes()).hexdigest(),
                 adapter_sha256=hashlib.sha256((ROOT/'src/lht_adapter.cpp').read_bytes()).hexdigest(),
                 results=len(rows),failures=failures,timeout_seconds=30,
                 adaptation='One official one-vs-rest tree per class; beta=1, forest_rate=1; validation-select min split 8,16,32,64; weighted scores as official evaluation; input/split adapter only.',
                 concern='Official growTree truncates numerical features to long for leaf moments, does not reset moments per coordinate, and uses non-mean root/child centroids. Results cannot establish faithful mathematical LHT performance.'))

if __name__=='__main__':main()
