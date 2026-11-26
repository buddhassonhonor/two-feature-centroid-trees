"""Controlled geometries, parameter sensitivity, empirical perturbation and certificates."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import argparse, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from obliquetree import Classifier
from models import ProjectionTree
from run_experiments import ROOT, atomic_json, metrics
from threadpoolctl import threadpool_limits

GEOMS=['diagonal','axis','xor','circle','dense','imbalanced']

def synthetic(geometry, seed, n=2000, p=8, noise=.05):
    rng=np.random.default_rng(7100+seed)
    X=rng.normal(size=(n,p))
    if geometry=='diagonal': y=(1.15*X[:,0]+1.05*X[:,1]>.0).astype(int)
    elif geometry=='axis': y=(X[:,0]>0).astype(int)
    elif geometry=='xor': y=(X[:,0]*X[:,1]>0).astype(int)
    elif geometry=='circle': y=(X[:,0]**2+X[:,1]**2>1.4).astype(int)
    elif geometry=='dense': y=(X.sum(axis=1)>0).astype(int)
    elif geometry=='imbalanced': y=(X[:,0]+X[:,1]>1.5).astype(int)
    else: raise ValueError(geometry)
    flip=rng.random(n)<noise; y[flip]=1-y[flip]
    return X,y

def run_task(geometry,seed,factor,value,method):
    threadpool_limits(1)
    output=ROOT/f'exp/supplement/{geometry}/{seed}/{factor}_{value}/{method}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists(): return 'cached'
    n,p,noise,depth,leaf=2000,8,.05,3,8
    if factor=='n': n=int(value)
    elif factor=='p': p=int(value)
    elif factor=='noise': noise=float(value)
    elif factor=='depth': depth=int(value)
    elif factor=='leaf': leaf=int(value)
    X,y=synthetic(geometry,seed,n,p,noise)
    tr,te=train_test_split(np.arange(n),test_size=.25,stratify=y,random_state=120+seed)
    prep=StandardScaler().fit(X[tr]); xt,xe=prep.transform(X[tr]),prep.transform(X[te])
    if method=='CART': model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=seed)
    elif method=='Oblique-2': model=Classifier(max_depth=depth,min_samples_leaf=leaf,n_pair=2,random_state=seed)
    else:
        mode={'P2':'p2','P2-H':'hybrid','P2-Tuned':'tuned','P-Dense':'dense','P2-Random':'random_pair'}[method]
        model=ProjectionTree(depth,leaf,mode,seed)
    started=time.perf_counter(); model.fit(xt,y[tr]); runtime=time.perf_counter()-started
    pred=model.predict(xe); prob=model.predict_proba(xe)
    np.savez_compressed(output/'data.npz',X=X,y=y,train=tr,test=te,pred=pred,proba=prob)
    comp=model.complexity(xe) if isinstance(model,ProjectionTree) else {}
    row=dict(geometry=geometry,seed=seed,factor=factor,value=value,method=method,n=n,p=p,
             noise=noise,depth_limit=depth,leaf_limit=leaf,fit_seconds=runtime,**metrics(y[te],pred,prob),**comp)
    atomic_json(output/'result.json',row)
    return 'ok'

def verify():
    rng=np.random.default_rng(781)
    X=rng.normal(size=(10000,2)); a=np.array([-.7,-.1]); b=np.array([.9,.4])
    w=b-a; threshold=(b@b-a@a)/2
    direct=np.sum((X-a)**2,axis=1)<=np.sum((X-b)**2,axis=1)
    linear=X@w<=threshold
    assert np.array_equal(direct,linear)
    X,y=synthetic('diagonal',777,n=500,p=8)
    model=ProjectionTree(4,8,'hybrid',777).fit(X,y)
    checked=0; flips_outside=0
    for row in X[:200]:
        idx=0; radius=np.inf; limiting=None
        while not model.nodes[idx]['leaf']:
            node=model.nodes[idx]; ww=np.array(node['w']); z=row@ww-node['b']
            r=abs(z)/np.sum(abs(ww))
            if r<radius: radius=r; limiting=(ww,z)
            idx=node['left'] if z<=0 else node['right']
        if not np.isfinite(radius) or radius<1e-10: continue
        # Verify path (stronger than only verifying the final label).
        def path(x):
            i=0; result=[]
            while not model.nodes[i]['leaf']:
                nd=model.nodes[i]; route=x@np.array(nd['w'])<=nd['b']; result.append((i,bool(route)))
                i=nd['left'] if route else nd['right']
            return result
        perturb=rng.uniform(-1,1,size=(100,8))*(.99*radius)
        assert all(path(row+delta)==path(row) for delta in perturb)
        ww,z=limiting; delta=-np.sign(z)*np.sign(ww)*(1.01*radius)
        flips_outside+=path(row+delta)!=path(row); checked+=100
    atomic_json(ROOT/'exp/verification.json',dict(distance_linear_equal=True,samples=10000,
                inner_path_checks=checked,outside_path_flips=flips_outside,
                caveat='Open ball; path invariance only, not necessity for unchanged label.'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    tasks=[]
    for geom in GEOMS:
        for seed in range(5):
            for method in ['P2','P2-H','CART','Oblique-2','P2-Tuned','P-Dense','P2-Random']:
                tasks.append((geom,seed,'base',0,method))
    for factor,values in [('n',[500,1000,2000,4000]),('p',[2,8,16,32]),('noise',[0,.05,.1,.2]),
                          ('depth',[1,2,3,4,6]),('leaf',[2,4,8,16])]:
        for value in values:
            for seed in range(5):
                for method in ['P2','P2-H','CART','Oblique-2']:
                    tasks.append(('diagonal',seed,factor,value,method))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(run_task,*t):t for t in tasks}
        for i,f in enumerate(as_completed(fs),1): print(i,len(tasks),fs[f],f.result(),flush=True)
    verify()

if __name__=='__main__':main()
