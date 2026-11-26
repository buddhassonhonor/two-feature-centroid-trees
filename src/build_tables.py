import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/tex';OUT.mkdir(exist_ok=True)
def esc(x):return str(x).replace('_',r'\_').replace('&',r'\&')
def table(name,header,rows,columns):
    text=r'\begin{tabular}{'+columns+'}\n'+r'\toprule'+'\n'+' & '.join(header)+r' \\'+'\n'+r'\midrule'+'\n'
    text+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
    (OUT/(name+'.tex')).write_text(text,encoding='utf-8')
def main():
    o=pd.read_csv(ROOT/'exp/overall.csv').set_index('method')
    rows=[]
    for m,r in o.iterrows():
        rows.append([esc(m),f'{100*r.balanced_accuracy:.2f}',f'{100*r.accuracy:.2f}',f'{100*r.macro_f1:.2f}',
                     f'{r.mean_rank:.2f}',f'{r.fit_seconds*1000:.1f}', '--' if pd.isna(r.leaves) else f'{r.leaves:.2f}'])
    table('overall',['Method','BA (\%)','Acc. (\%)','F1 (\%)','Rank','Fit (ms)','Leaves'],rows,'lrrrrrr')
    rows=[]
    for m in ['P2','P2-H','CART','Oblique-2','Oblique-Dense','LightGBM','RBF-SVM']:
        r=o.loc[m];rows.append([m,f'{100*r.auc:.2f}',f'{100*r.average_precision:.2f}',f'{r.brier:.3f}',f'{r.search_seconds*1000:.1f}',f'{r.prediction_seconds*1000:.2f}'])
    table('probability',['Method','AUC (\%)','AP (\%)','Brier','Search (ms)','Pred. (ms)'],rows,'lrrrrr')
    s=pd.read_csv(ROOT/'exp/statistics.csv');rows=[]
    for _,r in s[s.other.isin(['CART','Oblique-2','Oblique-Dense','LightGBM'])].iterrows():
        rows.append([r.target,esc(r.other),f'{100*r.mean_difference:+.2f}',f'[{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]',
                    f'{int(r.wins)}/{int(r.ties)}/{int(r.losses)}',f'{r.holm_pvalue:.3f}'])
    table('statistics',['Target','Reference',r'$\Delta$ (pp)','95\% CI','W/T/L',r'$p_{\rm Holm}$'],rows,'llrrrr')
    for target,fname in [('P2','statistics_full_p2'),('P2-H','statistics_full_p2h')]:
        sub=s[s.target==target];frows=[]
        for _,r in sub.iterrows():
            p_raw='<0.001' if r.pvalue<0.001 else f'{r.pvalue:.3f}'
            p_holm='<0.001' if r.holm_pvalue<0.001 else f'{r.holm_pvalue:.3f}'
            frows.append([esc(r.other),f'{100*r.mean_difference:+.2f}',f'[{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]',
                          f'{int(r.wins)}/{int(r.ties)}/{int(r.losses)}',p_raw,p_holm])
        table(fname,['Reference',r'$\Delta$ (pp)','95\% CI','W/T/L',r'$p_{\rm raw}$',r'$p_{\rm Holm}$'],frows,'lrrrrr')
    matched=pd.read_csv(ROOT/'exp/matched_depth.csv')
    mstats=matched.groupby(['depth_limit','method'])[['balanced_accuracy','leaves','path_tests','path_burden']].mean()
    mrows=[]
    for d_lim in [1,2,3,4,6]:
        for m in ['CART','P2','P2-H','Oblique-2']:
            r=mstats.loc[(d_lim,m)]
            mrows.append([str(d_lim),esc(m),f'{100*r.balanced_accuracy:.2f}',f'{r.leaves:.2f}',f'{r.path_tests:.2f}',f'{r.path_burden:.2f}'])
    table('matched_frontier',['Depth','Method','BA (\%)','Leaves','Path tests','Feature burden'],mrows,'rlrrrr')
    d=pd.read_csv(ROOT/'exp/results.csv');means=d.groupby(['dataset','method']).balanced_accuracy.mean().unstack()
    std=d.groupby(['dataset','method']).balanced_accuracy.std().unstack();manifest=pd.read_csv(ROOT/'exp/data/manifest.csv').set_index('id')
    rows=[]
    for did,r in manifest.iterrows():
        short={'mfeat-morphological':'mfeat-morph.','banknote-authentication':'banknote','blood-transfusion-service-center':'blood-transfusion','wall-robot-navigation':'wall-robot','MagicTelescope':'MagicTelescope','balance-scale':'balance-scale'}
        rows.append([esc(short.get(r['name'],r['name']))]+[f'{100*means.loc[did,m]:.1f} $\pm$ {100*std.loc[did,m]:.1f}' for m in ['P2','P2-H','CART','Oblique-2','Oblique-Dense']])
    # Longtable source, split into two manageable supplementary tables.
    for i,subset in enumerate([rows[:18],rows[18:]]):
        table('datasets'+str(i+1),['Dataset','P2','P2-H','CART','O2','OD'],subset,'lrrrrr')
    composition=[]
    for did,r in manifest.iterrows():
        meta=json.loads((ROOT/f'exp/data/{did}.json').read_text());counts=np.array(meta['class_counts'])
        id_link=rf'\href{{https://www.openml.org/d/{did}}}{{{did}}}'
        composition.append([id_link,esc(r['name']),str(r.n),str(r.p),str(len(meta['numerical'])),str(len(meta['categorical'])),str(len(counts)),f'{100*counts.min()/counts.sum():.1f}',str(r.missing)])
    for i,subset in enumerate([composition[:18],composition[18:]]):
        table('composition'+str(i+1),['ID','Dataset','$n$','$p$','Num.','Cat.','$C$','Min. (\%)','Missing'],subset,'rlrrrrrrr')
    ab=pd.read_csv(ROOT/'exp/ablation.csv');ids=ab.dataset.unique();combined=pd.concat([ab,d[(d.dataset.isin(ids))&(d.method.isin(['P2','P2-H','CART','Oblique-2']))]])
    stats=combined.groupby('method')[['balanced_accuracy','leaves','fit_seconds']].mean();rows=[]
    for m,r in stats.iterrows():rows.append([esc(m),f'{100*r.balanced_accuracy:.2f}',f'{r.leaves:.2f}',f'{1000*r.fit_seconds:.1f}'])
    table('ablation',['Method','BA (\%)','Leaves','Fit (ms)'],rows,'lrrr')
    cases=json.loads((ROOT/'exp/case_studies.json').read_text());rows=[]
    for did in [11,37,1462,1510,1497,40983]:
        c=next(c for c in cases if c['dataset']==did and c['method']=='P2')
        terms=' $+$ '.join(f'{v["weight"]:.3f} {esc(v["feature"].replace("num__", "").replace("cat__", ""))}' for v in c['root_terms'])
        rows.append([esc(c['name']),r'\parbox[t]{0.61\textwidth}{'+terms+' $\\leq$ '+f'{c["threshold"]:.3f}'+'}'])
    table('cases',['Dataset','Root rule in transformed coordinates'],rows,r'lp{0.61\textwidth}')
    replication=pd.read_csv(ROOT/'exp/replication.csv');rows=[]
    for g in ['diagonal','axis','imbalanced','circle','xor','dense']:
        cells=[]
        for m in ['P2','P2-H','CART','Oblique-2']:
            vals=replication[(replication.geometry==g)&(replication.method==m)].balanced_accuracy
            cells.append(f'{100*vals.mean():.1f} $\pm$ {100*vals.std():.1f}')
        rows.append([g]+cells)
    table('replication',['Geometry','P2','P2-H','CART','Oblique-2'],rows,'lrrrr')

if __name__=='__main__':main()
