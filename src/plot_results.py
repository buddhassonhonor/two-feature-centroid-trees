from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':18,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16,
                     'legend.fontsize':16,'axes.titlesize':18,'pdf.fonttype':42,'ps.fonttype':42})
COLORS={'P2':'#2171b5','P2-H':'#d95f02','CART':'#636363','Oblique-2':'#238b45','Oblique-Dense':'#756bb1'}

def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=100,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    results=pd.read_csv(ROOT/'exp/results.csv')
    methods=['P2','P2-H','CART','Oblique-2','Oblique-Dense']
    means=results.groupby(['dataset','method']).balanced_accuracy.mean().unstack()
    names=results.groupby('dataset')['name'].first()
    diff=means.P2-means.CART
    order=diff.sort_values().index
    fig,ax=plt.subplots(figsize=(12,13))
    for j,m in enumerate(['P2','P2-H']):
        vals=(means[m]-means.CART).loc[order]*100
        ax.scatter(vals,np.arange(35)+(j-.5)*.22,label=m,color=COLORS[m],s=45)
    ax.axvline(0,color='black',lw=1);ax.set_yticks(np.arange(35),[names[i] for i in order]);ax.tick_params(axis='y',labelsize=16)
    ax.set_xlabel('Balanced accuracy difference vs CART (pp)');ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2)
    save(fig,'dataset_effects')
    supp=pd.read_csv(ROOT/'exp/supplement.csv');base=supp[supp.factor=='base']
    geoms=['diagonal','axis','imbalanced','circle','xor','dense']
    fig,ax=plt.subplots(figsize=(12,6))
    for j,m in enumerate(['P2','P2-H','CART','Oblique-2']):
        part=base[base.method==m].groupby('geometry').balanced_accuracy.agg(['mean','std']).loc[geoms]
        ax.bar(np.arange(6)+(j-1.5)*.18,part['mean'],.18,yerr=part['std'],label=m,color=COLORS[m],capsize=3)
    ax.set_xticks(np.arange(6),geoms);ax.set_ylim(.45,1);ax.set_ylabel('Balanced accuracy');ax.legend(ncol=4,loc='lower left');save(fig,'geometries')
    fig,axs=plt.subplots(2,2,figsize=(14,10))
    for ax,factor,label in zip(axs.flat,['depth','noise','leaf','p'],['Maximum depth','Label-flip probability','Minimum leaf size','Number of features']):
        for m in ['P2','P2-H','CART','Oblique-2']:
            part=supp[(supp.factor==factor)&(supp.method==m)].groupby('value').balanced_accuracy.agg(['mean','std'])
            ax.errorbar(part.index,part['mean'],yerr=part['std'],label=m,color=COLORS[m],marker='o',capsize=3)
        ax.set_xlabel(label);ax.set_ylabel('Balanced accuracy');ax.grid(alpha=.2)
    axs.flat[0].legend(ncol=2);save(fig,'sensitivity')
    pert=pd.read_csv(ROOT/'exp/perturbation.csv')
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    for m in methods:
        p=pert[pert.method==m].groupby('sigma')[['balanced_accuracy','change_rate']].mean()
        axs[0].plot(p.index,p.balanced_accuracy,label=m,color=COLORS[m],marker='o')
        axs[1].plot(p.index,p.change_rate,label=m,color=COLORS[m],marker='o')
    axs[0].set_ylabel('Balanced accuracy');axs[1].set_ylabel('Prediction change fraction')
    for ax in axs:ax.set_xlabel('Numeric Gaussian noise (standard deviations)');ax.grid(alpha=.2)
    axs[0].legend();save(fig,'perturbation')
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    for ax,factor,label in zip(axs,['n','p'],['Number of samples','Number of features']):
        for m in ['P2','P2-H','CART','Oblique-2']:
            part=supp[(supp.factor==factor)&(supp.method==m)].groupby('value').fit_seconds.agg(['mean','std'])
            ax.errorbar(part.index,part['mean'],yerr=part['std'],label=m,color=COLORS[m],marker='o',capsize=3)
        ax.set_yscale('log');ax.set_xlabel(label);ax.set_ylabel('Fit time (seconds)');ax.grid(alpha=.2)
    axs[0].legend();save(fig,'scaling')
    matched=pd.read_csv(ROOT/'exp/matched_depth.csv');stability=pd.read_csv(ROOT/'exp/stability.csv')
    fig,axs=plt.subplots(2,2,figsize=(14,10))
    for m in ['P2','P2-H','CART','Oblique-2']:
        part=matched[matched.method==m].groupby('depth_limit')[['balanced_accuracy','leaves']].mean()
        axs[0,0].plot(part.index,part.balanced_accuracy,label=m,color=COLORS[m],marker='o')
        axs[0,1].plot(part.index,part.leaves,label=m,color=COLORS[m],marker='o')
    axs[0,0].set_ylabel('Balanced accuracy');axs[0,1].set_ylabel('Number of leaves')
    for ax in axs[0]:ax.set_xlabel('Fixed maximum depth');ax.grid(alpha=.2)
    axs[0,0].legend(ncol=2)
    axs[1,0].boxplot([stability[stability.method==m].root_support_agreement for m in ['P2','P2-H']],tick_labels=['P2','P2-H'])
    axs[1,0].set_ylabel('Root-support agreement');axs[1,0].set_ylim(-.05,1.05)
    leaves=results.groupby(['dataset','method']).leaves.mean().unstack()
    axs[1,1].boxplot([leaves[m]-leaves.CART for m in ['P2','P2-H','Oblique-2']],tick_labels=['P2','P2-H','Oblique-2'])
    axs[1,1].axhline(0,color='black',lw=1);axs[1,1].set_ylabel('Mean leaf difference vs CART')
    save(fig,'matched_stability')
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    depths=[1,2,3,4,6]
    offsets={'CART':(5,-10),'P2':(5,-10),'P2-H':(-28,8),'Oblique-2':(5,5)}
    for m in ['P2','P2-H','CART','Oblique-2']:
        part=matched[matched.method==m].groupby('depth_limit')[['balanced_accuracy','leaves','path_burden']].mean().loc[depths]
        ba=part['balanced_accuracy']*100
        axs[0].plot(part['leaves'],ba,label=m,color=COLORS[m],marker='o',lw=2)
        axs[1].plot(part['path_burden'],ba,label=m,color=COLORS[m],marker='o',lw=2)
        for d in depths:
            dx,dy=offsets[m]
            axs[0].annotate(f'd={d}',(part.loc[d,'leaves'],ba.loc[d]),textcoords='offset points',xytext=(dx,dy),fontsize=11,color=COLORS[m])
            axs[1].annotate(f'd={d}',(part.loc[d,'path_burden'],ba.loc[d]),textcoords='offset points',xytext=(dx,dy),fontsize=11,color=COLORS[m])
    axs[0].set_xlabel('Mean number of leaves');axs[0].set_ylabel('Balanced accuracy (%)')
    axs[1].set_xlabel('Mean feature-occurrence burden per path');axs[1].set_ylabel('Balanced accuracy (%)')
    for ax in axs:ax.grid(alpha=.25)
    axs[0].legend(loc='lower right')
    save(fig,'complexity_frontier')

if __name__=='__main__':main()
