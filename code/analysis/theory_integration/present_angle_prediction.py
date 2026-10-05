"""Angle-first main figure: endpoint SVDs versus parameter-free predictions."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/angle_prediction'
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(OUT/'cache'))
(OUT/'cache').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'analysis'))
import numpy as np
from scipy.linalg import svd
from validate_index_model import quantiles,susceptibility,angles

DESIGN=dict(n=512,rank=8,draws=64,gamma=1e-4,seed=2026091802,bins=16,
            base='diagonal matrix of deterministic quarter-circle midpoint quantiles',
            update='independent Gaussian BA, normalized to base Frobenius norm',
            selection='all indices, all draws, contiguous equal-sized display bins')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    # Write the fixed design before drawing any update or computing endpoints.
    path=OUT/'design.json'
    if path.exists():assert json.loads(path.read_text())==DESIGN
    path.write_text(json.dumps(DESIGN,indent=2)+'\n')
    n=DESIGN['n'];s=quantiles(n);base=np.diag(s);energy=s@s
    chi,_,_=susceptibility(s)
    rng=np.random.default_rng(DESIGN['seed'])
    theta=np.empty((DESIGN['draws'],2,n))
    for d in range(DESIGN['draws']):
        e=rng.normal(size=(n,DESIGN['rank']))@rng.normal(size=(DESIGN['rank'],n))
        e*=np.sqrt(energy)/np.linalg.norm(e)
        u,_,vt=svd(base+DESIGN['gamma']*e,full_matrices=False,check_finite=False)
        theta[d]=[angles(u),angles(vt.T)]
        if (d+1)%16==0:print(f'Endpoint SVDs: {d+1}/{DESIGN["draws"]}',flush=True)
    np.savez_compressed(OUT/'quantile_endpoints.npz',s=s,chi=chi,theta=theta)


def summarize(theta,indices):
    # RMS over indices and independent draws; propagate two SEs in squared angle.
    z=(theta[:,indices]**2).mean(axis=1)
    mean=z.mean();se=z.std(ddof=1)/np.sqrt(len(z))
    return np.sqrt(mean),np.sqrt(max(0,mean-2*se)),np.sqrt(mean+2*se),mean,se


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter,NullLocator
    import figure_style as style
    style.configure()
    a=np.load(OUT/'quantile_endpoints.npz')
    old=ROOT/'results/fixed_matrix/fixed_matrix_validation.npz'
    b=np.load(old)
    n=DESIGN['n'];gamma=DESIGN['gamma'];degrees=180/np.pi
    bins=np.array_split(np.arange(n),DESIGN['bins'])
    xs=np.array([(idx+.5).mean()/n for idx in bins])
    bulk=gamma*np.sqrt(n/6*(4-a['s']**2))
    exact=gamma*np.sqrt(a['chi'])
    bulk_bin=np.array([np.sqrt((bulk[idx]**2).mean()) for idx in bins])
    exact_bin=np.array([np.sqrt((exact[idx]**2).mean()) for idx in bins])
    fig,ax=plt.subplots(1,2,figsize=(6.4,2.10))
    blue=style.COLORS['lora'];orange=style.COLORS['oft']
    ax[0].plot(xs,bulk_bin*degrees,color='0.5',ls='--',lw=1.2,label='Bulk index formula')
    ax[0].plot(xs,exact_bin*degrees,color='black',lw=1,label='Exact local prediction')
    rows=[]
    for side,marker in [(0,'o'),(1,'x')]:
        vals=np.array([summarize(a['theta'][:,side],idx) for idx in bins])
        ax[0].errorbar(xs,vals[:,0]*degrees,
            yerr=np.array([vals[:,0]-vals[:,1],vals[:,2]-vals[:,0]])*degrees,
            color=blue,marker=marker,markerfacecolor='white',ms=3.3,ls='none',capsize=1.5,
            elinewidth=.55,label=['Measured $U$','Measured $V$'][side])
        for i,idx in enumerate(bins):
            rows.append(dict(side=['U','V'][side],bin=i,index_start=int(idx[0]),index_end=int(idx[-1]),
                exact_prediction_deg=float(exact_bin[i]*degrees),bulk_prediction_deg=float(bulk_bin[i]*degrees),
                observed_deg=float(vals[i,0]*degrees),
                z_exact=float((vals[i,3]-exact_bin[i]**2)/vals[i,4])))
    ax[0].set(xlabel=r'Singular index $i/n$ (top $\rightarrow$ tail)',ylabel=r'RMS angle ($^{\circ}$)',
              xlim=(0,1),ylim=(0,.12))
    ax[0].set_title('(a) Larger index, larger angle',loc='left')
    ax[0].text(.03,.96,r'Quantile spectrum; $n=512$',transform=ax[0].transAxes,va='top',fontsize=7)
    ax[0].legend(loc='lower right',fontsize=6.8,labelspacing=.25)
    # Same strength as the already frozen Gaussian-base experiment, without rerunning it.
    strengths=[1e-7,1e-4,1e-3,1e-2,1e-1]
    gi=strengths.index(gamma)
    gaussian_bands=np.array_split(np.arange(256),3)
    gaussian_rows=[]
    for mi,sides,color in [(1,[0,1],blue),(3,[1],orange)]:
        pred=np.array([gamma*np.sqrt(b['chi'][idx].mean()) if mi==1 else gamma for idx in gaussian_bands])
        ax[1].plot(np.arange(3),pred*degrees,color='black',lw=1)
        for side in sides:
            theta=gamma*np.sqrt(b['measured'][:,mi,gi,side])
            vals=np.array([summarize(theta,idx) for idx in gaussian_bands])
            marker='s' if mi==3 else ['o','x'][side]
            ax[1].errorbar(np.arange(3),vals[:,0]*degrees,
                yerr=np.array([vals[:,0]-vals[:,1],vals[:,2]-vals[:,0]])*degrees,
                color=color,marker=marker,markerfacecolor='white' if side==0 else color,
                ms=4,ls='none',capsize=2)
            for bi in range(3):
                gaussian_rows.append(dict(method='rank8' if mi==1 else 'block32',side=['U','V'][side],
                    band=['top','middle','tail'][bi],predicted_deg=float(pred[bi]*degrees),
                    observed_deg=float(vals[bi,0]*degrees),z=float((vals[bi,3]-pred[bi]**2)/vals[bi,4])))
    ax[1].set_yscale('log')
    ax[1].set(xticks=[0,1,2],xticklabels=['Top','Middle','Tail'],
              ylabel=r'RMS angle ($^{\circ}$)',ylim=(.0035,.24))
    ax[1].set_yticks([.005,.01,.05,.1,.2])
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
    ax[1].yaxis.set_minor_locator(NullLocator())
    ax[1].set_title('(b) Predicting the three bands',loc='left')
    ax[1].text(.03,.96,r'Gaussian matrix; $n=256$',transform=ax[1].transAxes,va='top',fontsize=7)
    handles=[Line2D([],[],color='black',lw=1,label='Prediction'),
             Line2D([],[],color=blue,marker='o',mfc='white',ls='none',label='Rank-8 $U$'),
             Line2D([],[],color=blue,marker='x',ls='none',label='Rank-8 $V$'),
             Line2D([],[],color=orange,marker='s',ls='none',label='Block-32 $V$')]
    ax[1].legend(handles=handles,loc='center right',fontsize=6.8,labelspacing=.25)
    fig.subplots_adjust(left=.09,right=.99,top=.87,bottom=.21,wspace=.34)
    style.prepare(fig)
    figure=ROOT/'figures/spectral_angle_prediction.pdf'
    figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure,bbox_inches='tight',pad_inches=.035);plt.close(fig)
    report=dict(design=DESIGN,quantile_bins=rows,gaussian_bands=gaussian_rows,
        quantile_max_abs_z=max(abs(row['z_exact']) for row in rows),
        quantile_mean_absolute_relative_angle_error=float(np.mean([abs(row['observed_deg']/row['exact_prediction_deg']-1) for row in rows])),
        bulk_mean_absolute_relative_angle_error=float(np.mean([abs(row['observed_deg']/row['bulk_prediction_deg']-1) for row in rows])),
        gaussian_max_abs_z=max(abs(row['z']) for row in gaussian_rows),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'quantile_endpoints.npz',old,
                    Path(__file__),Path(__file__).with_name('validate_index_model.py'),ROOT/'analysis/plot_palette.py']},
        figure_sha256=sha(figure))
    (OUT/'angle_prediction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['quantile_max_abs_z','quantile_mean_absolute_relative_angle_error','gaussian_max_abs_z']},indent=2))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if '--plot-only' not in sys.argv:run()
    plot()
