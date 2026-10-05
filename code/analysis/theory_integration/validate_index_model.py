"""Fixed-base mathematical validation; no training or checkpoint evaluation.

Freeze the design before observing draws. Use actual gaps for conditional
predictions, and deterministic quantiles only for the quantile theorem.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/fixed_matrix'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mpl_cache'))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(ROOT / 'analysis'))
import numpy as np
from scipy.linalg import svd, solve
from scipy.optimize import brentq

DESIGN = dict(base='results/svd_perturbation/sweep.npz',
              base_seed=20260916, dimension=256, draw_seed=20260918,
              draws=64, strengths=[1e-7, 1e-4, 1e-3, 1e-2, 1e-1],
              methods=['dense', 'rank8', 'full_cayley', 'block32_cayley'],
              additive_normalization='per-draw norm equals base norm',
              cayley_normalization='fixed entry SD 1/(2 sqrt(block_size-1)); expected relative squared weight speed 1',
              pair_filter='none; all off-diagonal pairs',
              quantile_dimensions=[128, 256, 512, 1024, 2048],
              quantile_index_fractions=[.1, .5, .9])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(n):
    def survival(s):
        return 1 - (s*np.sqrt(max(0., 4-s*s))/2 + 2*np.arcsin(s/2))/np.pi
    return np.array([brentq(lambda s: survival(s)-x, 0., 2., xtol=1e-14)
                     for x in (np.arange(n)+.5)/n])


def susceptibility(s):
    diff=s[None, :]-s[:, None]
    total=s[None, :]+s[:, None]
    np.fill_diagonal(diff, np.inf)
    np.fill_diagonal(total, np.inf)
    d=(diff**-2).sum(axis=0)
    p=(total**-2).sum(axis=0)
    return np.dot(s,s)/(2*len(s)**2)*(d+p), d, p


def angular_coefficients(h, s):
    si, sj=s[None, :], s[:, None]
    gap=si*si-sj*sj
    np.fill_diagonal(gap, np.inf)
    return (si*h+sj*h.T)/gap, (sj*h+si*h.T)/gap


def angles(q):
    off=q.copy()
    np.fill_diagonal(off, 0)
    return np.arctan2(np.linalg.norm(off, axis=0), np.abs(np.diag(q)))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    source=ROOT/DESIGN['base']
    design={**DESIGN, 'source_sha256':digest(source), 'script_sha256':digest(Path(__file__))}
    config=OUT/'fixed_matrix_design.json'
    if config.exists():
        old=json.loads(config.read_text())
        assert {k:v for k,v in old.items() if k!='script_sha256'} == {k:v for k,v in design.items() if k!='script_sha256'}
    config.write_text(json.dumps(design,indent=2)+'\n')
    with np.load(source) as saved:
        w0,u0,s0,v0=[saved[k] for k in ['w0','u0','s0','v0']]
    n=len(s0); assert n==DESIGN['dimension']
    assert np.allclose(w0, (u0*s0)@v0.T, atol=1e-12)
    energy=np.sum(s0*s0); norm=np.sqrt(energy)
    chi,_,_=susceptibility(s0)
    bands=np.array_split(np.arange(n),3)
    rng=np.random.default_rng(DESIGN['draw_seed'])
    shape=(DESIGN['draws'],4,len(DESIGN['strengths']),2,n)
    measured=np.zeros(shape)
    tangent=np.zeros((DESIGN['draws'],4,2,n))
    error=np.zeros(shape[:3])
    displacement=np.zeros(shape[:3]+(2,))
    speed=np.zeros((DESIGN['draws'],4))
    spectral=np.zeros((DESIGN['draws'],2))
    identity=np.eye(n)
    off=~np.eye(n,dtype=bool)
    for draw in range(DESIGN['draws']):
        for method in range(4):
            if method<2:
                e=rng.normal(size=(n,n)) if method==0 else rng.normal(size=(n,8))@rng.normal(size=(8,n))
                e*=norm/np.linalg.norm(e)
                h=u0.T@e@v0
                a,b=angular_coefficients(h,s0)
                spectral[draw,method]=np.sum(np.diag(h)**2)/energy
            else:
                block=n if method==2 else 32
                k=np.zeros((n,n))
                for start in range(0,n,block):
                    upper=np.triu(rng.normal(size=(block,block)),1)
                    k[start:start+block,start:start+block]=(upper-upper.T)/(2*np.sqrt(block-1))
                t=v0.T@k@v0
                h=2*s0[:,None]*t
                a,b=angular_coefficients(h,s0)
                assert np.linalg.norm(a)<1e-9 and np.linalg.norm(b+2*t)<1e-9
            tangent[draw,method]=[np.sum(a*a,axis=0),np.sum(b*b,axis=0)]
            speed[draw,method]=np.sum(h*h)/energy
            predicted=np.r_[a[off],b[off]]
            for gi,gamma in enumerate(DESIGN['strengths']):
                if method<2:
                    matrix=np.diag(s0)+gamma*h
                else:
                    rotation=solve(identity-gamma*t,identity+gamma*t,assume_a='gen',check_finite=False)
                    matrix=s0[:,None]*rotation
                u,s,vt=svd(matrix,full_matrices=False,check_finite=False)
                v=vt.T
                signs=np.where(np.diag(u)+np.diag(v)<0,-1.,1.)
                u*=signs;v*=signs
                measured[draw,method,gi]=[angles(u)**2/gamma**2,angles(v)**2/gamma**2]
                actual=np.r_[u[off],v[off]]/gamma
                error[draw,method,gi]=np.linalg.norm(actual-predicted)/np.linalg.norm(predicted)
                displacement[draw,method,gi]=[np.linalg.norm(s-s0)/norm,np.linalg.norm(matrix-np.diag(s0))/norm]
        if (draw+1)%16==0:
            print(f'Completed {draw+1}/{DESIGN["draws"]} fixed-matrix draws',flush=True)
    np.savez_compressed(OUT/'fixed_matrix_validation.npz',w0=w0,s0=s0,chi=chi,
                        measured=measured,tangent=tangent,error=error,
                        displacement=displacement,speed=speed,spectral=spectral)
    report={'fixed_design':design,'methods':{}}
    for mi,name in enumerate(DESIGN['methods']):
        expected=chi if mi<2 else np.ones(n)
        rows=[]
        for side in (range(2) if mi<2 else [1]):
            for bi,indices in enumerate(bands):
                vals=measured[:,mi,0,side][:,indices].mean(axis=1)
                pred=expected[indices].mean()
                se=vals.std(ddof=1)/np.sqrt(len(vals))
                rows.append(dict(side=['U','V'][side],band=['top','middle','tail'][bi],
                                 prediction=float(pred),observed=float(vals.mean()),
                                 standard_error=float(se),z=float((vals.mean()-pred)/se)))
        report['methods'][name]={'band_squared_speed':rows,
             'max_smallest_strength_tangent_relative_error':float(error[:,mi,0].max()),
             'median_tangent_relative_error_by_strength':np.median(error[:,mi],axis=0).tolist(),
             'mean_relative_squared_weight_speed':float(speed[:,mi].mean())}
    qrows=[]
    for n_q in DESIGN['quantile_dimensions']:
        s=quantiles(n_q);qchi,d,p=susceptibility(s)
        for x in DESIGN['quantile_index_fractions']:
            i=min(n_q-1,round(n_q*x-.5));hx=(4-s[i]**2)/6
            qrows.append(dict(n=n_q,x=float((i+.5)/n_q),
                              normalized_susceptibility=float(qchi[i]/n_q),
                              prediction=float(hx),common_ratio=float(p[i]/d[i])))
    report['quantile_checks']=qrows
    report['spectral_fraction']={name:dict(mean=float(spectral[:,i].mean()),
        standard_error=float(spectral[:,i].std(ddof=1)/np.sqrt(len(spectral))),prediction=1/len(s0))
        for i,name in enumerate(DESIGN['methods'][:2])}
    report['maximum_cayley_spectral_displacement']=float(displacement[:,2:,:,0].max())
    (OUT/'fixed_matrix_report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert np.isfinite(measured).all() and np.isfinite(error).all()
    assert error[:,:,0].max()<1e-3
    assert displacement[:,2:,:,0].max()<1e-12
    # Algebra/numerics are asserted; stochastic agreement is reported without a pass filter.
    return report


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figure_style as style
    style.configure()
    saved=np.load(OUT/'fixed_matrix_validation.npz')
    report=json.loads((OUT/'fixed_matrix_report.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(6.4,2.05))
    colors=[style.COLORS['lora'],style.COLORS['oft']]
    s=quantiles(1024);chi,_,_=susceptibility(s);x=(np.arange(1024)+.5)/1024
    ax[0].plot(x,np.sqrt((4-s*s)/6),color='black',lw=1.2,label='Bulk limit')
    ax[0].plot(x,np.sqrt(chi/1024),color=colors[0],ls='--',lw=1.1,label='Exact quantile coefficient')
    ax[0].set(xlabel='Descending index fraction $x$',ylabel=r'RMS speed / $\sqrt{n}$',xlim=(0,1),ylim=(0,.88))
    ax[0].set_title('(a) Index prediction',loc='left')
    ax[0].legend(loc='lower right',fontsize=7)
    for mi,name in enumerate(DESIGN['methods']):
        for side in (range(2) if mi<2 else [1]):
            xband=np.arange(3)+(mi-1.5)*.10+(side-.5)*.035
            vals=[];err=[]
            for row in report['methods'][name]['band_squared_speed']:
                if row['side']!=['U','V'][side]:continue
                vals.append(row['observed']/row['prediction'])
                err.append(2*row['standard_error']/row['prediction'])
            label=['Dense','Rank-8','Full Cayley','Block-32'][mi]+r' $'+['U','V'][side]+r'$'
            ax[1].errorbar(xband,vals,yerr=err,color=colors[mi//2],marker=['o','s','^','D'][mi],
                          markerfacecolor='white' if side==0 else colors[mi//2],ms=3,
                          linestyle='none',capsize=2,label=label)
    ax[1].axhline(1,color='black',ls=':',lw=.8)
    ax[1].set(xticks=np.arange(3),xticklabels=['Top','Middle','Tail'],
              ylabel='Observed / predicted squared speed')
    ax[1].set_title('(b) Fixed Gaussian matrix',loc='left')
    handles,labels=ax[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.13),ncol=3,fontsize=7)
    fig.subplots_adjust(left=.09,right=.99,bottom=.23,top=.88,wspace=.38)
    style.prepare(fig)
    path=ROOT/'figures/spectral_index_validation.pdf'
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    (OUT/'figure_provenance.json').write_text(json.dumps({'figure':str(path.relative_to(ROOT)),
        'figure_sha256':digest(path),'script_sha256':digest(Path(__file__)),
        'input_sha256':digest(OUT/'fixed_matrix_validation.npz'),
        'report_sha256':digest(OUT/'fixed_matrix_report.json'),
        'palette':style.metadata()},indent=2)+'\n')


if __name__=='__main__':
    if '--plot-only' not in sys.argv:
        run()
    plot()
