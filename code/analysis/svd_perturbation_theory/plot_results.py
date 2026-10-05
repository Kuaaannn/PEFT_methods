"""Plots and numerical summaries for the synthetic SVD experiments."""
from pathlib import Path
import json,os,sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/svd_perturbation'
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')
import numpy as np
sys.path.insert(0,str(ROOT/'analysis'))
from figure_style import configure
from plot_palette import PALETTE
configure()
import matplotlib as _mpl
_mpl.rcParams['text.latex.preamble'] += r'\usepackage{bm}'
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                     'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})
A=np.load(OUT/'sweep.npz')
D=json.loads((OUT/'validation.json').read_text())
DEST=OUT/'figures';DEST.mkdir(exist_ok=True)
g=A['gammas'];bands=np.array_split(np.arange(256),3)
colors=[PALETTE['FullFTColor'],PALETTE['LoRAColor'],PALETTE['OFTColor'],mpl.colors.to_hex(.7*np.array(mpl.colors.to_rgb(PALETTE['OFTColor']))+.3)]
names=['Dense additive','Rank-8 additive','Full Cayley','Block-32 Cayley']
styles=['-','-','-','--'];markers=['s','o','D','^']

def finish(fig,name):
    for ax in fig.axes:
        for sp in ax.spines.values():sp.set_color('black')
        ax.tick_params(colors='black');ax.grid(axis='y',alpha=.16,lw=.4)
    for artist in fig.findobj(mpl.text.Text):
        value=artist.get_text().replace('$U$',r'$\bm{U}$').replace('$V$',r'$\bm{V}$').replace('$U;',r'$\bm{U};').replace('$V;',r'$\bm{V};')
        artist.set_text(value)
    fig.savefig(DEST/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(DEST/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)

def mean_sd_band(k,side,principal=False):
    vals=A['principal' if principal else 'angles'][:,:,k,side,:]*180/np.pi
    vals=np.stack([vals[...,b].mean(axis=-1) for b in bands],axis=-1)
    return vals.mean(axis=0),vals.std(axis=0,ddof=1)

fig,axs=plt.subplots(1,3,figsize=(7.2,2.15),layout='constrained')
x=np.arange(1,257)
axs[0].plot(x,A['s0'],color='black');axs[0].set_ylabel('Singular value')
axs[1].semilogy(x[:-1],-np.diff(A['s0']),color=colors[1],lw=.7);axs[1].set_ylabel('Adjacent gap')
axs[2].semilogy(x,np.sqrt(A['chi']),color=colors[1],lw=.7);axs[2].set_ylabel(r'$\sqrt{\chi_i}$ (radians / strength)')
for ax in axs:
    ax.set_xlabel('Singular index (descending spectrum)')
    for boundary in [86.5,171.5]:ax.axvline(boundary,color='black',ls=':',lw=.6)
finish(fig,'base_geometry')

k=int(np.argmin(abs(g-.01)))
fig,axs=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
for side,ax in enumerate(axs):
    mean,sd=mean_sd_band(k,side)
    for m in range(4):
        ax.errorbar(np.arange(3)+(m-1.5)*.025,mean[m],yerr=sd[m],color=colors[m],
                    marker=markers[m],ls=styles[m],lw=1.4,capsize=2,label=names[m])
    ax.set_xticks(range(3),['Top','Middle','Tail'])
    ax.set_title('$'+('U' if side==0 else 'V')+r'$ rotation at $\gamma=0.01$')
    ax.set_ylabel('Mean vector angle (degrees)');ax.set_ylim(-.5,14)
axs[0].legend(loc='upper left',fontsize=7.5)
finish(fig,'band_profiles')

fig,axs=plt.subplots(2,3,figsize=(7.2,3.6),layout='constrained',sharex=True,sharey=True)
vals=A['angles']*180/np.pi
for side in range(2):
 for b,idx in enumerate(bands):
    ax=axs[side,b]
    for m in range(4):
        y=vals[:,m,:,side,:][...,idx].mean(axis=-1)
        mean=y.mean(axis=0);sd=y.std(axis=0,ddof=1)
        if side==0 and m>=2:mean=np.zeros_like(mean);sd=np.zeros_like(sd)
        ax.semilogx(g[1:],mean[1:],color=colors[m],ls=styles[m],label=names[m])
        ax.fill_between(g[1:],np.maximum(0,mean[1:]-sd[1:]),mean[1:]+sd[1:],color=colors[m],alpha=.1,lw=0)
    ax.set_ylim(-2,93)
    if side==0:ax.set_title(['Top','Middle','Tail'][b])
    if b==0:ax.set_ylabel('$'+('U' if side==0 else 'V')+'$ angle (degrees)')
    if side==1:ax.set_xlabel(r'Strength $\gamma$')
axs[0,0].legend(loc='upper left',fontsize=7.2)
finish(fig,'band_paths')

fig,axs=plt.subplots(2,2,figsize=(7.2,3.6),layout='constrained')
for row,gamma in enumerate([1e-4,1e-2]):
 k=int(np.argmin(abs(g-gamma)))
 for side in [0,1]:
    ax=axs[row,side]
    for m in range(4):
        y=np.sqrt(np.mean(A['angles'][:,m,k,side,:]**2,axis=0))*180/np.pi
        if m>=2 and side==0:continue
        ax.plot(x,y,color=colors[m],ls=styles[m],lw=.65,alpha=.9,label=names[m])
    ax.plot(x,gamma*np.sqrt(A['chi'])*180/np.pi,color='black',ls=':',lw=.7,label='Local RMS prediction')
    ax.set_title('$'+('U' if side==0 else 'V')+r'$; $\gamma='+('10^{-4}' if row==0 else '10^{-2}')+'$')
    ax.set_ylabel('RMS angle over draws (degrees)')
    if row==1:ax.set_xlabel('Singular index')
    for boundary in [86.5,171.5]:ax.axvline(boundary,color='black',ls=':',lw=.4)
axs[0,1].legend(fontsize=6.2,ncol=2)
finish(fig,'index_profiles')

fig,axs=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
for m in [0,1]:
    ys=A['drift'][:,m,1:];mu=ys.mean(axis=0);sd=ys.std(axis=0,ddof=1)
    axs[0].loglog(g[1:],mu,color=colors[m],label=names[m])
    axs[0].fill_between(g[1:],mu-sd,mu+sd,color=colors[m],alpha=.15,lw=0)
    axs[1].semilogx(g[1:],mu/g[1:],color=colors[m],label=names[m])
small=(g>0)&(g<=.03)
axs[0].loglog(g[small],g[small]/16,color='black',ls=':',label=r'Local $\gamma/\sqrt n$')
axs[0].loglog(g[1:],np.sqrt(1+g[1:]**2)-1,color='black',ls='--',lw=.8,label='Dense large-$n$ limit')
axs[1].axhline(1/16,color='black',ls=':',label=r'Local $1/\sqrt n$')
axs[0].set_ylabel('Relative spectrum / restoration distance')
axs[1].set_ylabel('Spectrum distance / update distance')
axs[0].set_ylim(1e-7,20);axs[1].set_ylim(0,1.03)
for ax in axs:ax.set_xlabel(r'Strength $\gamma$');ax.legend(fontsize=7.2)
finish(fig,'spectrum_drift')

fig,axs=plt.subplots(1,3,figsize=(7.2,2.35),layout='constrained')
smooth=np.linspace(0,10,401);q=np.sqrt(1+smooth*smooth);cosine=4/(1+q)-1
for m in [2,3]:
    y=(A['angles'][:,m,:,1,:]*180/np.pi).mean(axis=-1)
    axs[0].plot(g,y.mean(axis=0),color=colors[m],ls=styles[m],label=names[m])
    axs[1].plot(g,A['displacement'][:,m].mean(axis=0),color=colors[m],ls=styles[m])
axs[0].plot(smooth,np.degrees(np.arccos(abs(cosine))),color='black',ls=':',label='Full large-$n$ limit')
axs[1].plot(smooth,2*smooth/(1+q),color='black',ls=':')
axs[0].axvline(np.sqrt(8),color='black',ls=':',lw=.5)
axs[0].set_ylabel('$V$ angle (degrees)');axs[1].set_ylabel('Relative weight displacement')
for ax in axs[:2]:ax.set_xlabel(r'Strength $\gamma$');ax.set_xlim(0,10)
axs[0].legend(fontsize=6.3,loc='lower center')
small=D['analytic_checks']['normalized_skew_small_n']
axs[2].plot(range(1,9),small['predicted'],color='black',lw=1,label='Exact normalized theory')
axs[2].errorbar(range(1,9),small['measured'],yerr=small['se'],fmt='o',ms=3,color=colors[2],label='Monte Carlo (50,000)')
axs[2].axhline(1,color='black',ls=':',lw=.5)
axs[2].set_ylabel('Expected squared angular speed');axs[2].set_xlabel('Singular index ($n=8$)')
axs[2].legend(fontsize=6.1)
finish(fig,'cayley_curve')

fig,axs=plt.subplots(2,3,figsize=(7.2,3.3),layout='constrained',sharey=True)
k=int(np.argmin(abs(g-1)))
for side in [0,1]:
    av,sv=mean_sd_band(k,side);ap,sp=mean_sd_band(k,side,True)
    for col,m in enumerate([0,1,2]):
        ax=axs[side,col]
        ax.errorbar(range(3),av[m],yerr=sv[m],color=colors[m],marker='o',label='Individual vectors',capsize=2)
        ax.errorbar(range(3),ap[m],yerr=sp[m],color=colors[m],marker='s',ls='--',label='Band principal angles',capsize=2)
        ax.set_xticks(range(3),['Top','Middle','Tail']);ax.set_ylim(-3,93)
        if side==0:ax.set_title(names[m])
        if col==0:ax.set_ylabel('$'+('U' if side==0 else 'V')+'$ mean angle (degrees)')
axs[0,0].legend(fontsize=7.2,loc='lower center')
finish(fig,'subspaces')

summary={'chi_rms_bands':[float(np.sqrt(A['chi'][b].mean())) for b in bands],
         'chi_squared_bands':[float(A['chi'][b].mean()) for b in bands],
         'normalized_full_rates_bands':[float(A['full_skew_rates'][b].mean()) for b in bands],
         'gamma_001':{},'gamma_1':{},'figures':sorted(p.name for p in DEST.glob('*.pdf'))}
for key,gamma in [('gamma_001',.01),('gamma_1',1)]:
    k=int(np.argmin(abs(g-gamma)))
    for m,name in enumerate(names):
        av,sd=mean_sd_band(k,0);ap,_=mean_sd_band(k,0,True)
        summary[key][name]={'mean_u_bands':av[m].tolist(),'mean_principal_u_bands':ap[m].tolist(),
                            'spectral_drift':float(A['drift'][:,m,k].mean())}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
