"""Fixed-matrix SVD perturbation checks. Never reads or changes manuscript data.

Run from the workspace root with .venv/bin/python. All arrays are float64.
The theory uses conditional expectations given one fixed base matrix.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve, svd
from scipy.special import roots_laguerre

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/svd_perturbation_theory"
METHODS = ["dense", "rank8", "full_cayley", "block32_cayley"]


def factor(w):
    u, s, vt = svd(w, full_matrices=False, check_finite=False)
    return u, s, vt.T


def vector_angles(q0, q):
    c = np.sum(q0 * q, axis=0)
    residual = q - q0 * c
    return np.arctan2(np.linalg.norm(residual, axis=0), np.abs(c))


def skew(rng, n, block):
    z = np.zeros((n, n))
    for start in range(0, n, block):
        stop = min(start + block, n)
        a = np.triu(rng.standard_normal((stop-start, stop-start)), 1)
        z[start:stop, start:stop] = a - a.T
    return z


def derivatives(s, h):
    si, sj = s[None, :], s[:, None]
    den = si**2 - sj**2
    np.fill_diagonal(den, np.inf)
    a = (si*h + sj*h.T) / den
    b = (sj*h + si*h.T) / den
    d2 = np.sum((si*(h*h+h.T*h.T)+2*sj*h*h.T)/den, axis=0)
    return np.diag(h).copy(), a, b, d2


def susceptibility(s, energy):
    a, b = s[:, None], s[None, :]
    den = (a*a-b*b)**2
    np.fill_diagonal(den, np.inf)
    pairs = energy/len(s)**2 * (a*a+b*b)/den
    return pairs.sum(axis=1), pairs


def exact_full_rates(s, order=100):
    """Laplace-transform quadrature for per-draw normalized full skew noise."""
    n = len(s)
    lam, energy = s*s, float(np.dot(s,s))
    weights = lam[:, None] + lam[None, :]
    pairs = weights[np.triu_indices(n, 1)]
    mu = (n-1)*energy
    x, qw = roots_laguerre(order)
    out = np.zeros(n)
    for t, wt in zip(x, qw):
        logprod = -.5*np.log1p(2*t*pairs/mu).sum()
        inv = 1/(1+2*t*weights/mu)
        np.fill_diagonal(inv, 0)
        out += wt*np.exp(t+logprod)*inv.sum(axis=1)/(n-1)
    return out


def finite_sweep():
    n, rank, block = 256, 8, 32
    rng = np.random.default_rng(20260916)
    w0 = rng.standard_normal((n, n))/np.sqrt(n)
    u0, s0, v0 = factor(w0)
    norm0 = np.linalg.norm(w0)
    bands = np.array_split(np.arange(n), 3)
    gammas = np.r_[0., np.logspace(-4, 1, 51)]
    seeds = [13, 37, 73]
    shape = (len(seeds), len(METHODS), len(gammas))
    angles = np.zeros((*shape, 2, n))
    principal = np.zeros((*shape, 2, n))
    spectra = np.zeros((*shape, n))
    displacement = np.zeros(shape)
    drift = np.zeros(shape)
    generators = np.zeros((len(seeds), len(METHODS), n, n))
    tangent = np.zeros((len(seeds), len(METHODS), 2, n))
    radial = np.zeros((len(seeds), len(METHODS)))
    checks = {"initial_speed_relative_errors": [], "orthogonality_errors": [],
              "transport_U_errors": [], "transport_V_errors": [],
              "orthogonal_spectrum_relative_errors": [],
              "local_angle_relative_errors": [], "local_spectrum_relative_errors": []}
    for d, seed in enumerate(seeds):
        rr = np.random.default_rng(seed)
        g = rr.standard_normal((n, n))
        ba = np.dot(rr.standard_normal((n, rank)), rr.standard_normal((rank, n)))
        zfull, zblock = skew(rr, n, n), skew(rr, n, block)
        directions = [norm0*g/np.linalg.norm(g), norm0*ba/np.linalg.norm(ba),
                      norm0*zfull/(2*np.linalg.norm(np.dot(w0,zfull))),
                      norm0*zblock/(2*np.linalg.norm(np.dot(w0,zblock)))]
        for m, direction in enumerate(directions):
            generators[d, m] = direction
            e = direction if m < 2 else 2*np.dot(w0,direction)
            checks["initial_speed_relative_errors"].append(abs(np.linalg.norm(e)/norm0-1))
            ds, a, b, _ = derivatives(s0, np.dot(np.dot(u0.T,e),v0))
            tangent[d, m, 0] = np.sum(a*a, axis=0)
            tangent[d, m, 1] = np.sum(b*b, axis=0)
            radial[d, m] = np.sum(ds*ds)/norm0**2
            # A much smaller point than the display sweep checks derivatives.
            h = 1e-7
            if m < 2:
                wu, ss, wv = factor(w0+h*e)
                for side, (q0, q) in enumerate([(u0, wu), (v0, wv)]):
                    observed = vector_angles(q0, q)/h
                    predicted = np.sqrt(tangent[d, m, side])
                    checks["local_angle_relative_errors"].append(float(
                        np.linalg.norm(observed-predicted)/np.linalg.norm(predicted)))
                checks["local_spectrum_relative_errors"].append(float(
                    np.linalg.norm((ss-s0)/h-ds)/np.linalg.norm(ds)))
            for k, gamma in enumerate(gammas):
                if gamma == 0:
                    spectra[d, m, k] = s0
                    continue
                if m < 2:
                    w = w0+gamma*direction
                else:
                    eye = np.eye(n)
                    r = solve(eye-gamma*direction, eye+gamma*direction,
                              assume_a="gen", check_finite=False)
                    w = np.dot(w0,r)
                    checks["orthogonality_errors"].append(float(np.linalg.norm(np.dot(r.T,r)-eye)/np.sqrt(n)))
                u, s, v = factor(w)
                spectra[d, m, k] = s
                displacement[d, m, k] = np.linalg.norm(w-w0)/norm0
                drift[d, m, k] = np.linalg.norm(s-s0)/norm0
                for side, (q0, q) in enumerate([(u0, u), (v0, v)]):
                    angles[d, m, k, side] = vector_angles(q0, q)
                    for inds in bands:
                        cross = np.dot(q0[:, inds].T,q[:, inds])
                        # Principal angles through residual SVD avoids acos near 1.
                        resid = q[:, inds]-np.dot(q0[:, inds],cross)
                        sins = svd(resid, compute_uv=False, check_finite=False)
                        principal[d, m, k, side, inds] = np.arcsin(np.clip(sins, 0, 1))
                if m >= 2:
                    signs = np.sign(np.sum(u0*u, axis=0))
                    checks["transport_U_errors"].append(float(np.linalg.norm(u*signs-u0)/np.sqrt(n)))
                    checks["transport_V_errors"].append(float(np.linalg.norm(v*signs-np.dot(r.T,v0))/np.sqrt(n)))
                    checks["orthogonal_spectrum_relative_errors"].append(float(drift[d, m, k]))
            print(f"finished draw={seed}, method={METHODS[m]}", flush=True)
    chi, pair = susceptibility(s0, norm0**2)
    rates = exact_full_rates(s0)
    rates2 = exact_full_rates(s0, 160)
    assert np.max(abs(rates-rates2)) < 1e-10
    assert abs(np.dot(s0*s0,rates)/norm0**2-1) < 1e-10
    assert np.all(np.diff(rates) >= -1e-12)
    for arr in [angles, principal, spectra, displacement, drift, generators, tangent, radial]:
        assert np.isfinite(arr).all()
    np.savez_compressed(OUT/"sweep.npz", w0=w0, u0=u0, s0=s0, v0=v0,
                        seeds=seeds, methods=METHODS, gammas=gammas,
                        angles=angles, principal=principal, spectra=spectra,
                        displacement=displacement, drift=drift, generators=generators,
                        tangent=tangent, radial=radial, chi=chi, pairs=pair,
                        full_skew_rates=rates)
    return {k:float(max(v)) for k,v in checks.items()}


def moment_checks():
    """Many draws check ensemble predictions independently of the 3-draw sweep."""
    data = np.load(OUT/"sweep.npz")
    s, w0, v0 = data["s0"], data["w0"], data["v0"]
    n, energy = len(s), float(np.dot(s,s))
    bands = np.array_split(np.arange(n), 3)
    rng = np.random.default_rng(271828)
    rows = []
    draws = 2048
    for kind in ["dense", "rank1", "rank8", "rank32"]:
        radial, uv, coherent, weight_components = [], [], [], []
        for _ in range(draws):
            if kind == "dense": h = rng.standard_normal((n,n))
            else:
                r = int(kind[4:])
                h = np.dot(rng.standard_normal((n,r)),rng.standard_normal((r,n)))
            h *= np.sqrt(energy)/np.linalg.norm(h)
            ds,a,b,_ = derivatives(s,h)
            radial.append(float(np.dot(ds,ds)/energy))
            uv.append([[float(np.mean(np.sum(t*t,axis=0)[idx])) for idx in bands] for t in [a,b]])
            coherent.append([float(np.sum((a+b)**2)),float(np.sum((a-b)**2))])
            weight_components.append([float(np.linalg.norm((h+h.T)/2)**2-np.dot(ds,ds)),
                                      float(np.linalg.norm((h-h.T)/2)**2)])
        rad=np.asarray(radial); vals=np.asarray(uv);co=np.asarray(coherent)
        prediction=np.array([data['chi'][idx].mean() for idx in bands])
        se=vals.std(axis=0,ddof=1)/np.sqrt(draws)
        z=np.max(abs(vals.mean(axis=0)-prediction)/se)
        assert z < 6
        assert abs(rad.mean()-1/n) < 6*rad.std(ddof=1)/np.sqrt(draws)
        rows.append(dict(kind=kind, draws=draws, spectral_fraction_mean=float(rad.mean()),
                         spectral_fraction_se=float(rad.std(ddof=1)/np.sqrt(draws)),
                         band_rates_mean=vals.mean(axis=0).tolist(),band_rates_se=se.tolist(),
                         max_standard_errors=float(z),
                         opposite_over_common_energy=float(co[:,1].sum()/co[:,0].sum()),
                         opposite_over_common_weight_energy=float(np.sum(weight_components,axis=0)[1]/np.sum(weight_components,axis=0)[0])))
    # Block-skew rates: fixed-base, actual per-draw normalization.
    for block in [32,256]:
        rates=[]
        for _ in range(512):
            z=skew(rng,n,block)
            k=z*np.sqrt(energy)/(2*np.linalg.norm(np.dot(w0,z)))
            rates.append(4*np.sum((np.dot(k,v0))**2,axis=0))
        rates=np.asarray(rates)
        rows.append(dict(kind=f'block{block}_normalized_skew',draws=len(rates),
                         band_rates_mean=[float(rates[:,idx].mean()) for idx in bands],
                         weighted_rate_error=float(np.max(abs(np.dot(rates,s*s)/energy-1)))))
    return rows


def analytic_checks():
    rng=np.random.default_rng(314159)
    out={}
    # Second derivatives, including a zero-diagonal tangent direction.
    n=9;s=np.linspace(3.,.6,n);h=rng.normal(size=(n,n));base=np.diag(s)
    ds,a,b,d2=derivatives(s,h)
    eps=1e-4
    sp=svd(base+eps*h,compute_uv=False);sm=svd(base-eps*h,compute_uv=False)
    err=np.linalg.norm((sp+sm-2*s)/eps**2-d2)/np.linalg.norm(d2)
    assert err < 1e-5
    out['second_derivative_relative_error']=float(err)
    # Local spectral restoration and normal projection.
    errs=[]
    for eps in [1e-3,1e-4,1e-5]:
        u,t,v=factor(base+eps*h);rest=np.dot(u*s,v.T)
        residual=(base+eps*h-rest)/eps
        errs.append(float(np.linalg.norm(residual-np.diag(ds))))
        np.testing.assert_allclose(np.linalg.norm(base+eps*h-rest),np.linalg.norm(t-s),atol=1e-13)
    assert errs[-1] < errs[0]/50
    out['restoration_projection_errors']=errs
    # Repeated identity: first-order spectrum is eigenvalues of symmetric noise.
    repeated=[]
    for _ in range(1024):
        e=rng.normal(size=(32,32));e/=np.linalg.norm(e)
        repeated.append(float(np.linalg.norm((e+e.T)/2)**2))
    out['repeated_identity_spectral_fraction']=float(np.mean(repeated))
    out['repeated_identity_prediction']=33/64
    # Rectangular derivative energy includes leakage into the larger null complement.
    m,n=48,24;s=np.linspace(2.,.4,n);energy=float(np.dot(s,s));tau2=energy/(m*n)
    chi,_=susceptibility(s,energy*n/m)
    rect=[]
    for _ in range(2048):
        e=rng.normal(size=(m,n));e*=np.sqrt(energy)/np.linalg.norm(e)
        _,a,b,_=derivatives(s,e[:n])
        rect.append([np.mean(np.sum(a*a,axis=0)+np.sum(e[n:]**2,axis=0)/s**2),
                     np.mean(np.sum(b*b,axis=0))])
    vals=np.asarray(rect);pred=[float(np.mean(chi+(m-n)*tau2/s**2)),float(chi.mean())]
    assert np.max(abs(vals.mean(axis=0)-pred)/(vals.std(axis=0,ddof=1)/np.sqrt(len(vals))))<6
    out['rectangular']={'measured':vals.mean(axis=0).tolist(),'predicted':pred}
    # Exact per-draw normalization bias at small dimension, where it is visible.
    s=np.geomspace(3.,.3,8);pred=exact_full_rates(s,160);rates=[]
    for _ in range(50000):
        z=skew(rng,8,8);q=np.linalg.norm(s[:,None]*z)**2
        rates.append((np.dot(s,s))*np.sum(z*z,axis=0)/q)
    vals=np.asarray(rates)
    assert np.max(abs(vals.mean(axis=0)-pred)/(vals.std(axis=0,ddof=1)/np.sqrt(len(vals)))) < 6
    out['normalized_skew_small_n']={'spectrum':s.tolist(),'predicted':pred.tolist(),
                                  'measured':vals.mean(axis=0).tolist(),
                                  'se':(vals.std(axis=0,ddof=1)/np.sqrt(len(vals))).tolist()}
    # An independent contraction checks representative BLAS products.
    x=rng.normal(size=(37,23));y=rng.normal(size=(23,31))
    out['dot_einsum_max_error']=float(np.max(abs(np.dot(x,y)-np.einsum('ik,kj->ij',x,y,optimize=False))))
    assert out['dot_einsum_max_error'] < 1e-12
    rank_limits=[]
    for n in [64,128,256,512]:
        base=rng.normal(size=(n,n))/np.sqrt(n);sig=svd(base,compute_uv=False)
        update=np.outer(rng.normal(size=n),rng.normal(size=n));update*=np.linalg.norm(base)/np.linalg.norm(update)
        drift=np.linalg.norm(svd(base+update,compute_uv=False)-sig)/np.linalg.norm(base)
        bound=max(0.,1-2*sig[0]/np.linalg.norm(base))
        assert bound <= drift+1e-12 <= 1+1e-12
        rank_limits.append(dict(n=n,rank=1,gamma=1.,drift=float(drift),lower_bound=float(bound)))
    out['rank_fixed_strength']=rank_limits
    # Deterministic identities and finite local band-subspace derivatives.
    n=12;s=np.linspace(2.5,.5,n);h=rng.normal(size=(n,n));ds,a,b,_=derivatives(s,h)
    si,sj=s[None,:],s[:,None];difference=si-sj;summation=si+sj
    np.fill_diagonal(difference,np.inf)
    common=(h+h.T)/(2*difference);opposite=(h-h.T)/(2*summation)
    weighted=np.sum(ds*ds)
    for i in range(n):
        for j in range(i+1,n):
            weighted+=2*((s[i]-s[j])**2*common[j,i]**2+(s[i]+s[j])**2*opposite[j,i]**2)
    out['gap_weighted_energy_relative_error']=float(abs(weighted/np.sum(h*h)-1))
    assert out['gap_weighted_energy_relative_error']<1e-12
    u,_,v=factor(np.diag(s)+1e-6*h);index=np.arange(4)
    pred=np.sum(a[4:,:4]**2);observed=np.linalg.norm(u[4:,:4])**2/1e-12
    out['local_subspace_relative_error']=float(abs(observed/pred-1))
    assert out['local_subspace_relative_error']<1e-4
    ut,_,vt=factor((np.diag(s)+.1*h).T)
    u,_,v=factor(np.diag(s)+.1*h)
    out['transpose_angle_swap_error']=float(max(np.max(abs(vector_angles(np.eye(n),u)-vector_angles(np.eye(n),vt))),np.max(abs(vector_angles(np.eye(n),v)-vector_angles(np.eye(n),ut)))))
    assert out['transpose_angle_swap_error']<1e-12
    # Exact block formula in a deliberately nonuniform six-dimensional base.
    from scipy.integrate import quad
    n=6;v,_=np.linalg.qr(rng.normal(size=(n,n)));s=np.geomspace(3.,.3,n)
    w=s[:,None]*v.T;energy=float(np.dot(s,s));basis=[]
    for start in [0,3]:
        for i in range(start,start+3):
            for j in range(i+1,start+3):
                z=np.zeros((n,n));z[i,j]=1;z[j,i]=-1;basis.append(z)
    basis=np.array(basis);p=len(basis)
    wz=np.einsum('ij,ejk->eik',w,basis);matrix=np.einsum('eij,fij->ef',wz,wz)
    lam,q=np.linalg.eigh(matrix);noise=rng.normal(size=(100000,p));den=np.einsum('bi,ij,bj->b',noise,matrix,noise)
    pred=[];measured=[];se=[]
    for i in range(n):
        zv=np.einsum('eij,j->ei',basis,v[:,i]);num=np.dot(zv,zv.T)
        rotated=np.diag(np.dot(np.dot(q.T,num),q))
        def integrand(t):
            diag=1+2*t*lam
            return energy*np.exp(-.5*np.log(diag).sum())*np.sum(rotated/diag)
        theory=quad(integrand,0,np.inf,epsabs=1e-10,epsrel=1e-10)[0]
        obs=energy*np.einsum('bi,ij,bj->b',noise,num,noise)/den
        pred.append(theory);measured.append(float(obs.mean()));se.append(float(obs.std(ddof=1)/np.sqrt(len(obs))))
    assert np.max(abs(np.array(pred)-measured)/se)<6
    out['normalized_block_integral']={'predicted':pred,'measured':measured,'se':se}
    # Affine orthogonal tangent curvature versus the actual Cayley curve.
    n=8;s=np.linspace(3.,.5,n);z=skew(rng,n,n);k=.1*z;w=np.diag(s);e=2*np.dot(w,k)
    _,_,_,curvature=derivatives(s,e)
    exact=4*s*np.sum(k*k,axis=0)
    np.testing.assert_allclose(curvature,exact,rtol=1e-12,atol=1e-12)
    out['affine_orthogonal_curvature_relative_error']=float(np.linalg.norm(curvature-exact)/np.linalg.norm(exact))
    # The regular-spacing proposition: linear quantiles f(x)=2-x.
    regular=[]
    for n in [64,128,256,512,1024]:
        s=2-(np.arange(n)+.5)/n;energy=float(np.dot(s,s));chi,_=susceptibility(s,energy)
        observed=float(chi[n//2]/n);predicted=7*np.pi**2/18
        regular.append({'n':n,'observed':observed,'limit':predicted})
    assert abs(regular[-1]['observed']/predicted-1)<.003
    out['regular_spacing']=regular
    return out


def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=True)
    checks=finite_sweep()
    assert checks['initial_speed_relative_errors']<1e-12
    assert checks['orthogonality_errors']<1e-12
    assert checks['transport_U_errors']<1e-10
    assert checks['transport_V_errors']<1e-10
    assert checks['orthogonal_spectrum_relative_errors']<1e-12
    assert checks['local_angle_relative_errors']<1e-3
    assert checks['local_spectrum_relative_errors']<1e-3
    results={'status':'pass','base_seed':20260916,'perturbation_seeds':[13,37,73],
             'shape':[256,256],'rank':8,'block_size':32,'finite_sweep_cells':624,
             'normalization':'per-draw initial Frobenius speed equals base Frobenius norm',
             'sweep_checks':checks,'moment_checks':moment_checks(),
             'analytic_checks':analytic_checks(),
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    results['elapsed_seconds']=time.time()-start
    (OUT/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'status':results['status'],'sweep_checks':checks,'seconds':results['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
