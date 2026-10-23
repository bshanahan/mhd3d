"""Switchback diagnostics for the stationary diagonal BOUT++ geometry.

Dependencies: numpy, matplotlib. Import into the notebook containing your arrays.
Fields must be (time,x,y,z) or (x,y,z); no automatic axis inference.
B components are contravariant; velocity components are covariant.
Assumes g_ij = diag(a(y)**2, 1, a(y)**2), J=a(y)**2, mu0=1,
and positive background radial field. g_11 is the COVARIANT metric.

Example (68 x points including TWO guards at EACH end):
    r = analyze(Bx, By, Bz, g_11, time=t_array, mxg=2)
    plot_summary(r)
    plot_event(r)

Optional: rho=rho, vx=vx, vy=vy, vz=vz, divB=divB, y=y_coordinates.
Use mxg=0 if X guards were already removed. myg is analogous for Y.
Supply actual time / physical radial coordinates; otherwise indices are labeled.
The input arrays are not modified. Only one snapshot is expanded at a time.
The automatically selected event is the strongest deflection in the allowed
radial range, not a representative or independently verified switchback.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _ratio(n, d):
    n, d = np.broadcast_arrays(n, d)
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0)


def analyze(Bx, By, Bz, g_11, *, time=None, y=None, mxg=0, myg=0,
            rho=None, vx=None, vy=None, vz=None, divB=None, p=None,
            dx=None, dy=None, dz=None,
            event_y_margin=0, event=None):
    """Return small (time,y) diagnostics and an X cut through one event.

    p: evolved thermal pressure, same shape and normalized units as fields.
    dx,dy,dz: positive SCALAR uniform logical coordinate spacings (not
        physical transverse lengths). Supply all three or none. With no
        divB dump, these enable a C2 estimate of (1/J) d_i(J B^i).
        X,Z are periodic with no duplicated endpoint; Y uses one-sided
        second-order differences at the first/last retained cells.
        This estimate need not equal BOUT++'s discrete Div operator.
        With spacings, epsilon_div = min(a*dx,dy,a*dz)*abs(DivB)/|B|.
        It is a scale-dependent diagnostic, not a universal accuracy test.
    g_11: scalar, (y,), (x,y), or (x,y,z), aligned with untrimmed fields.
    rho, velocities and divB must have the same shape as Bx when supplied.
    divB: preferably the model's dumped Div(B), in its native units.
    event_y_margin: exclude this many additional Y cells at each end ONLY
        from automatic event selection; full profiles are still returned.
    event: optional (time,x,y,z) indices AFTER guard removal. For a single
        snapshot the time index is 0. Without event, select minimum By/|B|.
    y/time: one-dimensional coordinates aligned with input arrays.

    Slice means are transverse (x,z). Walén comparisons use slice-mean rho;
    they are approximate for appreciable density fluctuations. The outward
    normalized cross helicity is -2<dv.dbA>/(<dv^2>+<dbA^2>), so +1 means
    dv=-dbA in this positive-By convention. Correlations are NaN if variance
    is zero. Zero-|B| points are excluded from direction fractions; their
    fraction is reported. Nonfinite inputs and nonpositive rho are rejected.
    """
    fields = [np.asarray(q) for q in (Bx, By, Bz)]
    shape = fields[0].shape
    if len(shape) not in (3, 4) or any(q.shape != shape for q in fields):
        raise ValueError('B fields must share shape (t,x,y,z) or (x,y,z).')
    single = len(shape) == 3
    if single:
        fields = [q[None] for q in fields]
    nt, nx, ny, nz = fields[0].shape
    for name, n, size in [('mxg', mxg, nx), ('myg', myg, ny),
                           ('event_y_margin', event_y_margin, ny-2*myg)]:
        if not isinstance(n, (int, np.integer)) or n < 0 or 2*n >= size:
            raise ValueError(f'{name} must be a nonnegative integer leaving cells.')
    xs = slice(mxg, nx-mxg)
    ys = slice(myg, ny-myg)
    fields = [q[:, xs, ys, :] for q in fields]
    spatial = fields[0].shape[1:]
    metric = np.asarray(g_11, dtype=float)
    if metric.ndim == 1:
        metric = metric[None, :, None]
    elif metric.ndim == 2:
        metric = metric[:, :, None]
    elif metric.ndim > 3:
        raise ValueError('g_11 must be static: scalar, (y,), (x,y), or (x,y,z).')
    try:
        metric = np.broadcast_to(metric, (nx, ny, nz))[xs, ys, :]
    except ValueError as exc:
        raise ValueError('g_11 does not match untrimmed spatial dimensions.') from exc
    if not np.all(np.isfinite(metric)) or np.any(metric <= 0):
        raise ValueError('g_11 must be finite and positive.')
    if not np.allclose(metric, metric[:1, :, :1], rtol=1e-7, atol=1e-12):
        raise ValueError('This script assumes a depends only on y.')
    a = np.sqrt(metric)
    extras = {}
    for name, value in [('rho',rho), ('vx',vx), ('vy',vy), ('vz',vz), ('divB',divB), ('p',p)]:
        if value is not None:
            q = np.asarray(value)
            if q.shape != shape:
                raise ValueError(f'{name} must have shape {shape}.')
            extras[name] = (q[None] if single else q)[:, xs, ys, :]
    if any(k in extras for k in ('vx','vy','vz')) and 'rho' not in extras:
        raise ValueError('Velocity diagnostics also require rho.')
    def coord(value, size, name):
        q = np.arange(size, dtype=float) if value is None else np.asarray(value, dtype=float)
        if q.shape != (size,) or not np.all(np.isfinite(q)) or np.any(np.diff(q) <= 0):
            raise ValueError(f'{name} must have {size} finite, increasing entries.')
        return q
    t = coord(time, nt, 'time')
    yc = coord(y, ny, 'y')[ys]
    result = {'time':t, 'y':yc, 'time_label':'Output index' if time is None else 'Time',
              'y_label':'Radial index y' if y is None else 'Radial coordinate y',
              'mxg':mxg, 'myg':myg}
    if event is not None:
        if len(event) != 4 or any(not isinstance(k, (int,np.integer)) or k < 0 or k >= n
                                  for k,n in zip(event, (nt,)+spatial)):
            raise ValueError('event must be valid trimmed (time,x,y,z) indices.')
    spacing = None
    if any(q is not None for q in (dx,dy,dz)):
        if any(q is None or np.ndim(q) != 0 or not np.isfinite(q) or q <= 0
               for q in (dx,dy,dz)):
            raise ValueError('Supply all of dx,dy,dz as positive scalar logical spacings.')
        spacing = tuple(float(q) for q in (dx,dy,dz))
        if min(spatial) < 3:
            raise ValueError('Divergence estimate requires at least 3 cells per axis.')
        ell = np.minimum(np.minimum(a*dx,dy),a*dz)
    result['divB_source'] = ('dumped' if 'divB' in extras else
                             'Python C2 estimate' if spacing else 'not supplied')
    profiles = {}
    best = np.inf
    mean = lambda q: np.mean(q, axis=(0,2), keepdims=True)
    def add(key, q):
        profiles.setdefault(key, []).append(np.asarray(q).reshape(-1))
    for it in range(nt):
        raw = [q[it].astype(float, copy=False) for q in fields]
        ex = {k:q[it].astype(float, copy=False) for k,q in extras.items()}
        if any(not np.all(np.isfinite(q)) for q in raw + list(ex.values())):
            raise ValueError(f'Nonfinite input in snapshot {it}.')
        if 'rho' in ex and np.any(ex['rho'] <= 0):
            raise ValueError(f'Nonpositive density in snapshot {it}.')
        b = [a*raw[0], raw[1], a*raw[2]]
        mag = np.sqrt(sum(q*q for q in b))
        direction = _ratio(b[1], mag)
        valid = mag > 0
        bm = mean(mag)
        db = [q-mean(q) for q in b]
        add('f_switchback', _ratio(mean((b[1]<0)&valid),mean(valid)))
        add('f_strong', _ratio(mean(direction < -0.5),mean(valid)))
        add('zero_B_fraction', mean(~valid))
        minimum = np.min(np.where(valid,direction,np.inf),axis=(0,2))
        minimum[~np.isfinite(minimum)] = np.nan
        add('min_direction', minimum)
        add('A_perp', _ratio(np.sqrt(mean(db[0]**2+db[2]**2)),np.abs(mean(b[1]))))
        add('Bmag_relative_std', _ratio(np.sqrt(mean((mag-bm)**2)),bm))
        add('mean_By',mean(b[1]))
        add('mean_Bmag',bm)
        if 'p' in ex:
            pmag = 0.5*mag**2
            ptot = ex['p']+pmag
            dp = ex['p']-mean(ex['p'])
            dpm = pmag-mean(pmag)
            add('p_nonpositive_fraction',mean(ex['p'] <= 0))
            add('ptotal_relative_std',_ratio(np.sqrt(mean((dp+dpm)**2)),mean(ptot)))
            add('pressure_corr',_ratio(mean(dp*dpm),np.sqrt(mean(dp**2)*mean(dpm**2))))
            add('pressure_cancellation',_ratio(np.sqrt(mean((dp+dpm)**2)),
                np.sqrt(mean(dp**2))+np.sqrt(mean(dpm**2))))
            for key,q in [('p',ex['p']),('pmag',pmag),('ptotal',ptot)]:
                add('mean_'+key,mean(q))
        if 'rho' in ex:
            rm = mean(ex['rho'])
            add('rho_relative_std', _ratio(np.sqrt(mean((ex['rho']-rm)**2)),rm))
        if 'vy' in ex:
            dvy = ex['vy']-mean(ex['vy'])
            target = -db[1]/np.sqrt(rm)
            cross = mean(dvy*target)
            add('walen_corr_y', _ratio(cross,np.sqrt(mean(dvy**2)*mean(target**2))))
            add('walen_slope_y', _ratio(cross,mean(target**2)))
        if all(k in ex for k in ('vx','vy','vz')):
            v = [ex['vx']/a, ex['vy'], ex['vz']/a]
            dv = [q-mean(q) for q in v]
            dbA = [q/np.sqrt(rm) for q in db]
            ev, eb = mean(sum(q*q for q in dv)),mean(sum(q*q for q in dbA))
            add('sigma_out',_ratio(-2*mean(sum(u*q for u,q in zip(dv,dbA))),ev+eb))
        if 'divB' not in ex and spacing:
            ex['divB'] = estimate_divergence(*raw, metric, *spacing)
        if 'divB' in ex:
            rev = b[1] < 0
            add('divB_rms_nonreversed',np.sqrt(_ratio(mean(ex['divB']**2*(~rev)),mean(~rev))))
            if spacing:
                eps = _ratio(ell*np.abs(ex['divB']),mag)
                good = np.isfinite(eps)
                safe_eps = np.where(good,eps,0)
                add('epsilon_div_rms',np.sqrt(_ratio(mean(safe_eps**2),mean(good))))
                for key,mask in [('reversed',rev),('nonreversed',~rev)]:
                    mask = mask & good
                    add('epsilon_div_rms_'+key,np.sqrt(_ratio(mean(safe_eps**2*mask),mean(mask))))
            add('divB_rms' , np.sqrt(mean(ex['divB']**2)))
            add('divB_max_abs', np.max(np.abs(ex['divB']),axis=(0,2)))
            add('divB_rms_reversed', np.sqrt(_ratio(mean(ex['divB']**2*(b[1]<0)),mean(b[1]<0))))
        if event is None:
            m = event_y_margin
            candidate = np.where(valid,direction,np.inf).copy()
            if m:
                candidate[:, :m, :] = candidate[:, -m:, :] = np.inf
            loc = np.unravel_index(np.argmin(candidate),spatial)
            score = candidate[loc]
            select = score < best
        else:
            select = it == event[0]
            loc = tuple(event[1:])
            score = direction[loc]
        if select:
            best = score
            ix, iy, iz = loc
            cut = (slice(None),iy,iz)
            e = {'indices':(it,ix,iy,iz), 'time':t[it], 'y':yc[iy],
                 'x_index':np.arange(spatial[0])+mxg,
                 'b_y':direction[cut].copy(),
                 'Bmag_relative':_ratio(mag[cut],bm[0,iy,0]),
                 'is_reversal':bool(b[1][loc]<0)}
            if 'rho' in ex:
                e['rho_relative'] = ex['rho'][cut]/rm[0,iy,0]
            if 'vy' in ex:
                e['delta_vy'] = dvy[cut].copy()
                e['walen_target'] = target[cut].copy()
            if 'p' in ex:
                for key,q in [('p',ex['p']),('pmag',pmag),('ptotal',ptot)]:
                    e[key] = q[cut].copy()
                    e['delta_'+key] = (q-mean(q))[cut].copy()
                e['mean_ptotal'] = float(mean(ptot)[0,iy,0])
            if 'divB' in ex and spacing:
                e['epsilon_div'] = eps[cut].copy()
            if 'divB' in ex:
                e['divB'] = ex['divB'][cut].copy()
            result['event'] = e
    result.update({k:np.stack(v) for k,v in profiles.items()})
    return result


def plot_summary(r, save=None):
    """Heatmaps; optional save='/path/summary.png'. Returns figure."""
    panels = [('f_switchback','Fraction with By < 0',0,None),
              ('f_strong','Fraction with By/|B| < -0.5',0,None),
              ('min_direction','Minimum By/|B|',-1,1),
              ('A_perp','RMS transverse fluctuation / |mean By|',0,None),
              ('Bmag_relative_std','std(|B|) / mean(|B|)',0,None)]
    for k,label,lo,hi in [('rho_relative_std','std(rho) / mean(rho)',0,None),
                          ('walen_corr_y','Radial Walen correlation',-1,1),
                          ('sigma_out','Outward cross helicity',-1,1),
                          ('divB_rms','RMS Div(B), native units',0,None)]:
        if k in r:
            panels.append((k,label,lo,hi))
    fig, axes = plt.subplots((len(panels)+1)//2,2,figsize=(13,3*((len(panels)+1)//2)),squeeze=False,layout='constrained')
    for ax,(key,label,lo,hi) in zip(axes.flat,panels):
        mesh=ax.pcolormesh(r['y'],r['time'],r[key],shading='auto',vmin=lo,vmax=hi,
                           cmap='RdBu_r' if lo == -1 else 'viridis')
        ax.set(xlabel=r['y_label'],ylabel=r['time_label'],title=label)
        fig.colorbar(mesh,ax=ax)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    if save:
        fig.savefig(save,dpi=160)
    return fig


def plot_event(r, save=None):
    """X cut at the selected time,y,z; shaded cells have negative By."""
    if 'event' not in r:
        raise ValueError('No nonzero magnetic field available for event selection.')
    e=r['event']; x=e['x_index']
    n=2+sum(k in e for k in ('delta_vy','rho_relative','divB'))
    fig,axes=plt.subplots(n,1,figsize=(10,2.1*n),sharex=True,layout='constrained')
    axes[0].plot(x,e['b_y']); axes[0].axhline(0,color='k',lw=.8)
    axes[0].set(ylabel='By / |B|',ylim=(-1.05,1.05))
    axes[1].plot(x,e['Bmag_relative']); axes[1].axhline(1,color='k',lw=.8)
    axes[1].set_ylabel('|B| / slice mean')
    j=2
    if 'delta_vy' in e:
        axes[j].plot(x,e['delta_vy'],label='delta vy')
        axes[j].plot(x,e['walen_target'],'--',label='-delta By / sqrt(mean rho)')
        axes[j].legend(); axes[j].set_ylabel('Velocity'); j+=1
    for key,label in [('rho_relative','rho / slice mean'),('divB','Div(B)')]:
        if key in e:
            axes[j].plot(x,e[key]); axes[j].set_ylabel(label); j+=1
    for ax in axes:
        ax.fill_between(x,0,1,where=e['b_y']<0,transform=ax.get_xaxis_transform(),color='orange',alpha=.18)
        ax.axvline(x[e['indices'][1]],color='gray',ls=':',lw=.8)
    axes[-1].set_xlabel('X index (original array)')
    status='candidate reversal' if e['is_reversal'] else 'no reversal at selected point'
    fig.suptitle(f"{status}; {r['time_label']}={e['time']:g}, y={e['y']:g}, trimmed z={e['indices'][3]}")
    if save:
        fig.savefig(save,dpi=160)
    return fig


def save_profiles(r, filename='switchback_profiles.npz'):
    """Save numerical profiles and selected cut without pickled Python objects."""
    payload={k:v for k,v in r.items() if k != 'event'}
    payload.update({'event_'+k:v for k,v in r.get('event',{}).items()})
    np.savez_compressed(Path(filename),**payload)



def estimate_divergence(Bx, By, Bz, J, dx, dy, dz):
    """One trimmed snapshot; uniform logical grid, periodic X,Z, open Y.

    J=a(y)**2, contravariant B. Inputs must broadcast to (x,y,z).
    Derivative uses retained cells only; Y endpoints have one-sided stencils.
    Prefer a model-dumped Div(B) when checking the solver's own constraint.
    """
    fx,fy,fz = (J*q for q in (Bx,By,Bz))
    return ((np.roll(fx,-1,axis=0)-np.roll(fx,1,axis=0))/(2*dx)
            +np.gradient(fy,dy,axis=1,edge_order=2)
            +(np.roll(fz,-1,axis=2)-np.roll(fz,1,axis=2))/(2*dz))/J


def plot_checks(r, save=None):
    """Pressure compensation and divergence heatmaps; NaN = undefined.

    pressure_cancellation = rms(delta p + delta pmag) /
        [rms(delta p)+rms(delta pmag)]. Near zero indicates compensation.
    Scalar total-pressure uniformity alone does not imply force balance:
    magnetic tension and inertia are not included in this check.
    Conditional divergence uses only reversed/nonreversed cells in each slice.
    Empty populations return NaN; compare absolute values as well as contrasts.
    """
    panels=[]
    for key,label,lo,hi in [
        ('ptotal_relative_std','std(total pressure) / mean(total pressure)',0,None),
        ('pressure_corr','Correlation: thermal vs magnetic pressure',-1,1),
        ('pressure_cancellation','Pressure cancellation ratio (0 = compensation)',0,1),
        ('divB_rms','RMS Div(B): all cells',0,None),
        ('divB_rms_reversed','RMS Div(B): reversed cells',0,None),
        ('divB_rms_nonreversed','RMS Div(B): nonreversed cells',0,None),
        ('epsilon_div_rms','RMS cell-scale divergence error: all cells',0,None),
        ('epsilon_div_rms_reversed','RMS cell-scale divergence error: reversed cells',0,None)]:
        if key in r:
            panels.append((key,label,lo,hi))
    if not panels:
        raise ValueError('Supply p and/or divB (or dx,dy,dz) to analyze first.')
    fig,axes=plt.subplots((len(panels)+1)//2,2,figsize=(13,3*((len(panels)+1)//2)),
                          squeeze=False,layout='constrained')
    rawkeys=['divB_rms','divB_rms_reversed','divB_rms_nonreversed']
    rawmax=max((np.max(r[k][np.isfinite(r[k])]) for k in rawkeys
                if k in r and np.any(np.isfinite(r[k]))),default=1.)
    epskeys=['epsilon_div_rms','epsilon_div_rms_reversed']
    epsmax=max((np.max(r[k][np.isfinite(r[k])]) for k in epskeys
                if k in r and np.any(np.isfinite(r[k]))),default=1.)
    for ax,(key,label,lo,hi) in zip(axes.flat,panels):
        if key in rawkeys: hi=rawmax or 1.
        if key in epskeys: hi=epsmax or 1.
        mesh=ax.pcolormesh(r['y'],r['time'],r[key],shading='auto',vmin=lo,vmax=hi,
                           cmap='RdBu_r' if lo == -1 else 'viridis')
        ax.set(title=label,xlabel=r['y_label'],ylabel=r['time_label'])
        fig.colorbar(mesh,ax=ax)
    for ax in list(axes.flat)[len(panels):]: ax.set_visible(False)
    fig.suptitle('Pressure and divergence checks; Div(B): '+r['divB_source'])
    if save: fig.savefig(save,dpi=160)
    return fig


def plot_event_checks(r, save=None):
    """Pressure and divergence at exactly the event selected by analyze.

    Pressure curves use a COMMON mean-total-pressure denominator so their
    amplitudes and cancellation are comparable. Also show their slice-mean
    subtracted fluctuations. This is an X cut, not a force-balance test.
    """
    e=r.get('event')
    if e is None: raise ValueError('No event available.')
    n=1+2*('p' in e)+('divB' in e)+('epsilon_div' in e)
    if n==1: raise ValueError('Supply p and/or divergence inputs to analyze.')
    fig,axes=plt.subplots(n,1,figsize=(11,2.3*n),sharex=True,layout='constrained')
    x=e['x_index']
    axes[0].plot(x,e['b_y']); axes[0].axhline(0,color='k',lw=.7)
    axes[0].set(ylabel='By / |B|',ylim=(-1.05,1.05))
    j=1
    if 'p' in e:
        for key,label in [('p','thermal p'),('pmag','magnetic B²/2'),('ptotal','total')]:
            axes[j].plot(x,_ratio(e[key],e['mean_ptotal']),label=label)
            axes[j+1].plot(x,_ratio(e['delta_'+key],e['mean_ptotal']),label=label)
        axes[j].set_ylabel('Pressure / mean total')
        axes[j+1].set_ylabel('Fluctuation / mean total')
        axes[j+1].axhline(0,color='k',lw=.7)
        axes[j].legend(); axes[j+1].legend(); j+=2
    for key,label in [('divB','Div(B), native units'),('epsilon_div','min(cell lengths) |DivB| / |B|')]:
        if key in e:
            axes[j].plot(x,e[key]); axes[j].set_ylabel(label)
            axes[j].axhline(0,color='k',lw=.7); j+=1
    for ax in axes:
        ax.fill_between(x,0,1,where=e['b_y']<0,transform=ax.get_xaxis_transform(),color='orange',alpha=.18)
        ax.axvline(x[e['indices'][1]],color='gray',ls=':',lw=.8)
    axes[-1].set_xlabel('X index (original array)')
    fig.suptitle(f"{r['time_label']}={e['time']:g}, y={e['y']:g}, trimmed z={e['indices'][3]}; Div(B): {r['divB_source']}")
    if save: fig.savefig(save,dpi=160)
    return fig


def analyze_directory(path='.', *, prefix='BOUT.dmp', start=0, stop=None,
                      step=1, event_y_margin=10, metric_prefix=None,
                      divb_name=None):
    """Read MPI dumps with boutdata; stream one output time at a time.

    Reads Bx,By,Bz,density,pressure,v_x,v_y,v_z and t_array by exact name.
    collect(xguards=False,yguards=False) removes guards; do not trim again.
    start:stop:step uses ordinary Python time slicing (stop is exclusive).
    metric_prefix: None checks standard and m-prefixed metric dumps agree;
        choose '' or 'm' explicitly if they differ, after checking the model.
    Assumes periodic X,Z, no repeated Z endpoint, and uniform logical spacing.
    Radial coordinate is distance from first retained Y cell (origin unknown).
    Requires numpy, matplotlib, boutdata (and its NetCDF dependencies).
    """
    from boutdata import collect
    def get(name, tind=None):
        return np.asarray(collect(name,path=str(path),prefix=prefix,tind=tind,
                                  xguards=False,yguards=False,strict=True,info=False))
    if step <= 0 or start < 0:
        raise ValueError('start must be nonnegative and step positive.')
    times=get('t_array').reshape(-1)
    indices=np.arange(len(times))[slice(start,stop,step)]
    if not len(indices): raise ValueError('No outputs selected.')
    if metric_prefix not in (None,'','m'):
        raise ValueError("metric_prefix must be None, '' or 'm'.")
    names=['g_11','g_22','g_33','g_12','g_13','g_23','J','dx','dy','dz']
    mp='' if metric_prefix is None else metric_prefix
    def spatial_metric(q, name):
        # BOUT++ scalar/Field2D/Field3D dumps use different storage ranks.
        # Canonicalize by coordinate meaning, not NumPy's trailing-axis rules.
        q = np.asarray(q)
        if q.ndim == 0:
            return q
        if q.ndim == 1:
            return q[None, :, None]  # radial profile
        if q.ndim == 2:
            return q[:, :, None]   # (x,y) -> (x,y,1)
        if q.ndim == 3:
            return q
        raise ValueError(f'{name}: expected a static scalar or spatial metric; got {q.shape}.')

    def metric_equal(left, right, label):
        try:
            return np.allclose(left, right, rtol=1e-7, atol=1e-12)
        except ValueError as exc:
            raise ValueError(f'{label}: incompatible spatial shapes {np.shape(left)} '
                             f'and {np.shape(right)} after aligning X,Y,Z axes.') from exc

    metrics={k:spatial_metric(get(mp+k), mp+k) for k in names}
    if metric_prefix is None:
        # This model dumps both metric sets. Do not silently choose if different.
        for k in names:
            other=spatial_metric(get('m'+k), 'm'+k)
            if not metric_equal(metrics[k],other, k+' versus m'+k):
                raise ValueError(f'{k} and m{k} differ. Check which the model uses, '
                                 "then set metric_prefix='' or 'm' explicitly.")
    g=metrics['g_11']
    for k,expected in [('g_22',1.),('g_33',g),('J',g),
                        ('g_12',0.),('g_13',0.),('g_23',0.)]:
        if not metric_equal(metrics[k],expected,mp+k):
            raise ValueError(f'{mp+k} is incompatible with diag(a²,1,a²), J=a².')
    spacing={}
    for k in ('dx','dy','dz'):
        q=metrics[k]
        value=float(q.flat[0])
        if not np.all(np.isfinite(q)) or value <= 0 or not np.allclose(q,value,rtol=1e-8,atol=0):
            raise ValueError(f'{mp+k} is not a positive uniform logical spacing.')
        spacing[k]=value
    combined=None
    chunks={}
    best=np.inf
    for count,i in enumerate(indices):
        print(f'Analyzing output {i} ({count+1}/{len(indices)}), t={times[i]:g}',flush=True)
        def snapshot(name):
            q=get(name,tind=int(i))
            if q.ndim == 4 and q.shape[0] == 1: q=q[0]
            if q.ndim != 3:
                raise ValueError(f'{name}: expected (x,y,z) snapshot; received {q.shape}.')
            return q
        b=[snapshot(k) for k in ('Bx','By','Bz')]
        ex={arg:snapshot(name) for arg,name in
            [('rho','density'),('p','pressure'),('vx','v_x'),('vy','v_y'),('vz','v_z')]}
        if divb_name is not None: ex['divB']=snapshot(divb_name)
        one=analyze(*b,g,time=[times[i]],y=np.arange(b[0].shape[1])*spacing['dy'],
                    event_y_margin=event_y_margin,**ex,**spacing)
        if combined is None:
            combined={k:v for k,v in one.items()
                      if k != 'event' and not (isinstance(v,np.ndarray) and v.ndim==2)}
        for k,v in one.items():
            if isinstance(v,np.ndarray) and v.ndim==2:
                chunks.setdefault(k,[]).append(v)
        if 'event' in one:
            e=one['event']; score=e['b_y'][e['indices'][1]]
            if score < best:
                best=score
                e['indices']=(count,)+tuple(e['indices'][1:])
                e['source_output_index']=int(i)
                combined['event']=e
    combined.update({k:np.concatenate(v,axis=0) for k,v in chunks.items()})
    combined['time']=times[indices]
    combined['source_output_indices']=indices
    combined['y_label']='Radial distance from first retained cell'
    combined['metric_prefix']=mp
    return combined


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Analyze stationary expanding-tube BOUT++ dumps.')
    parser.add_argument('path',nargs='?',default='.',help='Directory containing BOUT.dmp.*.nc')
    parser.add_argument('--prefix',default='BOUT.dmp')
    parser.add_argument('--out',default='switchback_diagnostics')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--stop',type=int)
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--event-y-margin',type=int,default=10)
    parser.add_argument('--metric-prefix',choices=['','m'],default=None)
    parser.add_argument('--divb-name',default=None,help='Optional name of model-dumped Div(B)')
    args=parser.parse_args()
    # CLI writes figures without requiring a desktop display.
    plt.switch_backend('Agg')
    r=analyze_directory(args.path,prefix=args.prefix,start=args.start,stop=args.stop,
                        step=args.step,event_y_margin=args.event_y_margin,
                        metric_prefix=args.metric_prefix,divb_name=args.divb_name)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    for name,func in [('summary',plot_summary),('event',plot_event),
                      ('pressure_divergence',plot_checks),('event_checks',plot_event_checks)]:
        if name in ('event','event_checks') and 'event' not in r: continue
        fig=func(r,save=out/(name+'.png')); plt.close(fig)
    save_profiles(r,out/'profiles.npz')
    print(f'Saved plots and profiles to {out.resolve()}')
    print('Divergence source:',r['divB_source'])
    print('Python divergence uses one-sided Y endpoint stencils; inspect interior cells separately.')
    if 'event' in r: print('Selected source output:',r['event']['source_output_index'])


if __name__ == '__main__':
    main()
