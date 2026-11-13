#!/usr/bin/env python3
"""Compiled-model regression tests for the optional finite-volume MHD path."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from netCDF4 import Dataset


def input_text(n=64, end=.2, periodic=False, gamma=1.4, rho='1-H(y-pi)*0.875',
               pressure='1-H(y-pi)*0.9', vel=('0','0','0'), mag=('0','0','0'),
               metric=False, limiter='mc', extra='', nz=4, nx=8, ly=1.):
    geom='''
g11=1/geometry:a^2
g22=1
g33=1/geometry:a^2
g_11=geometry:a^2
g_22=1
g_33=geometry:a^2
g_12=0
g_13=0
g_23=0
J=geometry:a^2
''' if metric else ''
    s=f'''nout=1
timestep={end}
MXG=2
MYG=2
periodicX=true
[mesh]
nx={nx}
ny={n}
nz={nz}
dx=2*pi/(nx-4)
dy={ly}/ny
dz=2*pi/nz
ixseps1={'nx' if periodic else '-1'}
ixseps2={'nx' if periodic else '-1'}
symmetricGlobalX=true
symmetricGlobalY=true
{geom}
[mesh:paralleltransform]
type=identity
[solver]
type=rk3ssp
max_timestep=0.0005
[mhd]
shock_capturing=true
g={gamma}
diagnose=true
dump_metric=true
[mhd:shock]
limiter={limiter}
glm_ch=2
glm_damping=1
[all]
scale=1
bndry_all={'none' if periodic else 'neumann'}
[density]
function={rho}
[pressure]
function={pressure}
'''
    for name,value in zip(('v_x','v_y','v_z','Bx','By','Bz'),(*vel,*mag)):
        s+=f'[{name}]\nfunction={value}\n'
    if metric: s+='[geometry]\na=1+0.1*cos(y)\n'
    return s+extra


def read(folder):
    chunks=[]
    keys=('density','pressure','v_x','v_y','v_z','Bx','By','Bz','total_energy',
          'momentum_x','momentum_y','momentum_z','glm_psi','divB','Q_visc','Q_ohm')
    for file in folder.glob('BOUT.dmp.*.nc'):
        with Dataset(file) as d:
            ix,iy=int(d['PE_XIND'][...]),int(d['PE_YIND'][...])
            fields={k:np.asarray(d[k][:,2:-2,2:-2,:]) for k in keys}
            fields['mJ']=np.asarray(d['mJ'][2:-2,2:-2,:])
            fields['t']=np.asarray(d['t_array'][:])
            for k,v in fields.items(): assert np.isfinite(v).all(),(file,k)
            chunks.append((ix,iy,fields))
    rows=[]
    for iy in sorted(set(c[1] for c in chunks)):
        row=sorted((c for c in chunks if c[1]==iy),key=lambda c:c[0])
        rows.append({key:(row[0][2][key] if key=='t' else
                    np.concatenate([c[2][key] for c in row],axis=0 if key=='mJ' else 1))
                     for key in row[0][2]})
    return {key:(rows[0][key] if key=='t' else
                 np.concatenate([r[key] for r in rows],axis=1 if key=='mJ' else 2))
            for key in rows[0]}



def run(exe, root, label, inp, ranks=1, args=(), fail=None, nxpe=1):
    folder=root/label
    folder.mkdir(exist_ok=True)
    (folder/'BOUT.inp').write_text(inp)
    cmd=[str(exe),'-d',str(folder),f'NXPE={nxpe}',*args]
    if ranks>1: cmd=['mpirun','-np',str(ranks),*cmd]
    with (folder/'stdout.txt').open('w') as log:
        proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    if fail:
        assert proc.returncode and fail in (folder/'stdout.txt').read_text(),folder
        return None
    if proc.returncode: raise RuntimeError(f'{label}: see {folder}/stdout.txt')
    return read(folder)


def sod_exact(y,t,g=1.4):
    # Exact Riemann solution: left rarefaction, contact, right shock.
    rl,pl,rr,pr=1.,1.,.125,.1
    cl=np.sqrt(g*pl/rl); cr=np.sqrt(g*pr/rr)
    def f(p,r,pk,c):
        if p>pk: return (p-pk)*np.sqrt(2/((g+1)*r)/(p+(g-1)/(g+1)*pk))
        return 2*c/(g-1)*((p/pk)**((g-1)/(2*g))-1)
    lo,hi=pr,pl
    for _ in range(80):
        pm=.5*(lo+hi)
        if f(pm,rl,pl,cl)+f(pm,rr,pr,cr)>0: hi=pm
        else: lo=pm
    ps=.5*(lo+hi); us=.5*(f(ps,rr,pr,cr)-f(ps,rl,pl,cl))
    cs=cl*(ps/pl)**((g-1)/(2*g))
    sr=cr*np.sqrt((g+1)/(2*g)*ps/pr+(g-1)/(2*g))
    xi=(y-.5)/t
    r=np.full_like(y,rr); p=np.full_like(y,pr); u=np.zeros_like(y)
    left=xi<=-cl; fan=(xi>-cl)&(xi<us-cs); ls=(xi>=us-cs)&(xi<=us); rs=(xi>us)&(xi<sr)
    r[left]=rl;p[left]=pl
    cf=2/(g+1)*(cl-(g-1)/2*xi[fan])
    r[fan]=rl*(cf/cl)**(2/(g-1));p[fan]=pl*(cf/cl)**(2*g/(g-1));u[fan]=2/(g+1)*(cl+xi[fan])
    r[ls]=rl*(ps/pl)**(1/g);p[ls]=ps;u[ls]=us
    ratio=ps/pr
    r[rs]=rr*(ratio+(g-1)/(g+1))/((g-1)/(g+1)*ratio+1);p[rs]=ps;u[rs]=us
    return r,p,u


def budget(d,key):
    vals=np.sum(d[key]*d['mJ'][None,...],axis=(1,2,3))
    return float(np.max(np.abs(vals-vals[0]))/max(abs(vals[0]),1.))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--exe',type=Path,default=Path('build/mhd3d'));ap.add_argument('--workdir',type=Path,required=True)
    a=ap.parse_args();a.workdir.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='mhd-shock-',dir=a.workdir.resolve()));exe=a.exe.resolve();results={}
    print(root,flush=True)
    errors=[]
    for n in (64,128):
        d=run(exe,root,f'sod{n}',input_text(n=n))
        exact=sod_exact((np.arange(n)+.5)/n,.2)
        err=float(np.mean(np.abs(d['density'][-1,0,:,0]-exact[0])));errors.append(err)
        assert np.min(d['pressure'])>0 and np.min(d['density'])>0
        assert budget(d,'density')<1e-10 and budget(d,'total_energy')<1e-10
        results[f'sod{n}']={'rho_L1':err,'energy_error':budget(d,'total_energy')}
        print(f'sod{n}: {results[f"sod{n}"]}',flush=True)
    assert errors[1]<.8*errors[0] and errors[1]<.025
    d=run(exe,root,'brio_wu',input_text(n=128,end=.1,gamma=2,mag=('1-2*H(y-pi)','.75','0')))
    assert np.min(d['pressure'])>0 and np.min(d['density'])>0
    assert np.max(abs(d['By']-.75))<1e-12
    assert budget(d,'density')<1e-9 and budget(d,'total_energy')<1e-9
    results['brio_wu']={'min_pressure':float(d['pressure'].min()),'energy_error':budget(d,'total_energy')}
    print('Brio-Wu passed',flush=True)
    for lim in ('mc','minmod','first_order'):
        d=run(exe,root,'rest_'+lim,input_text(n=16,end=.01,periodic=True,rho='1',pressure='1',metric=True,limiter=lim,ly=2*np.pi))
        err=max(float(np.max(abs(d[k][-1]-d[k][0]))) for k in ('density','pressure','v_y','total_energy'))
        assert err<1e-12,err
        results['rest_'+lim]=err
    print('Warped-metric pressure balance passed',flush=True)
    wave_errors=[]
    for n in (32,64):
        d=run(exe,root,f'alfven{n}',input_text(n=n,end=.2,periodic=True,rho='1',pressure='1',
            vel=('-.1*cos(y)','0','-.1*sin(y)'),mag=('.1*cos(y)','1','.1*sin(y)'),ly=2*np.pi))
        y=(np.arange(n)+.5)*2*np.pi/n
        error=float(np.mean(abs(d['Bx'][-1,0,:,0]-.1*np.cos(y-.2))))
        wave_errors.append(error)
        assert budget(d,'total_energy')<1e-12
    assert wave_errors[0]/wave_errors[1]>2.5,wave_errors
    results['alfven_L1_errors']=wave_errors
    d=run(exe,root,'glm',input_text(n=64,end=.3,periodic=True,rho='1',pressure='1',mag=('0','.001*sin(y)','0'),ly=2*np.pi))
    amplitude=np.sqrt(np.mean(d['By'][-1]**2)/np.mean(d['By'][0]**2))
    w=np.sqrt(4-.25);expected=np.exp(-.15)*(np.cos(w*.3)+.5/w*np.sin(w*.3))
    assert abs(amplitude-expected)<.005,(amplitude,expected)
    assert np.max(abs(d['glm_psi'][-1]))>1e-4
    results['glm_amplitude']={'measured':float(amplitude),'exact_linear':float(expected)}
    # Primitive time-dependent boundary values must be evaluated at output time.
    driven=input_text(n=16,end=.01,rho='1',pressure='1',vel=('-.01*cos(y)','4','-.01*sin(y)'),
        mag=('.01*cos(y)','1','.01*sin(y)'),ly=2*np.pi,metric=True).replace('a=1+0.1*cos(y)','a=1+0.1*y/(2*pi)')
    driven=driven.replace('[v_x]','[v_x]\nbndry_ydown=dirichlet(-.01*cos(5*t))')
    driven=driven.replace('[Bx]','[Bx]\nbndry_ydown=dirichlet(.01*cos(5*t))')
    run(exe,root,'driven',driven)
    with Dataset(root/'driven'/'BOUT.dmp.0.nc') as f:
        face=.5*(f['v_x'][-1,2:-2,1,:]+f['v_x'][-1,2:-2,2,:])
        assert np.max(abs(face+.01*np.cos(.05)))<1e-12
    results['time_dependent_inlet']='passed'
    print('Alfven convergence, GLM and driven inlet passed',flush=True)
    inp=input_text(n=16,end=.01,periodic=True,rho='1+0.05*cos(2*pi*x)*cos(y)*cos(z)',pressure='1',
                   vel=('.1*sin(z)','.1*sin(2*pi*x)','.1*sin(y)'),mag=('.1*sin(y)','.1*sin(z)','1+.1*sin(2*pi*x)'),nx=12,nz=8,ly=2*np.pi)
    d=run(exe,root,'three_d',inp)
    e=run(exe,root,'three_d_mpi',inp,ranks=2)
    delta=max(float(np.max(abs(d[k]-e[k]))) for k in d if k not in ('t','mJ'))
    assert delta<1e-12,delta
    assert budget(d,'density')<1e-12 and budget(d,'total_energy')<1e-12
    results['three_d']={'mpi_difference':delta,'mass_error':budget(d,'density'),'energy_error':budget(d,'total_energy')}
    ex=run(exe,root,'three_d_mpi_x',inp,ranks=2,nxpe=2)
    delta_x=max(float(np.max(abs(d[k]-ex[k]))) for k in d if k not in ('t','mJ'))
    assert delta_x<1e-12,delta_x
    results['three_d']['mpi_x_difference']=delta_x
    curved=input_text(n=16,end=.01,periodic=True,rho='1+0.05*cos(2*pi*x)*cos(y)*cos(z)',pressure='1',
        vel=('.1*sin(z)','.1*sin(2*pi*x)','.1*sin(y)'),mag=('.1*sin(y)','.1*sin(z)','1+.1*sin(2*pi*x)'),
        nx=12,nz=8,ly=2*np.pi,metric=True).replace('diagnose=true','include_viscos=true\nviscos=.01\nbulk_viscos=.005\neta=.01\ndiagnose=true')
    dc=run(exe,root,'curved_dissipation',curved)
    ec=run(exe,root,'curved_dissipation_mpi',curved,ranks=2)
    delta_c=max(float(np.max(abs(dc[k]-ec[k]))) for k in dc if k not in ('t','mJ'))
    assert delta_c<1e-12,delta_c
    assert budget(dc,'density')<1e-12 and budget(dc,'total_energy')<1e-10
    assert dc['Q_visc'].max()>0 and dc['Q_ohm'].max()>0
    results['curved_dissipation']={'mpi_difference':delta_c,'energy_error':budget(dc,'total_energy')}
    # A short restart must agree with uninterrupted integration using the same fixed dt.
    r=run(exe,root,'restart',inp.replace('nout=1','nout=2'))
    run(exe,root,'restart_split',inp)
    q=run(exe,root,'restart_split',inp,args=('restart=true',))
    delta=max(float(np.max(abs(r[k][-1]-q[k][-1]))) for k in ('density','total_energy','Bx','By','Bz'))
    assert delta<1e-12,delta
    results['restart_difference']=delta
    run(exe,root,'restart_split',inp,args=('restart=true','mhd:shock_capturing=false'),fail='Restart formulation mismatch')
    run(exe,root,'bad_solver',inp.replace('type=rk3ssp','type=cvode'),fail='requires [solver] type=rk3ssp')
    run(exe,root,'bad_geometry',inp.replace('dx=2*pi/(nx-4)','g12=0.1\ndx=2*pi/(nx-4)'),fail='Shock capturing supports')
    run(exe,root,'negative_pressure',inp.replace('function=1\n[v_x]','function=-1\n[v_x]'),fail='Nonpositive/nonfinite conservative MHD cell state')
    auto=input_text(n=32,end=.02).replace('max_timestep=0.0005\n','')
    ad=run(exe,root,'automatic_cfl',auto)
    assert ad['pressure'].min()>0
    assert 'dt = 2.000000e-02' not in (root/'automatic_cfl'/'stdout.txt').read_text()
    results['automatic_initial_cfl_cap']='passed'
    from test_dissipation import run as run_primitive
    primitive=inp.replace('shock_capturing=true','shock_capturing=false').replace('type=rk3ssp','type=cvode').replace('max_timestep=0.0005','atol=1e-10\nrtol=1e-8')
    run_primitive(exe,root,'switch_disabled',primitive)
    results['disabled_switch_with_shock_options']='passed'
    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)
    print('All shock tests passed: '+str(root),flush=True)

if __name__=='__main__': main()
