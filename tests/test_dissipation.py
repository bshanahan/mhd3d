#!/usr/bin/env python3
"""Analytic tests of the compiled model. Requires numpy, netCDF4 and mpirun."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from netCDF4 import Dataset


def input_text(n=32, eta=0., nu=0., bulk=0., metric=False,
               vel=('0','0','0'), mag=('0','0','0'), density='2', end=.1):
    geom = '''
g11=1/geometry:a^2
g22=1
g33=1/geometry:a^2
g12=0
g13=0
g23=0
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
nx=12
ny={n if metric else 8}
nz={16 if metric else n}
dx=2*pi/(nx-4)
dy=2*pi/ny
dz=2*pi/nz
ixseps1=nx
ixseps2=nx
symmetricGlobalY=true
{geom}
[mesh:paralleltransform]
type=identity
[mesh:ddx]
first=C2
[mesh:ddy]
first=C2
[mesh:ddz]
first=C2
[solver]
type=cvode
atol=1e-11
rtol=1e-9
mxstep=10000
[mhd]
g=5/3
eta={eta}
include_viscos=true
viscos={nu}
bulk_viscos={bulk}
diagnose=true
dump_metric=true
[all]
scale=1
bndry_all=none
[density]
function={density}
[pressure]
function=1
'''
    for name,value in zip(('v_x','v_y','v_z','Bx','By','Bz'),(*vel,*mag)):
        s+=f'[{name}]\nfunction={value}\n'
    if metric: s+='[geometry]\na=1+0.1*cos(y)\n'
    return s


def run(exe, root, label, inp, ranks=1):
    folder=root/label
    folder.mkdir()
    (folder/'BOUT.inp').write_text(inp)
    cmd=[str(exe),'-d',str(folder),'NXPE=1']
    if ranks>1: cmd=['mpirun','-np',str(ranks),*cmd]
    with (folder/'stdout.txt').open('w') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if p.returncode:
        raise RuntimeError(f'{label}: exit={p.returncode}; see {folder}/stdout.txt')
    chunks=[]
    for file in folder.glob('BOUT.dmp.*.nc'):
        with Dataset(file) as d:
            ix,iy=int(d['PE_XIND'][...]),int(d['PE_YIND'][...])
            assert ix==0
            fields={}
            for key in ('density','pressure','v_x','v_y','v_z','Bx','By','Bz',
                        'energy_density','Q_visc','Q_ohm','viscous_work',
                        'resistive_magnetic_work','viscous_force_x',
                        'viscous_force_y','viscous_force_z','divB'):
                fields[key]=np.asarray(d[key][:,2:-2,2:-2,:])
                assert np.isfinite(fields[key]).all(),(label,key,'nonfinite')
            fields['t']=np.asarray(d['t_array'][:])
            chunks.append((iy,fields))
    chunks.sort(key=lambda c:c[0])
    assert len(chunks)==ranks
    return {key:(chunks[0][1][key] if key=='t' else
                 np.concatenate([c[1][key] for c in chunks],axis=2))
            for key in chunks[0][1]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--exe',type=Path,default=Path('build/mhd3d'))
    ap.add_argument('--workdir',type=Path,required=True)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='mhd-dissipation-',dir=args.workdir.resolve()))
    exe=args.exe.resolve()
    results={}
    for kind in ('ohmic','viscous'):
        errors=[]
        for n in (32,64):
            kwargs = dict(eta=.1,mag=('cos(z)','sin(z)','0'),density='1') if kind=='ohmic' else dict(nu=.1,vel=('cos(z)','sin(z)','0'))
            d=run(exe,root,f'{kind}{n}',input_text(n=n,**kwargs))
            target='Bx' if kind=='ohmic' else 'v_x'
            amplitude=np.sqrt(np.mean(d[target][-1]**2)/np.mean(d[target][0]**2))
            t=d['t'][-1]; kd=np.sin(2*np.pi/n)/(2*np.pi/n)
            assert abs(amplitude-np.exp(-.1*kd**2*t))<5e-8
            rho=1 if kind=='ohmic' else 2
            pressure=1+(2/3)*rho*.5*(1-amplitude**2)
            assert np.max(abs(d['pressure'][-1]-pressure))<5e-8
            energy_error=np.max(abs(d['energy_density'][-1]-d['energy_density'][0]))
            assert energy_error<8e-8
            heat='Q_ohm' if kind=='ohmic' else 'Q_visc'
            work='resistive_magnetic_work' if kind=='ohmic' else 'viscous_work'
            assert np.min(d[heat])>=-1e-13
            assert abs(np.mean(d[heat][0]+d[work][0]))<2e-12
            errors.append(abs(amplitude-np.exp(-.1*t)))
            results[f'{kind}{n}']={'energy_error':float(energy_error),'continuum_amplitude_error':float(errors[-1])}
        assert errors[0]/errors[1]>3.8

    d=run(exe,root,'killing',input_text(metric=True,nu=.1,bulk=.2,
                  vel=('geometry:a^2','0','0'),end=1e-6))
    assert np.max(abs(d['Q_visc'][0]))<1e-24
    assert np.max(abs(d['viscous_force_x'][0]))<1e-12
    results['killing_max_heat']=float(np.max(abs(d['Q_visc'][0])))

    errors=[]
    for n in (32,64):
        inp=input_text(n=n,metric=True,nu=.1,
                       vel=('geometry:a^2*sin(y)','0','0'),end=1e-6)
        d=run(exe,root,f'curved{n}',inp)
        y=(np.arange(n)+.5)*2*np.pi/n
        a=1+.1*np.cos(y); ap=-.1*np.sin(y)
        expected=.2*(-a*a*np.sin(y)+4*a*ap*np.cos(y))
        actual=d['viscous_force_x'][0].mean(axis=(0,2))
        errors.append(np.sqrt(np.mean((actual-expected)**2)))
        balance=np.mean((d['Q_visc'][0]+d['viscous_work'][0])*(a*a)[None,:,None])
        assert abs(balance)<2e-12
        kd=np.sin(2*np.pi/n)/(2*np.pi/n)
        expected_heat=.2*a*a*(kd*np.cos(y))**2
        assert np.max(abs(d['Q_visc'][0].mean(axis=(0,2))-expected_heat))<2e-12
        assert np.min(d['Q_visc'][0])>=-1e-13
        results[f'curved{n}']={'force_L2_error':float(errors[-1]),'energy_exchange_residual':float(balance)}
        if n==32:
            parallel=run(exe,root,'curved32_mpi2',inp,ranks=2)
            for key in ('Q_visc','viscous_force_x','viscous_work'):
                assert np.max(abs(parallel[key][0]-d[key][0]))<2e-12
    assert errors[0]/errors[1]>3.8

    n=32
    d=run(exe,root,'bulk',input_text(n=n,nu=.1,bulk=.2,
              vel=('0','0','sin(z)'),density='2+0.2*cos(z)',end=1e-6))
    z=np.arange(n)*2*np.pi/n
    kd=np.sin(2*np.pi/n)/(2*np.pi/n)
    expected=(2+.2*np.cos(z))*(4*.1/3+.2)*(kd*np.cos(z))**2
    assert np.max(abs(d['Q_visc'][0].mean(axis=(0,1))-expected))<2e-12
    assert abs(np.mean(d['Q_visc'][0]+d['viscous_work'][0]))<2e-12
    results['bulk_max_heat_error']=float(np.max(abs(d['Q_visc'][0].mean(axis=(0,1))-expected)))

    d=run(exe,root,'curved_ohmic',input_text(metric=True,eta=.1,
       mag=('cos(y)/geometry:a^2','0','sin(y)/geometry:a^2'),end=1e-6))
    y=(np.arange(32)+.5)*2*np.pi/32; j=(1+.1*np.cos(y))**2
    balance=np.mean((d['Q_ohm'][0]+d['resistive_magnetic_work'][0])*j[None,:,None])
    assert abs(balance)<2e-12
    assert np.max(abs(d['divB'][0]))<2e-12
    results['curved_ohmic_energy_exchange_residual']=float(balance)

    # Constant nonorthogonal coordinate transform X=x+0.3*y: test index raising.
    inp=input_text(n=32,nu=.1,vel=('cos(z)','0.3*cos(z)+sin(z)','0'))
    inp=inp.replace('[mesh:paralleltransform]', """g11=1.09
g22=1
g33=1
g12=-0.3
g13=0
g23=0
g_11=1
g_22=1.09
g_33=1
g_12=0.3
g_13=0
g_23=0
J=1
[mesh:paralleltransform]""")
    d=run(exe,root,'nonorthogonal',inp)
    expected_heat=.2*(np.sin(2*np.pi/32)/(2*np.pi/32))**2
    assert np.max(abs(d['Q_visc'][0]-expected_heat))<2e-12
    assert np.max(abs(d['energy_density'][-1]-d['energy_density'][0]))<8e-8
    results['nonorthogonal_energy_error']=float(np.max(abs(d['energy_density'][-1]-d['energy_density'][0])))

    # Short physical-boundary exercise. No claim that this state is steady.
    inp=input_text(metric=True,nu=.01,eta=.01,
        vel=('0.1*geometry:a^2*sin(z)','1','0'),
        mag=('0.1*cos(z)','1/geometry:a^2','0'),end=1e-5)
    inp=inp.replace('ixseps1=nx','ixseps1=-1').replace('ixseps2=nx','ixseps2=-1')
    inp=inp.replace('bndry_all=none','bndry_all=neumann').replace('a=1+0.1*cos(y)','a=1+0.2*y/(2*pi)')
    one=run(exe,root,'open_tube',inp)
    two=run(exe,root,'open_tube_mpi2',inp,ranks=2)
    err=max(float(np.max(abs(one[k]-two[k]))) for k in ('pressure','v_x','v_y','Bx','By','Q_visc','Q_ohm'))
    assert err<5e-8,err
    results['open_tube_mpi_max_difference']=err

    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    print('PASS; artifacts:',root)

if __name__=='__main__': main()
