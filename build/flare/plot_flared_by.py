"""Interactive Plotly view of By in a stationary expanding tube.

Install:
    python -m pip install numpy plotly scikit-image boutdata

CLI:
    python plot_flared_by.py . --index 44 --out flared_By.html

Notebook:
    fig = plot_flared_by('.', time_index=44)
    fig.show()

The plot shows one top-down central X-Y plane. Blue represents negative By;
red represents positive By. Gold surfaces mark By=0 in the 3-D volume.

Geometry:
    X = a(y)*x, Y = y-y_first, Z = a(y)*z, a = sqrt(g_11).

This is an illustrative embedding, not an exact isometric embedding of
BOUT++'s diagonal metric. Field colors represent the model's radial
component By, not a transformed Cartesian vector component.

Requires uniform logical dx,dy,dz; g_22=1, g_33=g_11, and zero
off-diagonal metric.
"""

import argparse

import numpy as np
import plotly.graph_objects as go


# For cmin=-limit and cmax=limit: negative values are blue, positive red.
COLORS = [
    [0.00, '#12658e'],
    [0.25, '#66bdd2'],
    [0.50, '#eef2f5'],
    [0.75, '#ef7f91'],
    [1.00, '#b42146'],
]


def make_figure(By, a_y, dx, dy, dz, *, time_label='', normalize=False,
                stride=1, slices=(0.12, 0.52, 0.90), longitudinal=True,
                reversals=True, clim=None, true_aspect=False, dark=False):
    """Make a transparent, top-down plot of one central X-Y plane.

    Gold By=0 surfaces are extracted from the full 3-D field. The legacy
    slices and longitudinal arguments are accepted for compatibility, but
    this view always displays only the central X-Y plane.
    """
    By = np.asarray(By, dtype=float)
    if By.ndim != 3 or min(By.shape) < 3 or not np.all(np.isfinite(By)):
        raise ValueError(
            'By must be a finite (x,y,z) array with >=3 points per axis.'
        )

    nx, ny, nz = By.shape

    a_y = np.asarray(a_y, dtype=float)
    if a_y.shape != (ny,) or np.any(a_y <= 0) or not np.all(np.isfinite(a_y)):
        raise ValueError('a_y must be a positive finite array of length ny.')

    if any(np.ndim(q) != 0 or not np.isfinite(q) or q <= 0
           for q in (dx, dy, dz)):
        raise ValueError('dx,dy,dz must be positive scalar logical spacings.')

    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError('stride must be a positive integer.')

    # Center X and Z; let Y start at zero.
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = np.arange(ny) * dy
    z = (np.arange(nz) - (nz - 1) / 2) * dz

    # Obtain the plane at z=0. For an even number of Z points, interpolate
    # between the two planes that bracket zero.
    k_hi = int(np.searchsorted(z, 0.0))
    if k_hi < nz and np.isclose(z[k_hi], 0.0):
        plane = By[:, :, k_hi]
    else:
        k_lo = k_hi - 1
        weight = (0.0 - z[k_lo]) / (z[k_hi] - z[k_lo])
        plane = (
            (1.0 - weight) * By[:, :, k_lo]
            + weight * By[:, :, k_hi]
        )

    if normalize:
        ref = np.abs(np.mean(By, axis=(0, 2)))
        threshold = (
            np.finfo(float).eps * max(1.0, float(np.max(np.abs(By))))
        )
        if np.any(ref <= threshold):
            raise ValueError(
                'Slice-mean By is too close to zero for normalization.'
            )
        plane = plane / ref[None, :]
        volume_for_contours = By / ref[None, :, None]
        color_title = 'By / |⟨By⟩xz|'
    else:
        volume_for_contours = By
        color_title = 'By'

    # Decimate rendered data while retaining the endpoints.
    ix = np.unique(np.r_[np.arange(0, nx, stride), nx - 1])
    iy = np.unique(np.r_[np.arange(0, ny, stride), ny - 1])
    iz = np.unique(np.r_[np.arange(0, nz, stride), nz - 1])

    xs = x[ix]
    ys = y[iy]
    zs = z[iz]
    a = a_y[iy]
    c = plane[np.ix_(ix, iy)]

    limit = float(np.max(np.abs(c))) if clim is None else float(clim)
    if clim is None and limit == 0:
        limit = 1.0
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('clim must be positive.')

    transparent = 'rgba(0,0,0,0)'
    foreground = '#e4edf5' if dark else '#233345'

    fig = go.Figure()

    # One warped X-Y field plane, viewed from directly above.
    X = xs[:, None] * a[None, :]
    Y = np.broadcast_to(ys[None, :], X.shape)
    Z = np.zeros_like(X)

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=c,
        coloraxis='coloraxis',
        name='Central X–Y plane',
        showlegend=False,
        opacity=0.86,
        lighting=dict(
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
            roughness=1.0,
        ),
        hovertemplate=(
            'X=%{x:.3g}<br>'
            'Y=%{y:.3g}<br>'
            + color_title
            + '=%{surfacecolor:.4g}<extra></extra>'
        ),
    ))

    # Extract the gold By=0 surfaces from the decimated 3-D field.
    contour_data = volume_for_contours[np.ix_(ix, iy, iz)]
    if reversals and np.min(contour_data) < 0 < np.max(contour_data):
        from skimage.measure import marching_cubes

        verts, faces, _, _ = marching_cubes(
            contour_data,
            level=0,
            allow_degenerate=False,
        )

        # Marching-cubes vertices are in array-index coordinates. Map them
        # back to logical coordinates, then apply the flared-tube embedding.
        xv = np.interp(verts[:, 0], np.arange(len(xs)), xs)
        yv = np.interp(verts[:, 1], np.arange(len(ys)), ys)
        zv = np.interp(verts[:, 2], np.arange(len(zs)), zs)
        av = np.interp(yv, ys, a)

        fig.add_trace(go.Mesh3d(
            x=av * xv,
            y=yv,
            z=av * zv,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name='By = 0 surfaces',
            color='#f6bf64',
            opacity=0.78,
            showlegend=True,
            lighting=dict(
                ambient=0.8,
                diffuse=0.45,
                specular=0.2,
                roughness=0.8,
            ),
            hovertemplate=(
                'By = 0<br>'
                'X=%{x:.3g}<br>'
                'Y=%{y:.3g}<br>'
                'Z=%{z:.3g}<extra></extra>'
            ),
        ))

    fig.update_layout(
        template='plotly_dark' if dark else 'plotly_white',
        paper_bgcolor=transparent,
        plot_bgcolor=transparent,
        font=dict(family='Arial', color=foreground, size=13),
        title=dict(
            text=(
                f'FLARED WIND · {color_title}'
                f'<br><sup>{time_label} · blue: By &lt; 0 · '
                'red: By &gt; 0 · gold: By = 0</sup>'
            ),
            x=0.04,
        ),
        scene=dict(
            bgcolor=transparent,
            xaxis=dict(
                title='X',
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showline=False,
            ),
            yaxis=dict(
                title='Radial Y',
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showline=False,
            ),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=0, y=0, z=2.5),
                up=dict(x=0, y=1, z=0),
                projection=dict(type='orthographic'),
            ),
            aspectmode='data' if true_aspect else 'manual',
            aspectratio=dict(x=1.15, y=3.2, z=1.15),
        ),
        coloraxis=dict(
            colorscale=COLORS,
            cmin=-limit,
            cmax=limit,
            colorbar=dict(
                title=dict(text=color_title),
                thickness=16,
                len=0.72,
            ),
        ),
        legend=dict(
            x=0.02,
            y=0.03,
            bgcolor=transparent,
            font=dict(size=11),
        ),
        width=1100,
        height=760,
        margin=dict(l=10, r=30, t=85, b=10),
        uirevision='flared-field',
    )
    return fig


def plot_flared_by(path='.', *, time_index=-1, time=None, prefix='BOUT.dmp',
                   metric_prefix='', html=None, **kwargs):
    """Load one MPI-assembled snapshot and make a figure; optionally save HTML.

    `time` selects the nearest saved simulation time and overrides
    `time_index`. `metric_prefix=''` reads g_11 etc.; `'m'` selects mg_11 etc.
    No guard trimming is needed by the caller. The saved HTML includes Plotly
    and works offline.
    """
    from boutdata import collect

    def get(name, tind=None):
        return np.asarray(collect(
            name,
            path=str(path),
            prefix=prefix,
            tind=tind,
            xguards=False,
            yguards=False,
            info=False,
            strict=True,
        ))

    times = get('t_array').reshape(-1)
    if not len(times) or not np.all(np.isfinite(times)):
        raise ValueError('No finite saved times found.')

    if time is not None:
        if not np.isfinite(time):
            raise ValueError('time must be finite.')
        idx = int(np.argmin(np.abs(times - time)))
    else:
        if (not isinstance(time_index, (int, np.integer))
                or not -len(times) <= time_index < len(times)):
            raise ValueError('time_index is outside saved outputs.')
        idx = int(time_index) % len(times)

    By = get('By', tind=idx)
    if By.ndim == 4 and By.shape[0] == 1:
        By = By[0]
    if By.ndim != 3:
        raise ValueError(f'Unexpected By shape {By.shape}.')

    def metric(name):
        q = get(metric_prefix + name)
        if q.ndim == 1:
            q = q[None, :, None]
        elif q.ndim == 2:
            q = q[:, :, None]
        try:
            return np.broadcast_to(q, By.shape)
        except ValueError as exc:
            raise ValueError(
                f'{name} shape {q.shape} does not match {By.shape}.'
            ) from exc

    g = metric('g_11')
    if (not np.all(np.isfinite(g))
            or np.any(g <= 0)
            or not np.allclose(g, g[:1, :, :1])):
        raise ValueError('Expected positive g_11 depending only on Y.')

    for name, expected in [
        ('g_22', 1.0),
        ('g_33', g),
        ('g_12', 0.0),
        ('g_13', 0.0),
        ('g_23', 0.0),
    ]:
        if not np.allclose(
            metric(name), expected, rtol=1e-7, atol=1e-12
        ):
            raise ValueError(
                f'{name} is incompatible with the assumed diagonal metric.'
            )

    spacings = []
    for name in ('dx', 'dy', 'dz'):
        q = get(metric_prefix + name)
        v = float(q.flat[0])
        if (not np.all(np.isfinite(q))
                or v <= 0
                or not np.allclose(q, v, rtol=1e-8, atol=0)):
            raise ValueError(f'{name} must be positive and uniform.')
        spacings.append(v)

    print(
        f'Loaded output {idx}, t={times[idx]:g}, '
        f'guard-free By shape={By.shape}'
    )

    fig = make_figure(
        By,
        np.sqrt(g[0, :, 0]),
        *spacings,
        time_label=f't = {times[idx]:g} · output {idx}',
        **kwargs,
    )

    if html:
        fig.write_html(str(html), include_plotlyjs=True, full_html=True)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('path', nargs='?', default='.')
    parser.add_argument('--index', type=int, default=-1)
    parser.add_argument('--time', type=float)
    parser.add_argument('--out', default='flared_By.html')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--clim', type=float)
    parser.add_argument('--true-aspect', action='store_true')
    parser.add_argument('--light', action='store_true')
    parser.add_argument('--no-reversals', action='store_true')
    parser.add_argument('--metric-prefix', choices=['', 'm'], default='')
    parser.add_argument('--prefix', default='BOUT.dmp')
    args = parser.parse_args()

    plot_flared_by(
        args.path,
        time_index=args.index,
        time=args.time,
        html=args.out,
        prefix=args.prefix,
        metric_prefix=args.metric_prefix,
        normalize=args.normalize,
        stride=args.stride,
        clim=args.clim,
        true_aspect=args.true_aspect,
        dark=not args.light,
        reversals=not args.no_reversals,
    )
    print('Saved', args.out)


if __name__ == '__main__':
    main()
