
from pylatex import (Document, TikZ, TikZNode,
                     TikZDraw, TikZCoordinate,
                     TikZUserPath, TikZOptions)

import numpy as np

# create trellis
T = 10
C = 64
# partition sizes of 16, sample half of them
S = 16

num_partitions = C // S

grid = [(t,z) for t in range(1,T) for z in range(C)]

edgegrid = [
    (t0,t1,z0,z1)
    for t0,t1 in zip(range(1,T),range(2,T))
    for z0 in range(C)
    for z1 in range(C)
]

#partitions = np.random.choice(num_partitions, size=(T,))
partitions = [0, 3, 2, 1, 1, 0, 2, 2, 3, 2, 2, 1]

# TODO: make this shared between partitions,
# and also all equal size
mask = np.zeros((num_partitions, S), dtype=np.bool)
gumbel_noise = np.random.gumbel(size=mask.shape)
idx = gumbel_noise.argsort()[:,:S//2]
np.put_along_axis(mask, idx, True, -1)

mask_flat = mask.reshape(-1)

dt = 4
dz = 4

doc = Document()

with doc.create(TikZ()) as pic:

    phi = TikZNode(
        text = "$\phi$",
        handle = "phi",
        options = TikZOptions(
            "draw",
            "rectangle",
            # fucking annoying
            xshift = "50pt",
            yshift = "140pt",
        )
    )

    psi = TikZNode(
        text = "$\psi$",
        handle = "psi",
        options = TikZOptions(
            "draw",
            "rectangle",
            # fucking annoying
            xshift = "50pt",
            yshift = "-30pt",
        )
    )

    pic.append(phi)
    pic.append(psi)

    for t in range(1, T):
        pt = dt * t / T
        pic.append(TikZDraw(
            [
                phi.south,
                "--",
                TikZCoordinate(pt, dz),
            ],
            options=TikZOptions({
                "line width": "0.1pt",
                "fill": "black",
                "bend left": "45",
            }),

        ))

    for t in range(1, T-1):
        pt = dt * (t+0.5) / T
        pic.append(TikZDraw(
            [
                psi.north,
                "--",
                TikZCoordinate(pt, 0),
            ],
            options=TikZOptions({
                "line width": "0.1pt",
                "fill": "black",
            }),

        ))


    """
    # Can't add name...
    pic.append(TikZDraw(
        [
            # bottom left
            TikZCoordinate(-2 * dt / T, 0),
            "rectangle",
            # top right
            TikZCoordinate(-dt / T, 4),
        ],
        options=TikZOptions({
        }),
    ))
    """

    # dropout mask
    t = 0
    for z in range(C):
        color = 'white' if mask_flat[z] else "black"
        size = "0.6pt"
        pic.append(TikZDraw(
            [
                TikZCoordinate(dt * t / T, dz * z / C),
                'circle',
            ],
            options=TikZOptions({
                "line width": "0.01pt",
                "fill": color,
                "radius": size,
            }),
        ))

    # edges
    for t0,t1,z0,z1 in edgegrid:
        in0 = (z0 // S) == partitions[t0]
        in1 = (z1 // S) == partitions[t1]
        on0 = mask[partitions[t0], z0 % S]
        on1 = mask[partitions[t1], z1 % S]
        if in0 and in1 and on0 and on1:
            #import pdb; pdb.set_trace()
            pt0 = dt * t0 / T
            pz0 = dz * z0 / C
            pt1 = dt * t1 / T
            pz1 = dz * z1 / C
            pic.append(TikZDraw(
                [
                    TikZCoordinate(pt0, pz0),
                    "--",
                    TikZCoordinate(pt1, pz1),
                ],
                options=TikZOptions({
                    "line width": "0.1pt",
                    "opacity": "0.15",
                    "fill": "black",
                }),

            ))


    # partitions
    for t in range(1,T):
        partition = partitions[t]
        # bottom left
        t0 = dt * (t - 0.25) / T
        z0 = dz * (S * partition - 0.5) / C
        # top right
        t1 = dt * (t + 0.25) / T
        z1 = dz * (S * (partition+1) + 0.5 - 1) / C
        pic.append(TikZDraw(
            [
                TikZCoordinate(t0, z0),
                'rectangle',
                TikZCoordinate(t1, z1),
            ],
            options=TikZOptions({
                "line width": "0.2pt",
            }),
        ))

    # dots
    for t,z in grid:
        partition = partitions[t]
        in_partition = (z // S) == partition
        color = 'white' if mask[partition,z % S] and in_partition else "black"
        size = "0.4pt" if in_partition else "0.1pt"
        pic.append(TikZDraw(
            [
                TikZCoordinate(dt * t / T, dz * z / C),
                'circle',
            ],
            options=TikZOptions({
                "line width": "0.1pt",
                "fill": color,
                "radius": size,
            }),
        ))



out_path = "tex/diagram"
doc.generate_pdf(out_path, clean_tex=False)
