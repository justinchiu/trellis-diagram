
from pylatex import (Document, TikZ, TikZNode,
                     TikZDraw, TikZCoordinate,
                     TikZUserPath, TikZOptions)

import numpy as np

# create trellis
T = 10
C = 32
# partition sizes of 16, sample half of them
S = 8

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
mask = np.zeros((C,), dtype=np.bool)
gumbel_noise = np.random.gumbel(size=mask.shape)
idx = gumbel_noise.argsort()[:int(C * 1/2)]
np.put_along_axis(mask, idx, True, -1)

dt = 6
dz = 6

dsize = "1.5pt"
ssize = "0.5pt"

doc = Document()

with doc.create(TikZ()) as pic:

    # dropout mask
    # edges
    for t0,t1,z0,z1 in edgegrid:
        in0 = (z0 // S) == partitions[t0]
        in1 = (z1 // S) == partitions[t1]
        in0 = True
        in1 = True
        on0 = mask[z0]
        on1 = mask[z1]
        if in0 and in1 and on0 and on1:
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
                    "opacity": "0.1",
                    "fill": "black",
                }),

            ))

    # labels
    for t in range(1, T):
        t0 = dt * t / T
        z0 = dz * -1.8 / C
        pic.append(TikZNode(
            handle = f"x{t}",
            at = TikZCoordinate(t0, z0),
            options = {
                #"fill": "white",
            },
            text = f"$x_{t}$",
        ))

    # dots
    for t,z in grid:
        partition = partitions[t]
        in_partition = (z // S) == partition
        color = 'white' if mask[z] else "black"
        size = dsize if mask[z] else ssize
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



out_path = "tex/diagram_full_drop"
doc.generate_pdf(out_path, clean_tex=False)
