"""Pose-only hard-view sampler. Filters are proxies, not visibility guarantees."""
import numpy as np


def sample_camera_span(views, num_source, num_target, rng, min_deg=5., max_deg=35., attempts=32):
    poses=np.stack([np.asarray(v.c2w,dtype=np.float64) for v in views])
    rotations=poses[:,:3,:3].reshape(len(views),9)
    angles=np.degrees(np.arccos(np.clip((rotations@rotations.T-1)/2,-1,1)))
    centers=poses[:,:3,3]
    distances=np.linalg.norm(centers[:,None]-centers[None],axis=-1)
    diameter=max(float(distances.max()),1e-9)
    for _ in range(attempts):
        targets=rng.sample(range(len(views)),num_target)
        eligible=[i for i in range(len(views)) if i not in targets and np.all(angles[targets,i]>=min_deg)]
        if len(eligible)<num_source:continue
        sources=rng.sample(eligible,num_source)
        # Every target needs at least one moderately rotated, spatially nearby
        # source. Angle lower bound holds for ALL sources, preventing a near-copy.
        supported=(angles[np.ix_(targets,sources)]<=max_deg)&(distances[np.ix_(targets,sources)]<=.5*diameter)
        if np.all(supported.any(axis=1)):
            return [views[i] for i in sources],[views[i] for i in targets]
    return None
