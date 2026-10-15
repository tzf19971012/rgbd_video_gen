import random

from .kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset


class Kubric4DPoseNVSDataset(Kubric4DRGBDNVSDataset):
    """Kubric4D variant for pose-NVS training.

    Same samples as the RGBD NVS dataset, with one structural change: the
    FIRST source view is pinned to index 0 and acts as the anchor of the
    world coordinate system (its pose becomes the identity after the
    relative-pose conversion in the training module).  The remaining source
    views (whose poses are treated as UNKNOWN by the model) are still
    randomly permuted, but only among positions 1..num_source_views-1 so
    the anchor never moves.

    No extra fields are needed: the module derives every mask from
    ``view_roles`` (source slots 0..S-1, target slots S..V-1) and
    ``num_source_views``:
      - RGB generated  : target slots only          (rgb loss mask, slot 1)
      - depth generated: all views                  (depth loss mask, slot 0)
      - pose unknown   : source slots 1..S-1        (anchor + targets known)
    """

    def _select_views(self, record, idx=0):
        source_views, target_views = super()._select_views(record, idx=idx)
        if self.src_order in ("pose_chain", "sequence"):
            # pose_chain: the chain already starts at the anchor (closest
            # source to the target); keep the trajectory order intact instead
            # of re-shuffling the tail.  sequence: the sources are already in
            # natural record order -- re-shuffling would defeat the mode.
            return source_views, target_views
        # super() may have shuffled ALL source views; keep index 0 as the
        # anchor and re-shuffle only the tail with a deterministically seeded
        # rng so eval reproducibility (deterministic_view_seed) is preserved.
        anchor, rest = source_views[0], list(source_views[1:])
        rng = random if self.deterministic_view_seed is None else random.Random(
            self.deterministic_view_seed + int(idx)
        )
        rng.shuffle(rest)
        return [anchor] + rest, target_views
