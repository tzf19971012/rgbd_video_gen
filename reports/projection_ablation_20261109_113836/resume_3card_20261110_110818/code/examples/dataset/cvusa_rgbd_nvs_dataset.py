"""CVUSA 5-view street -> satellite dataset.

Identical to Kubric4DRGBDNVSDataset except view selection: the record must
contain exactly one satellite view (view_id == "5_SAT") which is always the
generation target; the remaining (street) views are the conditioning sources.
"""
import random

from examples.dataset.kubric4d_rgbd_nvs_dataset import Kubric4DRGBDNVSDataset

SAT_VIEW_ID = "5_SAT"


class CVUSA5SatDataset(Kubric4DRGBDNVSDataset):
    def _select_views(self, record, idx=0):
        views = list(record["views"])
        if len(views) < self.num_views:
            raise ValueError("Not enough views in record.")

        rng = random if self.deterministic_view_seed is None else random.Random(
            self.deterministic_view_seed + int(idx)
        )

        target_pool = [v for v in views if v.view_id == SAT_VIEW_ID]
        source_pool = [v for v in views if v.view_id != SAT_VIEW_ID]
        if not target_pool:
            raise ValueError(f"scene {record.get('scene_id')!r} has no {SAT_VIEW_ID} view")
        if len(source_pool) < self.num_source_views:
            raise ValueError(
                f"scene {record.get('scene_id')!r}: need {self.num_source_views} "
                f"street views, found {len(source_pool)}"
            )

        target_views = rng.sample(target_pool, self.num_target_views)
        source_views = rng.sample(source_pool, self.num_source_views)
        if self.random_permute_source_views:
            rng.shuffle(source_views)
        return source_views, target_views
