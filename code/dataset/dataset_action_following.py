"""
ActionFollowingData dataset for Ctrl-World.

This loader consumes precomputed SVD latents produced by
scripts/precompute_latents_action_following.py. It samples at chunk level while
preserving the project taxonomy:

  clean + perturbed + random_feasible + counterfactual_replay + exploration

For perturbed data, each canonical sample contributes exactly one prefix chunk.
Trajectory-level families use stride-1 sliding windows during training unless a
manifest record pins a fixed chunk_start, as test_quick does.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


PROTOCOL_SAMPLE_WEIGHTS = {
    "mix_4to1to1to1to1": {
        "clean": 2.6762118221657243,
        "perturbed": 1.0,
        "random_feasible": 0.18518518518518517,
        "counterfactual_replay": 0.6645622196378135,
        "exploration": 3.389256058295204,
    },
    "mix_1to1to1to1to1": {
        "clean": 0.6690529555414311,
        "perturbed": 1.0,
        "random_feasible": 0.18518518518518517,
        "counterfactual_replay": 0.6645622196378135,
        "exploration": 3.389256058295204,
    },
    "enhanced_1to1to1to1": {
        "perturbed": 1.0,
        "random_feasible": 0.18518518518518517,
        "counterfactual_replay": 0.6645622196378135,
        "exploration": 3.389256058295204,
    },
    "mix_3to1to1to1": {
        "clean": 3.16,
        "perturbed": 1.0,
        "random_feasible": 0.185,
        "counterfactual_replay": 0.522,
    },
    "mix_1to1to1to1": {
        "clean": 1.05,
        "perturbed": 1.0,
        "random_feasible": 0.185,
        "counterfactual_replay": 0.522,
    },
    "clean_only": {"clean": 1.0},
    "enhanced_1to1to1": {
        "perturbed": 1.0,
        "random_feasible": 0.185,
        "counterfactual_replay": 0.522,
    },
}

PROTOCOL_TARGET_PROBS = {
    "mix_4to1to1to1to1": {
        "clean": 0.5,
        "perturbed": 0.125,
        "random_feasible": 0.125,
        "counterfactual_replay": 0.125,
        "exploration": 0.125,
    },
    "mix_1to1to1to1to1": {
        "clean": 0.2,
        "perturbed": 0.2,
        "random_feasible": 0.2,
        "counterfactual_replay": 0.2,
        "exploration": 0.2,
    },
    "enhanced_1to1to1to1": {
        "perturbed": 0.25,
        "random_feasible": 0.25,
        "counterfactual_replay": 0.25,
        "exploration": 0.25,
    },
    "mix_3to1to1to1": {
        "clean": 0.5,
        "perturbed": 1.0 / 6.0,
        "random_feasible": 1.0 / 6.0,
        "counterfactual_replay": 1.0 / 6.0,
    },
    "mix_1to1to1to1": {
        "clean": 0.25,
        "perturbed": 0.25,
        "random_feasible": 0.25,
        "counterfactual_replay": 0.25,
    },
    "clean_only": {"clean": 1.0},
    "enhanced_1to1to1": {
        "perturbed": 1.0 / 3.0,
        "random_feasible": 1.0 / 3.0,
        "counterfactual_replay": 1.0 / 3.0,
    },
}


def _read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _canonical_family(record):
    family = str(record.get("family", "")).lower()
    asset_id = str(record.get("asset_id", "")).lower()
    source = f"{family} {asset_id}"
    if "counterfactual" in source:
        return "counterfactual_replay"
    if "exploration" in source or "policy_rollout" in source:
        return "exploration"
    if "random_feasible" in source:
        return "random_feasible"
    if "perturbed" in source or family in {"pca", "raw"}:
        return "perturbed"
    if family in {"clean", "expert_clean", "expert"} or "clean" in source:
        return "clean"
    return family


def _canonical_subtype(record):
    subtype = record.get("subtype")
    if subtype:
        return str(subtype)
    text = f"{record.get('family', '')} {record.get('asset_id', '')} {record.get('sample_id', '')}".lower()
    for key in ["policy_rollout", "pca", "raw", "uniform", "weighted", "clean"]:
        if key in text:
            return key
    return "default"


def _sampling_mode(record, family):
    if record.get("fixed_start") is not None or record.get("chunk_start") is not None:
        return "fixed"
    if family == "perturbed" or record.get("unit_level") == "chunk":
        return "prefix"
    return "sliding"


def _record_file(record, latent_root):
    path = record.get("file") or record.get("latent_file")
    if path is None:
        raise KeyError(f"manifest record has no latent file: {record}")
    if os.path.isabs(path):
        return path
    return os.path.join(latent_root, path)


class ActionFollowingCtrlWorldDataset(Dataset):
    def __init__(self, args, mode: str = "train"):
        self.args = args
        self.mode = mode
        self.num_history = int(args.num_history)
        self.num_frames = int(args.num_frames)
        self.T = self.num_history + self.num_frames
        self.action_dim = int(getattr(args, "action_dim", 14))
        self.seed = int(getattr(args, "action_following_sampling_seed", 20260630))
        self.use_ee_head = bool(getattr(args, "use_ee_head", False))
        self.protocol = getattr(args, "action_following_sampling_protocol", "mix_4to1to1to1to1")
        self.latent_root = getattr(args, "action_following_latent_root", None)
        if not self.latent_root:
            raise ValueError("--action_following_latent_root is required")

        chunk_size = int(getattr(args, "action_following_chunk_size", self.T) or self.T)
        if chunk_size != self.T:
            raise ValueError(
                f"ActionFollowing chunk_size={chunk_size} but num_history+num_frames={self.T}. "
                "For chunk32 with Ctrl-World history=6, use --num_frames 26."
            )

        manifest_path = self._manifest_path(args, mode)
        self.records = _read_jsonl(manifest_path)
        if not self.records:
            raise RuntimeError(f"empty ActionFollowing manifest: {manifest_path}")

        stat_path = getattr(args, "action_following_stat_path", None)
        if not stat_path:
            stat_path = os.path.join(args.dataset_meta_info_path, args.dataset_cfgs.split("+")[0], "stat.json")
        with open(stat_path) as f:
            stat = json.load(f)
        self.p01 = np.array(stat["state_01"], dtype=np.float32)[: self.action_dim]
        self.p99 = np.array(stat["state_99"], dtype=np.float32)[: self.action_dim]

        for rec in self.records:
            family = _canonical_family(rec)
            rec["_family"] = family
            rec["_subtype"] = _canonical_subtype(rec)
            rec["_sampling_mode"] = _sampling_mode(rec, family)
            rec["_nwin"] = self._num_windows(rec)

        self.eval_mode = mode in {"val", "test", "test_quick"} or any(
            rec["_sampling_mode"] == "fixed" for rec in self.records
        )
        self.by_family = self._build_index(self.records)
        if self.eval_mode:
            self.family_names = sorted(self.by_family.keys())
            self.family_p = np.ones(len(self.family_names), dtype=np.float64) / max(1, len(self.family_names))
            self.sample_weights = None
            self.train_records = []
            self.train_cum_weights = np.array([], dtype=np.float64)
            self.train_total_weight = 0.0
        else:
            self._build_train_sampler()

        default_len = self._default_train_length() if not self.eval_mode else len(self.records)
        self.virtual_length = int(getattr(args, "action_following_dataset_length", 0) or default_len)
        if self.eval_mode:
            self.virtual_length = len(self.records)
        if self.virtual_length <= 0:
            raise RuntimeError("No ActionFollowing samples found.")

        self._print_summary(manifest_path)

    def _manifest_path(self, args, mode):
        if mode in {"val", "test_quick"}:
            explicit = getattr(args, "action_following_val_manifest", None)
            default_name = "test_quick.jsonl"
        else:
            explicit = getattr(args, "action_following_train_manifest", None)
            default_name = "train.jsonl"
        path = explicit or os.path.join(args.action_following_latent_root, "manifests", default_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"ActionFollowing manifest not found: {path}")
        return path

    def _num_windows(self, rec):
        if rec["_sampling_mode"] in {"fixed", "prefix"}:
            return 1
        length = int(rec.get("length", 0))
        return max(1, length - self.T + 1)

    def _build_index(self, records):
        by_family = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for rec in records:
            by_family[rec["_family"]][rec["_subtype"]][rec.get("task", "unknown")].append(rec)
        return by_family

    def _build_train_sampler(self):
        if self.protocol not in PROTOCOL_SAMPLE_WEIGHTS:
            raise ValueError(f"Unknown ActionFollowing sampling protocol: {self.protocol}")
        sample_weights = PROTOCOL_SAMPLE_WEIGHTS[self.protocol]
        missing = [family for family in sample_weights if family not in self.by_family]
        if missing:
            raise RuntimeError(f"protocol {self.protocol} requested missing families: {missing}")

        train_records = []
        base_record_weights = []
        effective_windows = defaultdict(int)
        base_weighted_mass = defaultdict(float)
        for rec in self.records:
            family = rec["_family"]
            if family not in sample_weights:
                continue
            nwin = int(rec["_nwin"])
            if nwin <= 0:
                continue
            train_records.append(rec)
            mass = float(sample_weights[family]) * float(nwin)
            base_record_weights.append(mass)
            effective_windows[family] += nwin
            base_weighted_mass[family] += mass

        if not train_records:
            raise RuntimeError(f"No train records available for protocol {self.protocol}.")

        target_probs = PROTOCOL_TARGET_PROBS.get(self.protocol)
        family_scale = {}
        if target_probs:
            target_total = sum(float(target_probs.get(family, 0.0)) for family in sample_weights)
            if target_total <= 0:
                raise RuntimeError(f"Non-positive target probability mass for protocol {self.protocol}.")
            for family in sample_weights:
                base_mass = float(base_weighted_mass.get(family, 0.0))
                if base_mass <= 0:
                    raise RuntimeError(f"Family {family} has no base sampler mass for protocol {self.protocol}.")
                family_scale[family] = float(target_probs[family]) / target_total / base_mass
        else:
            family_scale = {family: 1.0 for family in sample_weights}

        record_weights = [
            mass * family_scale[rec["_family"]]
            for rec, mass in zip(train_records, base_record_weights)
        ]
        final_per_chunk_weights = {
            family: float(sample_weights[family]) * float(family_scale[family])
            for family in sample_weights
        }
        ref_family = "perturbed" if "perturbed" in final_per_chunk_weights else next(iter(final_per_chunk_weights))
        ref_weight = final_per_chunk_weights[ref_family]
        if ref_weight > 0:
            final_per_chunk_weights_normalized = {
                family: weight / ref_weight
                for family, weight in final_per_chunk_weights.items()
            }
        else:
            final_per_chunk_weights_normalized = dict(final_per_chunk_weights)
        weighted_mass = defaultdict(float)
        for rec, mass in zip(train_records, record_weights):
            weighted_mass[rec["_family"]] += float(mass)

        cum_weights = np.cumsum(np.asarray(record_weights, dtype=np.float64))
        total = float(cum_weights[-1])
        if total <= 0:
            raise RuntimeError(f"Non-positive sampler mass for protocol {self.protocol}.")

        self.sample_weights = dict(sample_weights)
        self.train_records = train_records
        self.train_cum_weights = cum_weights
        self.train_total_weight = total
        self.family_names = list(sample_weights.keys())
        self.family_p = np.array(
            [weighted_mass.get(name, 0.0) / total for name in self.family_names],
            dtype=np.float64,
        )
        self.effective_windows_by_family = dict(effective_windows)
        self.base_weighted_mass_by_family = dict(base_weighted_mass)
        self.family_normalization_by_family = dict(family_scale)
        self.final_per_chunk_sample_weights = dict(final_per_chunk_weights)
        self.final_per_chunk_sample_weights_normalized = dict(final_per_chunk_weights_normalized)
        self.weighted_mass_by_family = dict(weighted_mass)

    def _default_train_length(self):
        if self.eval_mode:
            return len(self.records)
        return int(sum(rec["_nwin"] for rec in self.train_records))

    def _print_summary(self, manifest_path):
        counts = defaultdict(int)
        windows = defaultdict(int)
        for rec in self.records:
            counts[rec["_family"]] += 1
            windows[rec["_family"]] += rec["_nwin"]
        print("[ActionFollowingCtrlWorldDataset]")
        print(f"  mode={self.mode}, protocol={self.protocol}, T={self.T}, virtual_length={self.virtual_length}")
        print(f"  manifest={manifest_path}")
        print(f"  sampler_family_prob={dict(zip(self.family_names, self.family_p.tolist()))}")
        if not self.eval_mode:
            print(f"  per_chunk_sample_weights={self.sample_weights}")
            print(f"  final_per_chunk_sample_weights_normalized={self.final_per_chunk_sample_weights_normalized}")
            print(f"  manifest_family_normalization={self.family_normalization_by_family}")
        for family in sorted(counts):
            print(f"  {family}: records={counts[family]}, effective_windows={windows[family]}")

    def __len__(self):
        return self.virtual_length

    def _choose_train_record_from_rng(self, rng):
        draw = float(rng.random()) * self.train_total_weight
        rec_idx = int(np.searchsorted(self.train_cum_weights, draw, side="right"))
        if rec_idx >= len(self.train_records):
            rec_idx = len(self.train_records) - 1
        rec = self.train_records[rec_idx]
        if rec["_sampling_mode"] == "fixed":
            start = int(rec.get("fixed_start", rec.get("chunk_start", 0)))
        elif rec["_sampling_mode"] == "prefix":
            start = 0
        else:
            start = int(rng.integers(0, rec["_nwin"]))
        return rec, start

    def _choose_train_record(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        return self._choose_train_record_from_rng(rng)

    def audit_sampler(self, num_samples=10000, seed=None, tolerance=0.02, raise_on_fail=True):
        if self.eval_mode:
            raise RuntimeError("Sampler audit is only valid for train mode.")
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if seed is None:
            seed = self.seed
        seed = int(seed)
        tolerance = float(tolerance)
        counts = defaultdict(int)
        for i in range(num_samples):
            rng = np.random.default_rng(seed + i)
            rec, _ = self._choose_train_record_from_rng(rng)
            counts[rec["_family"]] += 1

        observed = {family: counts.get(family, 0) / float(num_samples) for family in self.family_names}
        target = PROTOCOL_TARGET_PROBS.get(self.protocol, {})
        deltas = {family: observed.get(family, 0.0) - target.get(family, 0.0) for family in self.family_names}
        passed = all(abs(deltas.get(family, 0.0)) <= tolerance for family in target)
        report = {
            "protocol": self.protocol,
            "seed": seed,
            "num_samples": num_samples,
            "tolerance": tolerance,
            "base_per_chunk_sample_weights": self.sample_weights,
            "final_per_chunk_sample_weights": self.final_per_chunk_sample_weights,
            "final_per_chunk_sample_weights_normalized": self.final_per_chunk_sample_weights_normalized,
            "effective_windows_by_family": self.effective_windows_by_family,
            "base_weighted_mass_by_family": self.base_weighted_mass_by_family,
            "manifest_family_normalization": self.family_normalization_by_family,
            "final_weighted_mass_by_family": self.weighted_mass_by_family,
            "sampler_family_prob_from_mass": dict(zip(self.family_names, self.family_p.tolist())),
            "target_family_prob": target,
            "observed_family_prob": observed,
            "counts": dict(counts),
            "delta": deltas,
            "passed": passed,
        }
        if raise_on_fail and not passed:
            raise RuntimeError(f"ActionFollowing sampler audit failed: {json.dumps(report, sort_keys=True)}")
        return report

    def _choose_eval_record(self, idx):
        rec = self.records[int(idx) % len(self.records)]
        start = int(rec.get("fixed_start", rec.get("chunk_start", 0)))
        return rec, start

    def __getitem__(self, idx):
        rec, start = self._choose_eval_record(idx) if self.eval_mode else self._choose_train_record(idx)
        data = torch.load(_record_file(rec, self.latent_root), map_location="cpu", weights_only=False)

        latent_full = data["latent"]
        action_pos = np.asarray(data["action_pos"], dtype=np.float32)
        if self.use_ee_head:
            if "ee_target" not in data:
                raise KeyError(
                    f"EE head requested but ee_target missing in {rec}. "
                    "Clean LeRobot ActionFollowing latents currently do not provide EE targets."
                )
            ee_target_full = np.asarray(data["ee_target"], dtype=np.float32)

        T_actual = min(int(latent_full.shape[0]), int(action_pos.shape[0]))
        if self.use_ee_head:
            T_actual = min(T_actual, int(ee_target_full.shape[0]))
        latent_full = latent_full[:T_actual]
        action_pos = action_pos[:T_actual]
        if self.use_ee_head:
            ee_target_full = ee_target_full[:T_actual]

        if T_actual < self.T:
            pad = self.T - T_actual
            latent_full = torch.cat([latent_full, latent_full[-1:].repeat(pad, 1, 1, 1)], dim=0)
            action_pos = np.concatenate([action_pos, action_pos[-1:].repeat(pad, axis=0)], axis=0)
            if self.use_ee_head:
                ee_target_full = np.concatenate([ee_target_full, ee_target_full[-1:].repeat(pad, axis=0)], axis=0)
            T_actual = self.T
            start = 0

        start = min(max(0, int(start)), max(0, T_actual - self.T))
        latent = latent_full[start:start + self.T]
        action = action_pos[start:start + self.T, : self.action_dim]
        action = np.clip(2 * (action - self.p01) / (self.p99 - self.p01 + 1e-8) - 1, -1, 1)

        sample = {
            "latent": latent.float(),
            "text": rec.get("text") or data.get("text") or str(rec.get("task", "")).replace("_", " "),
            "action": torch.tensor(action, dtype=torch.float32),
            "family": rec["_family"],
            "task": rec.get("task", ""),
            "sample_id": rec.get("sample_id", ""),
            "chunk_start": start,
        }
        if self.use_ee_head:
            sample["ee_target"] = torch.tensor(ee_target_full[start:start + self.T], dtype=torch.float32)
        return sample
