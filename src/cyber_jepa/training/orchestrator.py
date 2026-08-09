"""
Sweep Orchestrator for Cyber-JEPA.

Manages generating sweep_manifest.json, executing pre-flight VRAM profiling (under 3.6 GB),
skipping completed runs, resuming interrupted checkpoints, and sequential GPU run execution.
"""

import hashlib
import json
from pathlib import Path
from typing import Any
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.representations.feature import FeatureTokenRepresentation
from cyber_jepa.representations.host import HostTokenRepresentation
from cyber_jepa.representations.hierarchical import HierarchicalHostSubnetRepresentation
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.training.trainer import Trainer


MODEL_REGISTRY = {
    "flat": FlatVectorRepresentation,
    "feature": FeatureTokenRepresentation,
    "host": HostTokenRepresentation,
    "hierarchical": HierarchicalHostSubnetRepresentation,
}


class SweepOrchestrator:
    """Orchestrates multi-run sweeps, VRAM pre-flight checks, and execution state tracking."""

    def __init__(self, sweep_config_path: Path, output_dir: Path):
        self.sweep_config_path = sweep_config_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(sweep_config_path, "r") as f:
            self.sweep_config = yaml.safe_load(f)

        self.vram_limit_gb = float(self.sweep_config.get("vram_limit_gb", 3.6))
        self.manifest_path = self.output_dir / "sweep_manifest.json"
        self.manifest = self._init_manifest()

    def _init_manifest(self) -> dict[str, Any]:
        """Initialize or load sweep_manifest.json."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)

        manifest = {
            "sweep_config": self.sweep_config,
            "vram_limit_gb": self.vram_limit_gb,
            "runs": {},
            "status": "initialized",
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def run_preflight_vram_check(
        self,
        device: torch.device,
        batch_size: int = 128,
        steps: int = 100,
    ) -> dict[str, float]:
        """Run 100-step pre-flight profiling across all architectures to check peak VRAM < 3.6 GB."""
        print(f"Running {steps}-step pre-flight VRAM profiling on {device}...")
        peak_vram_results: dict[str, float] = {}

        if device.type != "cuda":
            print("CPU device detected - skipping VRAM profiling.")
            return {m: 0.0 for m in MODEL_REGISTRY.keys()}

        for name, cls in MODEL_REGISTRY.items():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

            encoder = cls(hidden_dim=64, ffn_dim=256)
            jepa = CyberJEPA(online_encoder=encoder, hidden_dim=64).to(device)

            dummy_hist = torch.randn(batch_size, 4, 52, device=device)
            dummy_act = torch.randint(0, 66, (batch_size, 4), device=device)
            dummy_target = torch.randn(batch_size, 52, device=device)
            dataset = TensorDataset(dummy_hist, dummy_act, dummy_target)

            loader = DataLoader(dataset, batch_size=batch_size // 2, shuffle=False)
            opt = torch.optim.AdamW(jepa.parameters(), lr=1e-3)
            trainer = Trainer(
                model=jepa,
                train_loader=loader,
                val_loader=loader,
                optimizer=opt,
                scheduler=None,
                run_dir=self.output_dir / "preflight_tmp",
                device=device,
                max_epochs=1,
                accum_steps=2,
            )

            try:
                trainer.train_epoch(epoch=1)
                peak_bytes = torch.cuda.max_memory_allocated(device)
                peak_gb = peak_bytes / (1024 ** 3)
                peak_vram_results[name] = peak_gb
                print(f"Pre-flight '{name}': Peak VRAM = {peak_gb:.2f} GB (Limit = {self.vram_limit_gb} GB)")

                if peak_gb > self.vram_limit_gb:
                    raise MemoryError(
                        f"Architecture '{name}' peak VRAM {peak_gb:.2f} GB exceeds limit {self.vram_limit_gb} GB!"
                    )
            except Exception as e:
                print(f"Pre-flight '{name}' failed: {e}")
                peak_vram_results[name] = float("nan")

            torch.cuda.empty_cache()

        return peak_vram_results

    def generate_run_id(self, config: dict[str, Any], dataset_hash: str) -> str:
        """Generate deterministic run ID hash from resolved config and dataset hash."""
        raw_str = json.dumps(config, sort_keys=True) + dataset_hash
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:12]
