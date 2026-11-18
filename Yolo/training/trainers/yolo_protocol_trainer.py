import math
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
import time

import torch

from ultralytics.engine import trainer as trainer_module
from ultralytics.models.yolo.detect import (
    DetectionTrainer,
    DetectionValidator,
)


class YOLOProtocolValidator(DetectionValidator):
    """Validate using actual model weights, not EMA weights."""

    def get_model(self, model, trainer=None):
        if trainer is not None:
            return trainer.model

        return super().get_model(model, trainer)


class YOLOProtocolTrainer(DetectionTrainer):
    """
    Custom YOLO11s trainer for the COMP9517 project protocol.
    """

    PEAK_LR = 0.001
    INITIAL_LR_RATIO = 0.10
    FINAL_LR_RATIO = 0.01
    WARMUP_EPOCHS = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_callback("on_train_epoch_start", self._protocol_epoch_start)
        self.add_callback("on_train_epoch_end", self._protocol_epoch_end)

    def _protocol_epoch_start(self, trainer):
        """Record the start of training for this epoch."""
        self.protocol_epoch_start_time = time.perf_counter()
        self.protocol_epoch_start_update = self.protocol_update_count

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _protocol_epoch_end(self, trainer):
        """Record training duration and peak GPU memory."""
        self.protocol_train_seconds = (
            time.perf_counter() - self.protocol_epoch_start_time
        )

        self.protocol_epoch_updates = (
            self.protocol_update_count - self.protocol_epoch_start_update
        )

        if torch.cuda.is_available() and self.device.type == "cuda":
            self.protocol_peak_memory_mb = torch.cuda.max_memory_allocated(
                self.device
            ) / (1024**2)
        else:
            self.protocol_peak_memory_mb = 0.0

    def _get_warmup_iterations(self, num_batches):
        """Disable Ultralytics' default LR and accumulation warm-up."""
        return 0

    def get_validator(self):
        """Use the non-EMA protocol validator."""
        return YOLOProtocolValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def resume_training(self, ckpt):
        """Restore normal training state and custom protocol state."""

        # Defaults for a new training run.
        self.protocol_update_count = 0
        self.protocol_best_map50 = float("-inf")
        self.protocol_best_map5095 = float("-inf")

        super().resume_training(ckpt)

        # Restore custom state when resuming.
        if ckpt is not None and self.resume:
            self.protocol_update_count = ckpt.get("protocol_update_count", 0)
            self.protocol_best_map50 = ckpt.get("protocol_best_map50", float("-inf"))
            self.protocol_best_map5095 = ckpt.get(
                "protocol_best_map5095", float("-inf")
            )

    def _setup_train(self):
        """Set up training and register the protocol LR hook."""

        super()._setup_train()

        print("\nGradient accumulation verification:")
        print("Configured batch size:", self.args.batch)
        print("Accumulation factor:", self.accumulate)
        print(
            "Effective batch size:",
            self.args.batch * self.accumulate,
        )

        if self.args.batch != 8 or self.accumulate != 2:
            raise ValueError("Protocol requires batch size 8 and accumulation 2.")

        self.protocol_total_epochs = self.epochs

        batches_per_epoch = len(self.train_loader)
        total_batches = self.protocol_total_epochs * batches_per_epoch

        self.protocol_total_updates = total_batches // self.accumulate

        self.protocol_warmup_updates = (
            self.WARMUP_EPOCHS * batches_per_epoch
        ) // self.accumulate

        if self.protocol_total_updates < 2:
            raise ValueError("Insufficient optimiser updates.")

        if self.protocol_warmup_updates < 2:
            raise ValueError("Insufficient warm-up updates.")

        if self.protocol_update_count > self.protocol_total_updates:
            raise ValueError("Restored LR update counter exceeds schedule.")

        # Register after resume_training has restored the counter.
        self.optimizer.register_step_pre_hook(self._update_protocol_lr)

        print("\nProtocol LR configuration:")
        print("Total epochs:", self.protocol_total_epochs)
        print("Batches per epoch:", batches_per_epoch)
        print("Gradient accumulation:", self.accumulate)
        print("Total optimiser updates:", self.protocol_total_updates)
        print("Warm-up optimiser updates:", self.protocol_warmup_updates)
        print("Restored update counter:", self.protocol_update_count)

    def get_lr_multiplier(self, update):
        """Linear warm-up followed by cosine decay."""

        warmup_updates = self.protocol_warmup_updates
        total_updates = self.protocol_total_updates

        if update < warmup_updates:
            progress = update / (warmup_updates - 1)

            return self.INITIAL_LR_RATIO + (1.0 - self.INITIAL_LR_RATIO) * progress

        decay_updates = total_updates - warmup_updates

        if decay_updates <= 1:
            return self.FINAL_LR_RATIO

        progress = (update - warmup_updates) / (decay_updates - 1)
        progress = min(max(progress, 0.0), 1.0)

        return self.FINAL_LR_RATIO + 0.5 * (1.0 - self.FINAL_LR_RATIO) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _update_protocol_lr(self, optimizer, args, kwargs):
        """Apply protocol LR before each actual optimiser step."""

        update = self.protocol_update_count

        if update >= self.protocol_total_updates:
            raise RuntimeError("LR schedule has exceeded its update budget.")

        learning_rate = self.PEAK_LR * self.get_lr_multiplier(update)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        self.protocol_update_count += 1

    def validate(self):
        """
        Validate using non-EMA weights and record protocol metrics.
        AP50 remains the checkpoint fitness.
        """
        validation_start = time.perf_counter()

        metrics = self.validator(self)

        validation_seconds = time.perf_counter() - validation_start

        if metrics is None:
            return None, None

        ap50 = metrics["metrics/mAP50(B)"]
        ap5095 = metrics["metrics/mAP50-95(B)"]
        ap75 = self.validator.metrics.box.map75

        # Keep Ultralytics' fitness field out of checkpoint ranking.
        metrics.pop("fitness", None)

        if self.best_fitness is None or ap50 > self.best_fitness:
            self.best_fitness = ap50

        # Additional protocol logging.
        metrics["metrics/mAP75(B)"] = float(ap75)
        metrics["protocol/optimizer_updates"] = self.protocol_update_count
        metrics["protocol/epoch_optimizer_updates"] = self.protocol_epoch_updates
        metrics["protocol/train_time_seconds"] = self.protocol_train_seconds
        metrics["protocol/val_time_seconds"] = validation_seconds
        metrics["protocol/peak_gpu_memory_mb"] = self.protocol_peak_memory_mb

        print(
            f"\nProtocol validation: "
            f"AP50={ap50:.5f}, "
            f"AP50-95={ap5095:.5f}, "
            f"AP75={ap75:.5f}"
        )

        return metrics, ap50

    def save_model(self):
        """
        Save actual model weights, optimiser state and protocol state.
        EMA weights are never saved.
        """

        model = (
            deepcopy(trainer_module.unwrap_model(self.model))
            .half()
            .to(memory_format=torch.contiguous_format)
        )

        if hasattr(model, "criterion"):
            model.criterion = None

        # Prevent non-finite values from entering saved weights.
        for value in model.state_dict().values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                torch.nan_to_num_(value)

        checkpoint = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "stopper": {
                "best_fitness": self.stopper.best_fitness,
                "best_epoch": self.stopper.best_epoch,
            },
            "model": model,
            "ema": None,
            "updates": 0,
            "modelopt": None,
            "optimizer": trainer_module.convert_optimizer_state_dict_to_fp16(
                deepcopy(self.optimizer.state_dict())
            ),
            "scaler": self.scaler.state_dict(),
            "train_args": vars(self.args),
            "train_metrics": {
                **self.metrics,
                "fitness": self.fitness,
            },
            "train_results": self.read_results_csv(),
            "date": datetime.now().astimezone().isoformat(),
            "version": trainer_module.__version__,
            "git": {
                "root": str(trainer_module.GIT.root),
                "branch": trainer_module.GIT.branch,
                "commit": trainer_module.GIT.commit,
                "message": trainer_module.GIT.message,
                "origin": trainer_module.GIT.origin,
            },
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "protocol_update_count": self.protocol_update_count,
            "protocol_best_map50": self.protocol_best_map50,
            "protocol_best_map5095": self.protocol_best_map5095,
        }

        self.wdir.mkdir(parents=True, exist_ok=True)

        # Apply AP50-first, AP50-95-second ranking.
        if self.metrics and "metrics/mAP50(B)" in self.metrics:
            current_map50 = self.metrics["metrics/mAP50(B)"]
            current_map5095 = self.metrics["metrics/mAP50-95(B)"]

            is_better = current_map50 > self.protocol_best_map50 or (
                current_map50 == self.protocol_best_map50
                and current_map5095 > self.protocol_best_map5095
            )

            if is_better:
                self.protocol_best_map50 = current_map50
                self.protocol_best_map5095 = current_map5095

                checkpoint["protocol_best_map50"] = current_map50
                checkpoint["protocol_best_map5095"] = current_map5095

                torch.save(checkpoint, self.best)

                print(
                    f"New best checkpoint: "
                    f"AP50={current_map50:.5f}, "
                    f"AP50-95={current_map5095:.5f}"
                )

        # Always save the latest checkpoint.
        torch.save(checkpoint, self.last)

        # Save periodic epoch archives.
        if self.save_period > 0 and (self.epoch + 1) % self.save_period == 0:
            epoch_path = self.wdir / f"epoch{self.epoch + 1}.pt"
            torch.save(checkpoint, epoch_path)

        return True
