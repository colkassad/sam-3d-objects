"""Profile staged SAM 3D inference on a representative image and mask."""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebook"))

from inference import Inference, load_image, load_single_mask  # noqa: E402


class SystemTelemetry:
    """Sample GPU clocks/thermals and Linux host memory during inference."""

    def __init__(self, interval_seconds=0.5):
        self.interval_seconds = interval_seconds
        self.samples = []
        self.errors = []
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._nvml_handle = None

    @staticmethod
    def _read_kib_file(path):
        values = {}
        try:
            for line in Path(path).read_text().splitlines():
                key, raw = line.split(":", 1)
                fields = raw.split()
                if fields:
                    try:
                        values[key] = int(fields[0]) * 1024
                    except ValueError:
                        continue
        except OSError as error:
            return {}, str(error)
        return values, None

    def _initialize_nvml(self):
        if not torch.cuda.is_available():
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                torch.cuda.current_device()
            )
        except Exception as error:
            self.errors.append(f"NVML telemetry unavailable: {error}")
            self._nvml = None
            self._nvml_handle = None

    def _record_error_once(self, message):
        if message not in self.errors:
            self.errors.append(message)

    def sample_now(self):
        sample = {"monotonic_seconds": time.perf_counter()}
        if self._nvml_handle is not None:
            nvml = self._nvml

            def read_nvml(key, getter):
                try:
                    sample[key] = getter()
                except Exception as error:
                    self._record_error_once(f"NVML {key} unavailable: {error}")

            read_nvml(
                "gpu_temperature_c",
                lambda: nvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, nvml.NVML_TEMPERATURE_GPU
                ),
            )
            read_nvml(
                "gpu_sm_clock_mhz",
                lambda: nvml.nvmlDeviceGetClockInfo(
                    self._nvml_handle, nvml.NVML_CLOCK_SM
                ),
            )
            read_nvml(
                "gpu_memory_clock_mhz",
                lambda: nvml.nvmlDeviceGetClockInfo(
                    self._nvml_handle, nvml.NVML_CLOCK_MEM
                ),
            )
            read_nvml(
                "gpu_power_watts",
                lambda: nvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0,
            )
            try:
                utilization = nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                sample["gpu_utilization_percent"] = utilization.gpu
                sample["gpu_memory_utilization_percent"] = utilization.memory
            except Exception as error:
                self._record_error_once(f"NVML utilization unavailable: {error}")

        meminfo, mem_error = self._read_kib_file("/proc/meminfo")
        status, status_error = self._read_kib_file("/proc/self/status")
        if mem_error is None:
            sample["host_memory_available_bytes"] = meminfo.get("MemAvailable")
            sample["host_swap_used_bytes"] = max(
                0, meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)
            )
        if status_error is None:
            sample["process_rss_bytes"] = status.get("VmRSS")
            sample["process_swap_bytes"] = status.get("VmSwap", 0)
        self.samples.append(sample)
        return sample

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self.sample_now()

    def start(self):
        self._initialize_nvml()
        self.sample_now()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self.sample_now()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception as error:
                self.errors.append(f"NVML shutdown failed: {error}")

    def summarize(self, started, ended):
        samples = [
            item
            for item in self.samples
            if started <= item["monotonic_seconds"] <= ended
        ]
        summary = {"sample_count": len(samples)}
        aggregations = {
            "gpu_temperature_c": ("max",),
            "gpu_sm_clock_mhz": ("min", "max", "mean"),
            "gpu_memory_clock_mhz": ("min", "max", "mean"),
            "gpu_power_watts": ("max", "mean"),
            "gpu_utilization_percent": ("max", "mean"),
            "gpu_memory_utilization_percent": ("max", "mean"),
            "host_memory_available_bytes": ("min",),
            "host_swap_used_bytes": ("max",),
            "process_rss_bytes": ("max",),
            "process_swap_bytes": ("max",),
        }
        for key, operations in aggregations.items():
            values = [item[key] for item in samples if item.get(key) is not None]
            if not values:
                continue
            for operation in operations:
                if operation == "min":
                    value = min(values)
                elif operation == "max":
                    value = max(values)
                else:
                    value = sum(values) / len(values)
                summary[f"{key}_{operation}"] = value
        return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument(
        "--image",
        default="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png",
    )
    parser.add_argument(
        "--mask-dir",
        default="notebook/images/shutterstock_stylish_kidsroom_1640806567",
    )
    parser.add_argument("--mask-index", type=int, default=14)
    parser.add_argument(
        "--memory-profile", choices=("auto", "low_vram", "resident"), default="auto"
    )
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--max-peak-gib", type=float, default=14.5)
    parser.add_argument("--max-live-growth-mib", type=float, default=16.0)
    parser.add_argument("--telemetry-interval", type=float, default=0.5)
    parser.add_argument("--mesh-target-faces", type=int)
    parser.add_argument("--flat-shading", action="store_true")
    parser.add_argument("--stage1-inference-steps", type=int)
    parser.add_argument("--stage2-inference-steps", type=int)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    inference = Inference(
        args.config,
        compile=False,
        memory_profile=args.memory_profile,
        output_formats=("mesh",),
        profile_memory=True,
    )
    startup_seconds = time.perf_counter() - started
    image = load_image(args.image)
    mask = load_single_mask(args.mask_dir, index=args.mask_index)

    run_seconds = []
    run_ranges = []
    live_allocated_bytes = []
    mesh_face_counts = []
    initial_measurements = len(inference.get_memory_report())
    telemetry = SystemTelemetry(args.telemetry_interval)
    telemetry.start()
    try:
        for run_index in range(args.warm_runs + 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            run_started = time.perf_counter()
            output = inference(
                image,
                mask,
                seed=42,
                mesh_target_faces=args.mesh_target_faces,
                flat_shading=args.flat_shading,
                stage1_inference_steps=args.stage1_inference_steps,
                stage2_inference_steps=args.stage2_inference_steps,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            run_ended = time.perf_counter()
            elapsed = run_ended - run_started
            mesh_face_counts.append(int(output["mesh"][0].faces.shape[0]))
            run_ranges.append(
                {
                    "run_index": run_index,
                    "kind": "cold" if run_index == 0 else "warm",
                    "elapsed_seconds": elapsed,
                    "started_monotonic_seconds": run_started,
                    "ended_monotonic_seconds": run_ended,
                }
            )
            if run_index > 0:
                run_seconds.append(elapsed)
                if torch.cuda.is_available():
                    live_allocated_bytes.append(torch.cuda.memory_allocated())
            if output["mesh"][0].vertices.is_cuda:
                raise RuntimeError("Low-VRAM mesh output retained a CUDA tensor")
    finally:
        telemetry.stop()

    measurements = [
        dict(item)
        for item in inference.get_memory_report()[initial_measurements:]
    ]
    for item in measurements:
        compute_started = item.get("compute_started_monotonic_seconds")
        compute_ended = item.get("compute_ended_monotonic_seconds")
        if compute_started is not None and compute_ended is not None:
            item["compute_telemetry"] = telemetry.summarize(
                compute_started, compute_ended
            )
    run_telemetry = []
    for item in run_ranges:
        run_summary = dict(item)
        run_summary["telemetry"] = telemetry.summarize(
            item["started_monotonic_seconds"], item["ended_monotonic_seconds"]
        )
        run_telemetry.append(run_summary)
    peak_allocated = max(
        (item["peak_allocated_bytes"] for item in measurements), default=0
    )
    peak_reserved = max(
        (item["peak_reserved_bytes"] for item in measurements), default=0
    )
    report = {
        "resolved_memory_profile": inference._pipeline.memory_profile,
        "startup_seconds": startup_seconds,
        "warm_run_seconds": run_seconds,
        "median_warm_seconds": (
            sorted(run_seconds)[len(run_seconds) // 2] if run_seconds else None
        ),
        "peak_allocated_gib": peak_allocated / 1024**3,
        "peak_reserved_gib": peak_reserved / 1024**3,
        "live_allocated_bytes": live_allocated_bytes,
        "mesh_face_counts": mesh_face_counts,
        "mesh_target_faces": args.mesh_target_faces,
        "flat_shading": args.flat_shading,
        "stage1_inference_steps": args.stage1_inference_steps,
        "stage2_inference_steps": args.stage2_inference_steps,
        "run_telemetry": run_telemetry,
        "telemetry_errors": telemetry.errors,
        "stages": measurements,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.write_text(rendered + "\n")
    if (
        report["resolved_memory_profile"] == "low_vram"
        and report["peak_reserved_gib"] > args.max_peak_gib
    ):
        raise SystemExit(
            f"Peak reserved VRAM {report['peak_reserved_gib']:.2f} GiB exceeded "
            f"{args.max_peak_gib:.2f} GiB"
        )
    if live_allocated_bytes:
        growth = max(live_allocated_bytes) - min(live_allocated_bytes)
        if growth > args.max_live_growth_mib * 1024**2:
            raise SystemExit(
                f"Live CUDA allocations varied by {growth / 1024**2:.2f} MiB "
                "across warm requests"
            )


if __name__ == "__main__":
    main()
