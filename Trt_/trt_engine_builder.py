import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ultralytics import YOLO

#python Trt_/trt_engine_builder.py --ball-pt models/ball.pt --player-pt models/player.pt --court-pt models/courtdetection.pt --half --require-fp16 --force-onnx --force-engine --device 0 --workspace-mb 4096 --include-court

@dataclass
class ExportTarget:
    label: str
    pt_path: Path
    imgsz: int

    @property
    def onnx_path(self) -> Path:
        return self.pt_path.with_suffix(".onnx")

    @property
    def engine_path(self) -> Path:
        return self.pt_path.with_suffix(".engine")


def _strip_ultralytics_engine_header(raw: bytes) -> bytes:
    try:
        meta_len = int.from_bytes(raw[:4], byteorder="little")
        _ = json.loads(raw[4:4 + meta_len].decode("utf-8"))
        return raw[4 + meta_len:]
    except Exception:
        return raw


def _inspect_engine_io_dtypes(engine_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    try:
        import tensorrt as trt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"TensorRT package not importable for verification: {e}") from e

    raw = _strip_ultralytics_engine_header(engine_path.read_bytes())
    logger = trt.Logger(trt.Logger.ERROR)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(raw)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine for verification: {engine_path}")

    inputs: Dict[str, str] = {}
    outputs: Dict[str, str] = {}
    is_trt10 = not hasattr(engine, "num_bindings")
    if is_trt10:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = str(engine.get_tensor_dtype(name))
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs[name] = dtype
            else:
                outputs[name] = dtype
    else:
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = str(engine.get_binding_dtype(i))
            if engine.binding_is_input(i):
                inputs[name] = dtype
            else:
                outputs[name] = dtype
    return inputs, outputs


def _verify_engine_precision(
    engine_path: Path,
    *,
    require_fp16: bool,
    strict_fp16_io: bool,
    label: str,
) -> None:
    if not engine_path.exists():
        raise FileNotFoundError(f"[{label}] Missing engine for verification: {engine_path}")
    inputs, outputs = _inspect_engine_io_dtypes(engine_path)
    print(f"[{label}] Engine IO dtypes: inputs={inputs}, outputs={outputs}")

    if not require_fp16:
        return

    def _is_fp16_dtype(dt_obj) -> bool:
        s = str(dt_obj).upper()
        return ("FLOAT16" in s) or ("HALF" in s)

    has_fp16_input = any(_is_fp16_dtype(dt) for dt in inputs.values())
    has_fp16_output = any(_is_fp16_dtype(dt) for dt in outputs.values())
    has_any_fp16_io = has_fp16_input or has_fp16_output

    if strict_fp16_io and not (has_fp16_input and has_fp16_output):
        raise RuntimeError(
            f"[{label}] Engine IO is not fully FP16 (strict-fp16-io enabled): {engine_path}"
        )
    if not strict_fp16_io and not has_any_fp16_io:
        print(
            f"[{label}] Note: engine IO bindings are FP32. "
            "Internal kernels may still use FP16 tactics."
        )


def _build_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    strict_fp16_io: bool,
    workspace_mb: int,
    label: str,
) -> None:
    try:
        import tensorrt as trt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"TensorRT package not importable for build: {e}") from e

    if not onnx_path.exists():
        raise FileNotFoundError(f"[{label}] ONNX not found for TRT build: {onnx_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    raw = onnx_path.read_bytes()
    if not parser.parse(raw):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(
            f"[{label}] ONNX parse failed for {onnx_path}:\n" + "\n".join(errs[:25])
        )

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * 1024 * 1024
        )
    else:
        config.max_workspace_size = int(workspace_mb) * 1024 * 1024

    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError(f"[{label}] FP16 requested but platform_has_fast_fp16=False")
        config.set_flag(trt.BuilderFlag.FP16)

    dynamic = False
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if any(int(d) < 0 for d in inp.shape):
            dynamic = True
            break

    if dynamic:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            name = inp.name
            shape = tuple(int(d) for d in inp.shape)
            if len(shape) != 4:
                raise RuntimeError(
                    f"[{label}] Dynamic input '{name}' rank {len(shape)} unsupported (need NCHW rank-4)."
                )
            n = shape[0] if shape[0] > 0 else 1
            c = shape[1] if shape[1] > 0 else 3
            h = shape[2] if shape[2] > 0 else 640
            w = shape[3] if shape[3] > 0 else 640
            fixed = (int(n), int(c), int(h), int(w))
            profile.set_shape(name, fixed, fixed, fixed)
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"[{label}] TensorRT build returned None for {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(
        f"[{label}] Built TensorRT engine from ONNX: {engine_path} "
        f"(fp16={bool(fp16)}, workspace={int(workspace_mb)}MB)"
    )


def _ensure_target(
    target: ExportTarget,
    *,
    device: str,
    half: bool,
    force_onnx: bool,
    force_engine: bool,
    require_fp16: bool,
    strict_fp16_io: bool,
    workspace_mb: int,
) -> Tuple[bool, bool, bool]:
    if not target.pt_path.exists():
        print(f"[{target.label}] Skip: missing {target.pt_path}")
        return False, False, True

    need_onnx = force_onnx or not target.onnx_path.exists()
    need_engine = force_engine or not target.engine_path.exists()

    if not need_onnx and not need_engine:
        print(f"[{target.label}] OK: {target.onnx_path.name} + {target.engine_path.name} already exist")
        _verify_engine_precision(
            target.engine_path,
            require_fp16=require_fp16,
            strict_fp16_io=strict_fp16_io,
            label=target.label,
        )
        return False, False, False

    built_onnx = False
    built_engine = False

    if need_onnx:
        print(
            f"[{target.label}] Export ONNX -> {target.onnx_path} "
            f"(imgsz={target.imgsz}, device={device}, half={bool(half)})"
        )
        model = YOLO(str(target.pt_path))
        out = model.export(
            format="onnx",
            imgsz=int(target.imgsz),
            device=device,
            half=bool(half),
            simplify=True,
        )
        # Ultralytics may return a different object/path; ensure expected ONNX path exists.
        if isinstance(out, (str, Path)):
            out_path = Path(out)
            if out_path.exists() and out_path.resolve() != target.onnx_path.resolve():
                target.onnx_path.parent.mkdir(parents=True, exist_ok=True)
                target.onnx_path.write_bytes(out_path.read_bytes())
        if not target.onnx_path.exists():
            raise RuntimeError(f"[{target.label}] ONNX export did not produce {target.onnx_path}")
        built_onnx = True

    if need_engine:
        _build_engine_from_onnx(
            target.onnx_path,
            target.engine_path,
            fp16=bool(half),
            strict_fp16_io=bool(strict_fp16_io),
            workspace_mb=int(workspace_mb),
            label=target.label,
        )
        built_engine = True
        _verify_engine_precision(
            target.engine_path,
            require_fp16=require_fp16,
            strict_fp16_io=strict_fp16_io,
            label=target.label,
        )

    return built_onnx, built_engine, False


def main():
    p = argparse.ArgumentParser(
        description="Build ONNX + TensorRT engines with enforced FP16 options."
    )
    p.add_argument("--ball-pt", default="models/ball.pt", help="Ball checkpoint path (.pt)")
    p.add_argument("--player-pt", default="models/player.pt", help="Player checkpoint path (.pt)")
    p.add_argument("--court-pt", default="models/courtdetection.pt", help="Court checkpoint path (.pt)")
    p.add_argument("--imgsz-ball", type=int, default=1280, help="Ball export size (default: 1280)")
    p.add_argument("--imgsz-player", type=int, default=640, help="Player export size (default: 640)")
    p.add_argument("--imgsz-court", type=int, default=1280, help="Court export size (default: 1280)")
    p.add_argument("--device", default="0", help="Export device passed to Ultralytics (default: 0)")
    p.add_argument("--workspace-mb", type=int, default=4096, help="TensorRT workspace MB (default: 4096)")
    p.add_argument("--half", dest="half", action="store_true", help="Use FP16 ONNX export + FP16 TRT build (default)")
    p.add_argument("--no-half", dest="half", action="store_false", help="Use FP32 export/build")
    p.add_argument("--require-fp16", dest="require_fp16", action="store_true", help="Enforce FP16 verification (default)")
    p.add_argument("--no-require-fp16", dest="require_fp16", action="store_false", help="Disable FP16 verification")
    p.add_argument(
        "--strict-fp16-io",
        action="store_true",
        help="Require both input and output bindings to be FP16",
    )
    p.add_argument("--force-onnx", action="store_true", help="Rebuild ONNX even if it exists")
    p.add_argument("--force-engine", action="store_true", help="Rebuild engine even if it exists")
    p.add_argument(
        "--include-court",
        dest="include_court",
        action="store_true",
        help="Also build court ONNX/engine (default: on)",
    )
    p.add_argument(
        "--no-include-court",
        dest="include_court",
        action="store_false",
        help="Skip court ONNX/engine build",
    )
    p.set_defaults(half=True, require_fp16=True, strict_fp16_io=False, include_court=True)
    args = p.parse_args()

    targets = [
        ExportTarget("ball", Path(args.ball_pt), int(args.imgsz_ball)),
        ExportTarget("player", Path(args.player_pt), int(args.imgsz_player)),
    ]
    if args.include_court:
        targets.append(ExportTarget("court", Path(args.court_pt), int(args.imgsz_court)))

    summary: Dict[str, int] = {"built_onnx": 0, "built_engine": 0, "skipped_missing": 0}
    for target in targets:
        built_onnx, built_engine, skipped_missing = _ensure_target(
            target,
            device=str(args.device),
            half=bool(args.half),
            force_onnx=bool(args.force_onnx),
            force_engine=bool(args.force_engine),
            require_fp16=bool(args.require_fp16),
            strict_fp16_io=bool(args.strict_fp16_io),
            workspace_mb=max(256, int(args.workspace_mb)),
        )
        summary["built_onnx"] += int(built_onnx)
        summary["built_engine"] += int(built_engine)
        summary["skipped_missing"] += int(skipped_missing)

    print(
        "[done] built_onnx={built_onnx}, built_engine={built_engine}, skipped_missing={skipped_missing}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
