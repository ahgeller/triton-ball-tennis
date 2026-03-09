import json
from pathlib import Path

import tensorrt as trt


def strip_ultralytics_header(raw: bytes) -> bytes:
    try:
        meta_len = int.from_bytes(raw[:4], "little")
        _ = json.loads(raw[4:4 + meta_len].decode("utf-8"))
        return raw[4 + meta_len:]
    except Exception:
        return raw


def inspect_engine(path: Path):
    print(f"== {path} ==")
    if not path.exists():
        print("missing")
        return

    raw = strip_ultralytics_header(path.read_bytes())
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    eng = rt.deserialize_cuda_engine(raw)
    if eng is None:
        print("deserialize failed")
        return

    is_trt10 = not hasattr(eng, "num_bindings")
    if is_trt10:
        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            dtype = str(eng.get_tensor_dtype(name))
            shape = tuple(int(x) for x in eng.get_tensor_shape(name))
            mode = eng.get_tensor_mode(name)
            kind = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
            print(f"{kind:6} name={name} dtype={dtype} shape={shape}")
    else:
        for i in range(eng.num_bindings):
            name = eng.get_binding_name(i)
            dtype = str(eng.get_binding_dtype(i))
            shape = tuple(int(x) for x in eng.get_binding_shape(i))
            kind = "INPUT" if eng.binding_is_input(i) else "OUTPUT"
            print(f"{kind:6} name={name} dtype={dtype} shape={shape}")
    print()


def main():
    for p in (
        Path("models/ball.engine"),
        Path("models/player.engine"),
        Path("models/courtdetection.engine"),
    ):
        inspect_engine(p)


if __name__ == "__main__":
    main()
