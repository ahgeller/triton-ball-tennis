"""Encoder choice, the FFmpeg command builder, VideoWriter's fallback chain and the frame reader."""
import io

import numpy as np
import pytest

from tennis_tracker import utils, video_io
from tennis_tracker.config import Config
from tennis_tracker.utils import (
    ENCODER_LIBX264,
    ENCODER_NVENC,
    ENCODER_OPENCV,
    EncoderChoice,
    choose_encoder,
    print_platform_report,
    resolve_runtime,
)
from tennis_tracker.video_io import ThreadedFrameReader, VideoWriter, ffmpeg_command
from tests._helpers import Capture


def _ffmpeg_with(monkeypatch, path, encoders):
    monkeypatch.setattr(utils, "find_ffmpeg", lambda prefer_encoder="h264_nvenc": path)
    monkeypatch.setattr(utils, "ffmpeg_has_encoder", lambda ffmpeg, encoder: bool(ffmpeg) and encoder in encoders)


def test_choose_encoder_orders_nvenc_libx264_opencv(monkeypatch):
    _ffmpeg_with(monkeypatch, "/opt/ffmpeg", {"h264_nvenc", "libx264"})
    assert choose_encoder(Config(use_nvenc=True)).chain == (ENCODER_NVENC, ENCODER_LIBX264, ENCODER_OPENCV)
    assert choose_encoder(Config(use_nvenc=True), is_cuda=False).chain == (ENCODER_LIBX264, ENCODER_OPENCV)
    assert choose_encoder(Config(use_nvenc=False)).chain == (ENCODER_OPENCV,)
    _ffmpeg_with(monkeypatch, "/opt/ffmpeg", {"libx264"})
    assert choose_encoder(Config(use_nvenc=True)).chain == (ENCODER_LIBX264, ENCODER_OPENCV)
    _ffmpeg_with(monkeypatch, None, set())
    choice = choose_encoder(Config(use_nvenc=True))
    assert choice.chain == (ENCODER_OPENCV,) and choice.ffmpeg is None and choice.ffmpeg_encoders == ()


def test_resolve_runtime_sets_the_device_on_a_copy_and_reports_the_platform(monkeypatch, capsys):
    _ffmpeg_with(monkeypatch, "/opt/ffmpeg", {"libx264"})
    cfg = Config(use_nvenc=True, device="auto")
    out, choice = resolve_runtime(cfg, "0")
    assert out is not cfg and out.use_nvenc is True
    assert out.device == "0" and cfg.device == "auto"      # the caller's Config is untouched
    assert choice.preferred == ENCODER_LIBX264
    assert not any(name.startswith("_") for name in vars(out))
    assert capsys.readouterr().out == ""                    # resolving is silent
    print_platform_report(out, choice)
    report = capsys.readouterr().out
    assert "NVENC disabled (ffmpeg lacks h264_nvenc), using libx264" in report
    assert "[platform] Device: 0" in report and "libx264 encoding" in report


def test_ffmpeg_command_differs_only_in_codec_options():
    cfg = Config(nvenc_preset="p4", nvenc_bitrate="12M")
    nvenc = ffmpeg_command("/opt/ffmpeg", ENCODER_NVENC, 1920, 1080, 59.94, "out.mp4", cfg)
    x264 = ffmpeg_command("/opt/ffmpeg", ENCODER_LIBX264, 1920, 1080, 59.94, "out.mp4", cfg)
    assert nvenc[:14] == x264[:14] == [
        "/opt/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "1920x1080", "-r", "59.94", "-i", "-",
    ]
    assert nvenc[14:] == ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "12M", "-pix_fmt", "yuv420p", "out.mp4"]
    assert x264[14:] == ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "out.mp4"]
    with pytest.raises(ValueError):
        ffmpeg_command("/opt/ffmpeg", ENCODER_OPENCV, 1, 1, 1, "x", cfg)


class RecordingPipe(io.BytesIO):
    """stdin that keeps what was written after the writer closes it."""

    data = b""

    def close(self):
        if not self.closed:
            self.data = self.getvalue()
        super().close()


class FakeProcess:
    """A Popen stand-in that either exits at once or accepts frames on stdin."""

    def __init__(self, cmd, alive, returncode=0, stderr=b""):
        self.cmd = cmd
        self.alive = alive
        self.stdin = RecordingPipe()
        self.stderr = io.BytesIO(stderr)
        self.returncode = None if alive else returncode
        self._final = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self._final
        return self.returncode

    def kill(self):
        self.returncode = -9


def _fake_popen(monkeypatch, behaviour):
    started = []

    def popen(cmd, **kwargs):
        name = cmd[cmd.index("-c:v") + 1]
        started.append(name)
        assert kwargs.get("shell") is False
        return behaviour[name](cmd)

    monkeypatch.setattr(video_io.subprocess, "Popen", popen)
    return started


def test_writer_falls_back_from_a_failing_nvenc_to_libx264(monkeypatch, capsys):
    started = _fake_popen(monkeypatch, {
        "h264_nvenc": lambda cmd: FakeProcess(cmd, alive=False, returncode=1, stderr=b"No NVENC capable devices found"),
        "libx264": lambda cmd: FakeProcess(cmd, alive=True),
    })
    choice = EncoderChoice("/opt/ffmpeg", (ENCODER_NVENC, ENCODER_LIBX264, ENCODER_OPENCV))
    writer = VideoWriter("out.mp4", 30.0, 4, 2, Config(use_async_writer=False), encoder=choice)
    assert started == ["h264_nvenc", "libx264"]
    assert writer.encoder == ENCODER_LIBX264
    assert "nvenc init failed; falling back (No NVENC capable devices found)" in capsys.readouterr().out

    frame = np.arange(4 * 2 * 3, dtype=np.uint8).reshape(2, 4, 3)
    proc = writer._proc
    writer.write(frame)
    writer.close()
    assert proc.stdin.data == frame.tobytes()
    assert proc.stdin.closed and writer._proc is None


def test_writer_reports_an_ffmpeg_failure_on_write_and_close(monkeypatch):
    _fake_popen(monkeypatch, {"libx264": lambda cmd: FakeProcess(cmd, alive=True, returncode=1, stderr=b"muxer error")})
    choice = EncoderChoice("/opt/ffmpeg", (ENCODER_LIBX264, ENCODER_OPENCV))
    writer = VideoWriter("out.mp4", 30.0, 4, 2, Config(use_async_writer=False), encoder=choice)
    writer._proc.returncode = 1  # the encoder died between frames
    with pytest.raises(RuntimeError, match="exited before frame write.*libx264.*returncode=1.*muxer error"):
        writer.write(np.zeros((2, 4, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="non-zero status.*libx264.*returncode=1"):
        writer.close()
    assert writer._proc is None  # close() always releases the process


def test_writer_uses_opencv_when_every_ffmpeg_encoder_fails(monkeypatch, tmp_path):
    _fake_popen(monkeypatch, {
        "h264_nvenc": lambda cmd: FakeProcess(cmd, alive=False, returncode=1),
        "libx264": lambda cmd: FakeProcess(cmd, alive=False, returncode=1),
    })
    out = tmp_path / "out.avi"
    choice = EncoderChoice("/opt/ffmpeg", (ENCODER_NVENC, ENCODER_LIBX264, ENCODER_OPENCV))
    writer = VideoWriter(str(out), 30.0, 64, 32, Config(use_async_writer=True, async_queue=2), encoder=choice)
    assert writer.encoder == ENCODER_OPENCV
    for _ in range(5):
        writer.write(np.zeros((32, 64, 3), dtype=np.uint8))
    writer.close()
    assert out.stat().st_size > 0


def test_writer_defaults_to_opencv_without_use_nvenc(tmp_path):
    out = tmp_path / "out.avi"
    writer = VideoWriter(str(out), 30.0, 64, 32, Config(use_nvenc=False, use_async_writer=False))
    assert writer.encoder == ENCODER_OPENCV
    writer.write(np.zeros((32, 64, 3), dtype=np.uint8))
    writer.close()
    assert out.exists()


class FailingCapture(Capture):
    """A capture whose decoder raises on frame ``fail_at``."""

    def __init__(self, count, fail_at):
        super().__init__(count)
        self.fail_at = fail_at

    def read(self):
        if self.index == self.fail_at:
            raise MemoryError("decoder out of memory")
        return super().read()


def test_frame_reader_hands_frames_over_in_order_then_none():
    reader = ThreadedFrameReader(Capture(3), prefetch=2)
    frames = [reader.read() for _ in range(3)]
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 1, 2]
    assert reader.read() is None and reader.read() is None
    reader.release()


def test_frame_reader_raises_the_decoder_error_instead_of_blocking():
    reader = ThreadedFrameReader(FailingCapture(5, fail_at=2), prefetch=2)
    assert int(reader.read()[0, 0, 0]) == 0
    assert int(reader.read()[0, 0, 0]) == 1
    with pytest.raises(RuntimeError, match="frame reader failed") as info:
        reader.read()
    assert isinstance(info.value.__cause__, MemoryError)
    assert reader.read() is None     # the reader is finished, not stuck
    reader.release()
