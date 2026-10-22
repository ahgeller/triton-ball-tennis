"""Encoder choice, the FFmpeg command builder, VideoWriter's fallback chain and the frame reader."""
import io
import threading

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
from tennis_tracker.video_io import (
    VIDEO_SUFFIXES,
    ThreadedFrameReader,
    VideoWriter,
    check_writable_video_path,
    ffmpeg_command,
)
from tests._helpers import Capture, FailingWriter


@pytest.mark.parametrize("suffix", VIDEO_SUFFIXES + (".MP4",))
def test_writable_video_path_accepts_a_muxable_suffix_in_a_folder_to_create(tmp_path, suffix):
    # The writers create missing parents when they open, so a path under a
    # folder that does not exist yet is fine; the suffix is matched case-insensitively.
    check_writable_video_path(tmp_path / "later" / "deeper" / f"tracking{suffix}")
    check_writable_video_path(str(tmp_path / f"tracking{suffix}"))


def test_writable_video_path_rejects_a_directory(tmp_path):
    # The user's case: --output <folder> reached FFmpeg, which refused it after pass 1.
    with pytest.raises(ValueError, match=r"Tracking video .*is a directory; name a file in it, such as .*tracking\.mp4"):
        check_writable_video_path(tmp_path, "Tracking video")


@pytest.mark.parametrize("name", ["tracking", "tracking.txt", "tracking.webm", ".mp4"])
def test_writable_video_path_rejects_a_suffix_ffmpeg_cannot_mux(tmp_path, name):
    with pytest.raises(ValueError, match=r"Video output .*needs a video container suffix.*\.mp4, \.mkv, \.mov, \.avi"):
        check_writable_video_path(tmp_path / name)


def test_writable_video_path_rejects_a_parent_that_is_a_file(tmp_path):
    (tmp_path / "taken").write_bytes(b"")
    with pytest.raises(ValueError, match=r"cannot be written: .*taken is not a directory"):
        check_writable_video_path(tmp_path / "taken" / "deeper" / "tracking.mp4")


def _ffmpeg_with(monkeypatch, path, encoders, failing=()):
    """An FFmpeg at `path` listing `encoders`, of which `failing` cannot encode when tried.
    Returns the list the dry run appends each encoder it is asked about to."""
    dry_runs = []

    def encoder_failure(ffmpeg, encoder):
        dry_runs.append(encoder)
        return f"{encoder} refused the test frame" if encoder in failing else None

    monkeypatch.setattr(utils, "find_ffmpeg", lambda prefer_encoder="h264_nvenc": path)
    monkeypatch.setattr(utils, "ffmpeg_has_encoder", lambda ffmpeg, encoder: bool(ffmpeg) and encoder in encoders)
    monkeypatch.setattr(utils, "ffmpeg_encoder_failure", encoder_failure)
    return dry_runs


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


def test_choose_encoder_drops_an_nvenc_that_lists_but_cannot_encode(monkeypatch, capsys):
    # FFmpeg lists h264_nvenc on a box whose driver it cannot load (or whose sessions are all
    # taken); the encoder only fails at the first frame, after pass 1 and selection. The dry run
    # finds out first, the chain starts at libx264 and the platform report says why.
    dry_runs = _ffmpeg_with(monkeypatch, "/opt/ffmpeg", {"h264_nvenc", "libx264"}, failing={"h264_nvenc"})
    choice = choose_encoder(Config(use_nvenc=True))
    assert choice.chain == (ENCODER_LIBX264, ENCODER_OPENCV)
    assert choice.nvenc_failure == "h264_nvenc refused the test frame"
    assert dry_runs == ["h264_nvenc"]                                   # only NVENC needs the dry run
    print_platform_report(Config(use_nvenc=True, device="0"), choice)
    assert "NVENC disabled (h264_nvenc cannot encode here: h264_nvenc refused the test frame), using libx264" \
        in capsys.readouterr().out
    assert choose_encoder(Config(use_nvenc=True), is_cuda=False).nvenc_failure is None   # not even tried without CUDA
    assert dry_runs == ["h264_nvenc"]


def test_encoder_dry_run_reports_the_last_stderr_line_and_remembers_the_answer(monkeypatch):
    calls = []

    class Run:
        def __init__(self, returncode, stderr=""):
            self.returncode, self.stderr = returncode, stderr

    answers = {"h264_nvenc": Run(127, "[h264_nvenc @ 0x1] Cannot load libcuda.so.1\n[vost#0:0] Could not open encoder before EOF\n"),
               "libx264": Run(0)}

    def run(command, **kwargs):
        calls.append(command)
        assert command[:2] == ["/opt/ffmpeg", "-hide_banner"] and command[-3:] == ["-f", "null", "-"]
        assert "lavfi" in command and kwargs.get("shell") is False and kwargs.get("timeout")
        return answers[command[command.index("-c:v") + 1]]

    monkeypatch.setattr(utils.subprocess, "run", run)
    monkeypatch.setattr(utils, "_FFMPEG_DRY_RUN_CACHE", {})
    assert utils.ffmpeg_encoder_failure("/opt/ffmpeg", "h264_nvenc") == "[vost#0:0] Could not open encoder before EOF"
    assert utils.ffmpeg_encoder_failure("/opt/ffmpeg", "libx264") is None
    assert utils.ffmpeg_encoder_failure("/opt/ffmpeg", "h264_nvenc") == "[vost#0:0] Could not open encoder before EOF"
    assert len(calls) == 2                                                   # one dry run per encoder per process

    def broken(command, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(utils.subprocess, "run", broken)
    monkeypatch.setattr(utils, "_FFMPEG_DRY_RUN_CACHE", {})
    assert utils.ffmpeg_encoder_failure("/opt/ffmpeg", "libx264") == "could not run ffmpeg (no such file)"


def test_resolve_runtime_sets_the_device_on_a_copy_and_reports_the_platform(monkeypatch, capsys):
    _ffmpeg_with(monkeypatch, "/opt/ffmpeg", {"libx264"})
    cfg = Config(use_nvenc=True, device="auto")
    out, choice = resolve_runtime(cfg, "0", writes_video=True)
    assert out is not cfg and out.use_nvenc is True
    assert out.device == "0" and cfg.device == "auto"      # the caller's Config is untouched
    assert choice.preferred == ENCODER_LIBX264
    assert not any(name.startswith("_") for name in vars(out))
    assert capsys.readouterr().out == ""                    # resolving is silent
    print_platform_report(out, choice)
    report = capsys.readouterr().out
    assert "NVENC disabled (ffmpeg lacks h264_nvenc), using libx264" in report
    assert "[platform] Device: 0" in report and "libx264 encoding" in report


def test_resolve_runtime_chooses_no_encoder_and_runs_no_ffmpeg_when_no_video_is_written(monkeypatch, capsys):
    # A --no-video run and every headless evaluation run write only the JSON; the NVENC dry run
    # cost each of them a second for an encoder nothing would use.
    dry_runs = _ffmpeg_with(monkeypatch, "/opt/ffmpeg", {"h264_nvenc", "libx264"})
    monkeypatch.setattr(utils, "find_ffmpeg", lambda prefer_encoder="h264_nvenc": pytest.fail("looked for ffmpeg"))
    out, choice = resolve_runtime(Config(use_nvenc=True, device="auto"), "0", writes_video=False)
    assert out.device == "0" and choice is None and dry_runs == []
    print_platform_report(out, choice)
    report = capsys.readouterr().out
    assert "[platform] Device: 0" in report and "no video written" in report
    assert "encoding" not in report and "NVENC disabled" not in report


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


def test_writer_failure_does_not_block_full_queue(tmp_path):
    writer = FailingWriter(str(tmp_path / "out.avi"), Config(use_async_writer=True, async_queue=1))
    frame = np.zeros((32, 64, 3), np.uint8)
    writer.write(frame)                # taken by the drain thread, which then blocks
    assert writer.entered.wait(1)
    writer.write(frame)                # fills the one-slot queue
    writer.release.set()               # the encoder now fails with the queue still full
    assert writer.drained.wait(1)
    errors = []

    def close():
        try:
            writer.close()
        except RuntimeError as error:
            errors.append(str(error))
    closer = threading.Thread(target=close, daemon=True)
    closer.start()
    closer.join(1)
    assert not closer.is_alive()
    assert "encoder failed" in errors[0]
