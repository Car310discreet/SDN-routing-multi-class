"""
video_quality.py - PSNR / SSIM scoring that survives badly damaged video.

The baseline evaluator (baseline/multimedia_evaluator.py) scores the received video with the system
FFmpeg. On Fedora that is ffmpeg-free, whose only H.264 decoder is OpenH264. OpenH264 has no error
concealment: on a stream damaged by heavy loss it decodes no frames at all, and FFmpeg aborts the
comparison ("Conversion failed!"). GEANT2's Extremely High paths damage the stream that badly.

Evaluator here is the baseline evaluator with two changes:
  * it scores with FFmpeg's native H.264 decoder, which conceals errors (rebuilds damaged frames from
    their neighbours, the usual way to score damaged video). It uses the static FFmpeg of the
    imageio-ffmpeg package, or the system FFmpeg when that is missing;
  * frames_received counts the frames actually decoded from the received stream. The PSNR filter pads
    a short stream by repeating its last frame, so counting compared frames always gives the full 4,007.
A comparison that still fails is logged and scored as no video (PSNR/SSIM None), and the run continues.
Every approach is scored with the same decoder.
"""
from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geant2_topo  # noqa: E402,F401  (puts baseline/ on the path)
from multimedia_evaluator import Evaluator as BaselineEvaluator  # noqa: E402

log = logging.getLogger(__name__)


def ffmpeg_binary() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        log.warning("imageio-ffmpeg not installed; scoring with the system ffmpeg (may fail on damaged video)")
        return "ffmpeg"


class Evaluator(BaselineEvaluator):
    ffmpeg = ffmpeg_binary()

    def decoded_frames(self, path: str) -> int:
        proc = subprocess.run([self.ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0", "-f", "null", "-"],
                              capture_output=True, text=True)
        counts = re.findall(r"frame=\s*(\d+)", proc.stderr)
        return int(counts[-1]) if counts else 0

    def video_quality(self, received_path: str, csv_path: str) -> tuple[int, float | None, float | None]:
        if not os.path.exists(received_path) or os.path.getsize(received_path) == 0:
            log.warning("No video received")
            return 0, None, None
        decoded = self.decoded_frames(received_path)
        if decoded == 0:
            log.warning("No frame of the received video could be decoded")
            return 0, None, None
        out_dir = os.path.dirname(csv_path)
        psnr_log, ssim_log = os.path.join(out_dir, "psnr.log"), os.path.join(out_dir, "ssim.log")
        graph = (f"[0:v]split[r1][r2];[1:v]split[d1][d2];"
                 f"[r1][d1]psnr=stats_file={psnr_log};[r2][d2]ssim=stats_file={ssim_log}")
        proc = subprocess.run([self.ffmpeg, "-hide_banner", "-i", self.video_path, "-i", received_path,
                               "-lavfi", graph, "-f", "null", "-"], capture_output=True, text=True)
        psnr = re.search(r"PSNR .*average:(\S+)", proc.stderr)
        ssim = re.search(r"SSIM .*All:(\S+)", proc.stderr)
        if proc.returncode != 0 or not psnr or not ssim:
            log.warning("Video quality analysis failed (%d frames decoded): %s", decoded, proc.stderr[-300:])
            return decoded, None, None
        frames = {}
        for path, key, field_name in ((psnr_log, "psnr_db", "psnr_avg"), (ssim_log, "ssim", "All")):
            with open(path) as handle:
                for line in handle:
                    fields = dict(item.split(":", 1) for item in line.split() if ":" in item)
                    frames.setdefault(int(fields["n"]), {})[key] = float(fields[field_name])
            os.remove(path)
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "psnr_db", "ssim"])
            writer.writeheader()
            writer.writerows({"frame": n, **values} for n, values in sorted(frames.items()))
        log.info("Received video: %d of %d frames decodable", decoded, len(frames))
        return decoded, float(psnr.group(1)), float(ssim.group(1))
