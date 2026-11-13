"""Final-output clock and copied-audio integrity, not an editorial score.

A retained raster can be correct while a later container transcode duplicates
or drops a cut frame. Its final video clock is rebuilt from decoded ordinals;
audio is copied directly from the original MP4, never through a rounded MKV
clock. The actual final packets must independently satisfy both contracts.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import subprocess

from .render_frame_schedule import render_frame_clock_from_probe


CONTRACT = 'final_zero_origin_ordinal_video_and_original_audio_packets_v1'


def output_rate(fps):
    if (type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0):
        raise ValueError('output media clock requires a positive finite rate')
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    if abs(float(rate) - fps) > 1e-12:
        raise ValueError('output rate cannot be represented on the declared rational clock')
    return rate


def ordinal_video_encoder_arguments(fps):
    """Every decoded input raster owns one output ordinal, regardless of PTS."""
    rate = output_rate(fps)
    return ['-vf', f'scale=out_color_matrix=bt709:out_range=tv,'
            f'settb=expr=1/{rate.numerator},setpts=N*{rate.denominator}',
        '-fps_mode', 'passthrough',
        '-enc_time_base', f'1/{rate.numerator}',
        '-video_track_timescale', str(rate.numerator),
        '-avoid_negative_ts', 'disabled']


def _identity(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _probe(path, stream, *, timeout_seconds, ffprobe):
    before = _identity(path)
    if stream == 'v:0':
        entries = ('stream=codec_name,avg_frame_rate,r_frame_rate,time_base,start_time,nb_frames:'
                   'packet=pts,duration')
        extra = []
    else:
        entries = ('stream=codec_name,sample_rate,channels,channel_layout,time_base,extradata_hash:'
                   'packet=pts,dts,duration,size,data_hash,side_data_list')
        extra = ['-show_data_hash', 'sha256']
    result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', stream,
        '-show_streams', '-show_packets', '-show_entries', entries, *extra,
        '-of', 'json', str(path)], capture_output=True, check=True,
        timeout=timeout_seconds)
    if _identity(path) != before:
        raise ValueError('output media clock input changed during packet verification')
    return json.loads(result.stdout)


def _audio_signature(payload):
    try:
        if len(payload['streams']) != 1 or not payload['packets']:
            raise ValueError('copied audio requires one complete nonempty packet stream')
        stream = payload['streams'][0]
        tb = Fraction(stream['time_base'])
        if tb <= 0:
            raise ValueError('audio time base must be positive')
        rows = []
        for packet in payload['packets']:
            size = packet['size']
            if isinstance(size, str) and size.isascii() and size.isdigit(): size = int(size)
            values = [packet[k] for k in ('pts', 'dts', 'duration')] + [size]
            if any(type(v) is not int for v in values) or values[2] <= 0 or values[3] <= 0:
                raise ValueError('audio packet timing or payload size is unmeasured')
            digest = packet['data_hash']
            if (not isinstance(digest, str) or not digest.startswith('SHA256:')
                    or len(digest) != 71):
                raise ValueError('audio payload needs an actual packet hash')
            rows.append(dict(pts=str(packet['pts'] * tb), dts=str(packet['dts'] * tb),
                duration=str(packet['duration'] * tb), size=size, data_hash=digest,
                side_data_list=packet.get('side_data_list', [])))
        return dict(codec={k: stream.get(k) for k in
            ('codec_name', 'sample_rate', 'channels', 'channel_layout', 'extradata_hash')}, packets=rows)
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise ValueError('copied audio packet evidence is incomplete') from error


def verify_final_output_media_clock(*, movie, original_audio_movie, frame_count, fps,
        timeout_seconds=20., ffprobe='ffprobe'):
    if (type(frame_count) is not int or frame_count < 2
            or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('final output verification needs an exact frame count and bounded deadline')
    rate = output_rate(fps)
    video = _probe(movie, 'v:0', timeout_seconds=timeout_seconds, ffprobe=ffprobe)
    clock = render_frame_clock_from_probe(video)
    if clock is None or clock.frame_count != frame_count or clock.frame_rate != rate:
        raise ValueError('final video lost its zero-origin complete ordinal presentation clock')
    original = _audio_signature(_probe(original_audio_movie, 'a:0',
        timeout_seconds=timeout_seconds, ffprobe=ffprobe))
    delivered = _audio_signature(_probe(movie, 'a:0',
        timeout_seconds=timeout_seconds, ffprobe=ffprobe))
    if original != delivered:
        raise ValueError('final copied audio changed packet payload, timing, or preroll/skip samples')
    audio_digest = hashlib.sha256(json.dumps(original, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return dict(contract=CONTRACT, movie=str(movie), original_audio_movie=str(original_audio_movie),
        final_zero_origin_video_clock=clock.to_dict(), original_audio_packet_count=len(original['packets']),
        original_audio_packet_signature_sha256=audio_digest,
        original_audio_payload_and_timing_equal=True, original_audio_preroll_and_skip_samples_equal=True,
        encoded_raster_content_identity_claimed=False, editorial_pose_match_approved=False,
        quality_approved=False)
