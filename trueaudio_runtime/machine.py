from __future__ import annotations

import ctypes
import sys
import time
import uuid
from ctypes import wintypes
from typing import Any

import numpy as np


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "GUID":
        parsed = uuid.UUID(value)
        fields = parsed.fields
        data4 = parsed.bytes[8:]
        return cls(
            fields[0],
            fields[1],
            fields[2],
            (ctypes.c_ubyte * 8).from_buffer_copy(data4),
        )

    def as_uuid(self) -> uuid.UUID:
        node = int.from_bytes(bytes(self.Data4[2:]), "big")
        return uuid.UUID(fields=(self.Data1, self.Data2, self.Data3, self.Data4[0], self.Data4[1], node))


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class WAVEFORMATEXTENSIBLE(ctypes.Structure):
    _fields_ = WAVEFORMATEX._fields_ + [
        ("wValidBitsPerSample", wintypes.WORD),
        ("dwChannelMask", wintypes.DWORD),
        ("SubFormat", GUID),
    ]


CLSCTX_ALL = 23
COINIT_MULTITHREADED = 0
RPC_E_CHANGED_MODE = 0x80010106

E_RENDER = 0
E_CONSOLE = 0

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_BUFFERFLAGS_SILENT = 0x00000002

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

SUBTYPE_PCM = "00000001-0000-0010-8000-00aa00389b71"
SUBTYPE_IEEE_FLOAT = "00000003-0000-0010-8000-00aa00389b71"

CLSID_MMDEVICE_ENUMERATOR = GUID.from_string("bcde0395-e52f-467c-8e3d-c4579291692e")
IID_IMMDEVICE_ENUMERATOR = GUID.from_string("a95664d2-9614-4f35-a746-de8db63617e6")
IID_IAUDIO_CLIENT = GUID.from_string("1cb9ad4c-dbfa-4c32-b178-c2f568a703b2")
IID_IAUDIO_CAPTURE_CLIENT = GUID.from_string("c8adbd64-e71e-48a0-a4de-185c395cd317")


def _hresult_failed(hr: int) -> bool:
    return ctypes.c_long(hr).value < 0


def _check_hresult(hr: int, action: str) -> None:
    if _hresult_failed(hr):
        raise OSError(f"{action} failed with HRESULT 0x{hr & 0xFFFFFFFF:08x}")


def _com_method(ptr: ctypes.c_void_p, index: int, restype: Any, *argtypes: Any) -> Any:
    vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    function = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])

    def bound(*args: Any) -> Any:
        return function(ptr, *args)

    return bound


def _release(ptr: ctypes.c_void_p | None) -> None:
    if ptr:
        _com_method(ptr, 2, ctypes.c_ulong)()


def _wave_subtype(format_ptr: ctypes.c_void_p, format_tag: int) -> str:
    if format_tag == WAVE_FORMAT_PCM:
        return SUBTYPE_PCM
    if format_tag == WAVE_FORMAT_IEEE_FLOAT:
        return SUBTYPE_IEEE_FLOAT
    if format_tag == WAVE_FORMAT_EXTENSIBLE:
        extensible = ctypes.cast(format_ptr, ctypes.POINTER(WAVEFORMATEXTENSIBLE)).contents
        return str(extensible.SubFormat.as_uuid()).lower()
    return ""


def _pcm_bytes_to_float32(raw: bytes, *, channels: int, bits_per_sample: int, subtype: str) -> np.ndarray:
    if not raw:
        return np.zeros((0, channels), dtype=np.float32)
    subtype = subtype.lower()
    if subtype == SUBTYPE_IEEE_FLOAT and bits_per_sample == 32:
        data = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif subtype == SUBTYPE_PCM and bits_per_sample == 16:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif subtype == SUBTYPE_PCM and bits_per_sample == 24:
        byte_values = np.frombuffer(raw, dtype=np.uint8).reshape((-1, 3))
        signed = (
            byte_values[:, 0].astype(np.int32)
            | (byte_values[:, 1].astype(np.int32) << 8)
            | (byte_values[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed | ~0xFFFFFF, signed)
        data = signed.astype(np.float32) / 8388608.0
    elif subtype == SUBTYPE_PCM and bits_per_sample == 32:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported WASAPI mix format subtype={subtype} bits={bits_per_sample}")

    usable = (data.size // channels) * channels
    if usable != data.size:
        data = data[:usable]
    return np.clip(data.reshape((-1, channels)), -1.0, 1.0)


def _to_stereo(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if samples.shape[1] == 1:
        return np.repeat(samples, 2, axis=1).astype(np.float32)
    if samples.shape[1] >= 2:
        return samples[:, :2].astype(np.float32)
    return np.zeros((samples.shape[0], 2), dtype=np.float32)


def capture_windows_wasapi_loopback(*, duration_seconds: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Capture the default Windows render endpoint through WASAPI loopback.

    This samples the machine output mix before it reaches the physical speaker
    device. It returns float32 stereo state input and metadata; callers decide
    whether to save derived state, not raw PCM.
    """
    if sys.platform != "win32":
        raise RuntimeError("Windows WASAPI loopback capture is only available on Windows")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")

    ole32 = ctypes.OleDLL("ole32")
    initialized = False
    hr = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
    if hr == 0:
        initialized = True
    elif (hr & 0xFFFFFFFF) != RPC_E_CHANGED_MODE:
        _check_hresult(hr, "CoInitializeEx")

    enumerator = ctypes.c_void_p()
    device = ctypes.c_void_p()
    audio_client = ctypes.c_void_p()
    capture_client = ctypes.c_void_p()
    format_ptr = ctypes.c_void_p()

    try:
        _check_hresult(
            ole32.CoCreateInstance(
                ctypes.byref(CLSID_MMDEVICE_ENUMERATOR),
                None,
                CLSCTX_ALL,
                ctypes.byref(IID_IMMDEVICE_ENUMERATOR),
                ctypes.byref(enumerator),
            ),
            "CoCreateInstance(MMDeviceEnumerator)",
        )

        get_default_audio_endpoint = _com_method(
            enumerator,
            4,
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        )
        _check_hresult(
            get_default_audio_endpoint(E_RENDER, E_CONSOLE, ctypes.byref(device)),
            "GetDefaultAudioEndpoint",
        )

        activate = _com_method(
            device,
            3,
            ctypes.c_long,
            ctypes.POINTER(GUID),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        _check_hresult(
            activate(ctypes.byref(IID_IAUDIO_CLIENT), CLSCTX_ALL, None, ctypes.byref(audio_client)),
            "IMMDevice.Activate(IAudioClient)",
        )

        get_mix_format = _com_method(audio_client, 8, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p))
        _check_hresult(get_mix_format(ctypes.byref(format_ptr)), "IAudioClient.GetMixFormat")
        wave_format = ctypes.cast(format_ptr, ctypes.POINTER(WAVEFORMATEX)).contents
        sample_rate = int(wave_format.nSamplesPerSec)
        channels = int(wave_format.nChannels)
        bits_per_sample = int(wave_format.wBitsPerSample)
        block_align = int(wave_format.nBlockAlign)
        subtype = _wave_subtype(format_ptr, int(wave_format.wFormatTag))

        initialize = _com_method(
            audio_client,
            3,
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        buffer_duration_hns = int(min(max(duration_seconds, 0.1), 1.0) * 10_000_000)
        _check_hresult(
            initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                buffer_duration_hns,
                0,
                format_ptr,
                None,
            ),
            "IAudioClient.Initialize(loopback)",
        )

        get_service = _com_method(
            audio_client,
            14,
            ctypes.c_long,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )
        _check_hresult(
            get_service(ctypes.byref(IID_IAUDIO_CAPTURE_CLIENT), ctypes.byref(capture_client)),
            "IAudioClient.GetService(IAudioCaptureClient)",
        )

        start = _com_method(audio_client, 10, ctypes.c_long)
        stop = _com_method(audio_client, 11, ctypes.c_long)
        get_next_packet_size = _com_method(
            capture_client,
            5,
            ctypes.c_long,
            ctypes.POINTER(wintypes.UINT),
        )
        get_buffer = _com_method(
            capture_client,
            3,
            ctypes.c_long,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(wintypes.UINT),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        )
        release_buffer = _com_method(capture_client, 4, ctypes.c_long, wintypes.UINT)

        chunks: list[np.ndarray] = []
        _check_hresult(start(), "IAudioClient.Start")
        try:
            deadline = time.perf_counter() + duration_seconds
            packet_size = wintypes.UINT(0)
            while time.perf_counter() < deadline:
                _check_hresult(get_next_packet_size(ctypes.byref(packet_size)), "GetNextPacketSize")
                while packet_size.value:
                    data_ptr = ctypes.POINTER(ctypes.c_ubyte)()
                    frames_to_read = wintypes.UINT(0)
                    flags = wintypes.DWORD(0)
                    device_position = ctypes.c_ulonglong(0)
                    qpc_position = ctypes.c_ulonglong(0)
                    _check_hresult(
                        get_buffer(
                            ctypes.byref(data_ptr),
                            ctypes.byref(frames_to_read),
                            ctypes.byref(flags),
                            ctypes.byref(device_position),
                            ctypes.byref(qpc_position),
                        ),
                        "IAudioCaptureClient.GetBuffer",
                    )
                    try:
                        if flags.value & AUDCLNT_BUFFERFLAGS_SILENT:
                            chunk = np.zeros((frames_to_read.value, channels), dtype=np.float32)
                        else:
                            byte_count = int(frames_to_read.value) * block_align
                            raw = ctypes.string_at(data_ptr, byte_count)
                            chunk = _pcm_bytes_to_float32(
                                raw,
                                channels=channels,
                                bits_per_sample=bits_per_sample,
                                subtype=subtype,
                            )
                        chunks.append(_to_stereo(chunk))
                    finally:
                        _check_hresult(release_buffer(frames_to_read), "IAudioCaptureClient.ReleaseBuffer")
                    _check_hresult(get_next_packet_size(ctypes.byref(packet_size)), "GetNextPacketSize")
                time.sleep(0.005)
        finally:
            stop()

        samples = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), dtype=np.float32)
        target_frames = int(round(duration_seconds * sample_rate))
        if samples.shape[0] < target_frames:
            pad = np.zeros((target_frames - samples.shape[0], 2), dtype=np.float32)
            samples = np.concatenate([samples, pad], axis=0)
        elif samples.shape[0] > target_frames:
            samples = samples[:target_frames]

        return samples.astype(np.float32), {
            "backend": "windows_wasapi_loopback",
            "sample_rate": sample_rate,
            "channels": 2,
            "native_channels": channels,
            "native_bits_per_sample": bits_per_sample,
            "native_subtype": subtype,
            "device_role": "default_render_endpoint_loopback",
        }
    finally:
        if format_ptr:
            ole32.CoTaskMemFree(format_ptr)
        _release(capture_client)
        _release(audio_client)
        _release(device)
        _release(enumerator)
        if initialized:
            ole32.CoUninitialize()
