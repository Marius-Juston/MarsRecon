"""Tests for the async download pipeline in src/temp.py.

Uses aioresponses to mock aiohttp.ClientSession without touching the network.
All tests are async and run automatically via asyncio_mode = "auto" in pyproject.toml.
"""

import pathlib
import sys
import threading

import aiohttp
import pytest
from aioresponses import aioresponses

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mars_hirise import _download_file

_URL = "https://hirise-pds.lpl.arizona.edu/PDS/test/PSP_001430_1780_RED.JP2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stop_event() -> threading.Event:
    return threading.Event()


@pytest.fixture
def dest(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "PSP_001430_1780_RED.JP2"


def _tmp(dest: pathlib.Path) -> pathlib.Path:
    return dest.with_name(f".tmp.{dest.name}")


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestDownloadFileSuccess:
    async def test_creates_final_file(self, dest, stop_event):
        with aioresponses() as m:
            m.get(_URL, body=b"hello jp2 data", status=200)
            async with aiohttp.ClientSession() as session:
                await _download_file(session, _URL, dest, stop_event)
        assert dest.exists()
        assert dest.read_bytes() == b"hello jp2 data"

    async def test_no_tmp_file_after_success(self, dest, stop_event):
        with aioresponses() as m:
            m.get(_URL, body=b"data", status=200)
            async with aiohttp.ClientSession() as session:
                await _download_file(session, _URL, dest, stop_event)
        assert not _tmp(dest).exists()

    async def test_skips_if_file_exists(self, dest, stop_event):
        dest.write_bytes(b"existing")
        with aioresponses() as m:
            async with aiohttp.ClientSession() as session:
                await _download_file(session, _URL, dest, stop_event)
            # No request should have been made.
            assert len(m.requests) == 0
        assert dest.read_bytes() == b"existing"

    async def test_content_length_match_succeeds(self, dest, stop_event):
        body = b"exact content"
        with aioresponses() as m:
            m.get(_URL, body=body, status=200, headers={"Content-Length": str(len(body))})
            async with aiohttp.ClientSession() as session:
                await _download_file(session, _URL, dest, stop_event)
        assert dest.exists()
        assert dest.read_bytes() == body


# ---------------------------------------------------------------------------
# Stop-event tests
# ---------------------------------------------------------------------------


class TestDownloadFileStopEvent:
    async def test_stop_before_start_makes_no_request(self, dest, stop_event):
        stop_event.set()
        with aioresponses() as m:
            async with aiohttp.ClientSession() as session:
                await _download_file(session, _URL, dest, stop_event)
            assert len(m.requests) == 0
        assert not dest.exists()
        assert not _tmp(dest).exists()


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


class TestDownloadFileRetry:
    async def test_retry_on_429(self, dest, stop_event):
        with aioresponses() as m:
            m.get(_URL, status=429)
            m.get(_URL, body=b"ok", status=200)
            async with aiohttp.ClientSession() as session:
                await _download_file(
                    session, _URL, dest, stop_event,
                    max_retries=2, base_delay=0.0,
                )
        assert dest.exists()

    async def test_retry_on_503(self, dest, stop_event):
        with aioresponses() as m:
            m.get(_URL, status=503)
            m.get(_URL, body=b"ok", status=200)
            async with aiohttp.ClientSession() as session:
                await _download_file(
                    session, _URL, dest, stop_event,
                    max_retries=2, base_delay=0.0,
                )
        assert dest.exists()

    async def test_no_retry_on_404(self, dest, stop_event):
        with aioresponses() as m:
            m.get(_URL, status=404)
            async with aiohttp.ClientSession() as session:
                await _download_file(
                    session, _URL, dest, stop_event,
                    max_retries=3, base_delay=0.0,
                )
            # Only one request should have been attempted.
            assert len(m.requests.get(("GET", aiohttp.client.URL(_URL)), [])) == 1
        assert not dest.exists()

    async def test_no_retry_on_400(self, dest, stop_event):
        with aioresponses() as m:
            m.get(_URL, status=400)
            async with aiohttp.ClientSession() as session:
                await _download_file(
                    session, _URL, dest, stop_event,
                    max_retries=3, base_delay=0.0,
                )
        assert not dest.exists()

    async def test_exhausted_retries_leaves_no_file(self, dest, stop_event):
        with aioresponses() as m:
            for _ in range(4):
                m.get(_URL, status=503)
            async with aiohttp.ClientSession() as session:
                await _download_file(
                    session, _URL, dest, stop_event,
                    max_retries=3, base_delay=0.0,
                )
        assert not dest.exists()
        assert not _tmp(dest).exists()


# ---------------------------------------------------------------------------
# Content-Length mismatch
# ---------------------------------------------------------------------------


class TestContentLengthValidation:
    async def test_mismatch_retries_and_eventually_succeeds(self, dest, stop_event):
        body = b"short"
        with aioresponses() as m:
            # First response: advertises 100 bytes but only sends 5.
            m.get(_URL, body=body, status=200, headers={"Content-Length": "100"})
            # Second response: correct.
            m.get(_URL, body=body, status=200, headers={"Content-Length": str(len(body))})
            async with aiohttp.ClientSession() as session:
                await _download_file(
                    session, _URL, dest, stop_event,
                    max_retries=2, base_delay=0.0,
                )
        assert dest.exists()
        assert dest.read_bytes() == body

    async def test_mismatch_leaves_no_tmp_after_failure(self, dest, stop_event):
        body = b"x"
        with aioresponses() as m:
            for _ in range(3):
                m.get(_URL, body=body, status=200, headers={"Content-Length": "999"})
            async with aiohttp.ClientSession() as session:
                await _download_file(
                    session, _URL, dest, stop_event,
                    max_retries=2, base_delay=0.0,
                )
        assert not dest.exists()
        assert not _tmp(dest).exists()


# ---------------------------------------------------------------------------
# Cleanup on failure
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Disk-space guard
# ---------------------------------------------------------------------------


class TestDiskSpaceGuard:
    async def test_low_disk_space_sets_stop_event(self, dest, stop_event):
        """shutil.disk_usage returns free < threshold → stop_event set, file not created."""
        import shutil
        from unittest.mock import patch
        from collections import namedtuple

        DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
        # 50 GB < 100 GB threshold
        low_disk = DiskUsage(total=500 * 1024**3, used=450 * 1024**3, free=50 * 1024**3)

        with patch.object(shutil, "disk_usage", return_value=low_disk):
            async with aiohttp.ClientSession() as session:
                await _download_file(session, _URL, dest, stop_event)

        assert not dest.exists()
        assert stop_event.is_set()


# ---------------------------------------------------------------------------
# Generic exception (not ClientResponseError / ClientConnectorError)
# ---------------------------------------------------------------------------


class TestGenericException:
    async def test_generic_exception_returns_immediately(self, dest, stop_event):
        """A non-aiohttp exception → returns immediately (no retry), tmp removed."""
        import aiohttp as _aiohttp
        from aioresponses import aioresponses as _aioresponses

        class _FakeSession:
            def get(self, url, **kw):
                raise ValueError("unexpected error")

        async with aiohttp.ClientSession() as real_session:
            # Inject a session whose .get() raises a generic exception
            class _BrokenSession:
                class _CtxMgr:
                    async def __aenter__(self):
                        raise ValueError("unexpected error")

                    async def __aexit__(self, *a):
                        pass

                def get(self, url, **kw):
                    return self._CtxMgr()

            broken = _BrokenSession()
            await _download_file(broken, _URL, dest, stop_event, max_retries=0)

        assert not dest.exists()
        assert not _tmp(dest).exists()


class TestDownloadCleanup:
    async def test_no_tmp_after_404(self, dest, stop_event):
        with aioresponses() as m:
            m.get(_URL, status=404)
            async with aiohttp.ClientSession() as session:
                await _download_file(session, _URL, dest, stop_event)
        assert not _tmp(dest).exists()

    async def test_no_tmp_after_exhausted_retries(self, dest, stop_event):
        with aioresponses() as m:
            for _ in range(5):
                m.get(_URL, status=503)
            async with aiohttp.ClientSession() as session:
                await _download_file(
                    session, _URL, dest, stop_event,
                    max_retries=4, base_delay=0.0,
                )
        assert not _tmp(dest).exists()
