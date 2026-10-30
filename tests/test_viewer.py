import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from microduck_ppo.navigation_artifacts import write_json
from microduck_ppo.viewer import make_server


@pytest.fixture
def viewer(tmp_path):
    run = tmp_path / "trial"
    write_json(run / "run.json", {"status": "training", "created_at": 0, "snapshots": []})
    (run / "clip.mp4").write_bytes(b"0123456789")
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", run
    server.shutdown()
    server.server_close()
    thread.join()


def test_video_range_seeking_and_run_status(viewer):
    base, _ = viewer
    with urlopen(
        Request(base + "/artifacts/trial/clip.mp4", headers={"Range": "bytes=2-5"})
    ) as response:
        assert response.status == 206
        assert response.headers["Content-Range"] == "bytes 2-5/10"
        assert response.read() == b"2345"
    with urlopen(base + "/api/runs/trial") as response:
        assert json.load(response)["status"] == "training"


def test_cancel_is_explicit_and_cross_origin_launch_is_rejected(viewer):
    base, run = viewer
    with urlopen(Request(base + "/api/runs/trial/stop", data=b"{}", method="POST")) as response:
        assert json.load(response)["status"] == "stop_requested"
    assert (run / "STOP").exists()
    with pytest.raises(HTTPError) as result:
        urlopen(
            Request(
                base + "/api/train",
                data=b"{}",
                method="POST",
                headers={"Origin": "https://example.com"},
            )
        )
    assert result.value.code == 403


def test_invalid_budget_cannot_launch_process(viewer):
    base, _ = viewer
    with pytest.raises(HTTPError) as result:
        urlopen(Request(base + "/api/train", data=b'{"timesteps":-1}', method="POST"))
    assert result.value.code == 400
