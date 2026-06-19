"""Reference client for the mattemodel daemon (stdlib only).

    from mattemodel import matte
    rgba_png = matte(open("frame.png", "rb").read())            # RGBA cutout
    alpha_png = matte(open("frame.png", "rb").read(), "alpha")   # grayscale matte

Plus control helpers (status/load/unload/park/activate). Backend (cuda|vulkan) is
a server build-time choice — the client and wire format are identical either way.
"""
import os
import json
import urllib.request

HOST = os.environ.get("MM_HOST", "127.0.0.1")
PORT = os.environ.get("MM_PORT", "48460")
BASE = os.environ.get("MM_HTTP_URL", f"http://{HOST}:{PORT}")


def matte(image_bytes: bytes, fmt: str = "cutout") -> bytes:
    """POST an image to mattemodel; return PNG bytes (cutout RGBA or alpha gray)."""
    boundary = "----mattemodel"
    head = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"frame.png\"\r\nContent-Type: image/png\r\n\r\n"
    ).encode()
    body = head + image_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/matte?format={fmt}", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _post(path: str) -> dict:
    # Control POSTs must carry a body (even empty) — the server 400s otherwise.
    req = urllib.request.Request(f"{BASE}/{path}", data=b"", method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def status() -> dict:
    with urllib.request.urlopen(f"{BASE}/status") as resp:
        return json.loads(resp.read().decode())


def load():     return _post("load")
def activate(): return _post("activate")
def unload():   return _post("unload")
def park():     return _post("park")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(json.dumps(status(), indent=2))
    else:
        fmt = sys.argv[2] if len(sys.argv) > 2 else "cutout"
        out = sys.argv[3] if len(sys.argv) > 3 else "matte.png"
        with open(sys.argv[1], "rb") as f:
            png = matte(f.read(), fmt)
        with open(out, "wb") as f:
            f.write(png)
        print(f"wrote {out} ({len(png)} bytes, format={fmt})")
