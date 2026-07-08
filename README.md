# mattemodel

System-wide **video/image matting daemon**. One [Robust Video Matting
(RVM)](https://github.com/PeterL1n/RobustVideoMatting) model loaded **once** on
the GPU and exposed over local HTTP, so any program on the machine can ask for a
soft-alpha matte — instead of each app re-embedding the model.

RVM produces a **soft alpha matte** (clean hair/edge detail) and does it **without
a green screen**. It mattes *humans* — for arbitrary objects use the sibling
**segmodel** (BiRefNet); for whisper STT there's **NulSpeech2Text**. All three share
the same service shape: systemd user daemon, loopback HTTP, model-manager control
API, **and a build-time backend choice**.

## Backends (pick at build time)

| `./build.sh <backend>` | Engine | Runs on |
|---|---|---|
| **`cuda`** *(default)* | ONNX Runtime + CUDA (RVM resnet50) | NVIDIA — what the camera pipeline trusts |
| **`vulkan`** | ncnn-Vulkan (RVM resnet50/mobilenetv3) | any GPU (AMD/Intel/NVIDIA) |

Same HTTP/control API either way, so clients never care which backend is built.

## Architecture

```
        ┌──────────── mattemodel-server ────────────┐
        │  RVM, one model warm on the GPU            │
        │  HTTP on 127.0.0.1:48460                   │
        │    POST /matte?format=cutout|alpha → PNG   │
        │    GET  /status                            │
        │    POST /load /activate /unload /park      │
        └───────────────────┬────────────────────────┘
          nulpaint / Kdenlive / OBS clients POST images, get mattes
```

`load/activate` + `unload/park` let a future Plasma model-manager flip GPU
residency uniformly across the `*model` services.

## Install

```bash
cd ~/programming && git clone https://github.com/trickeri/mattemodel.git
cd mattemodel
./fetch-model.sh cuda          # or: vulkan | all
./build.sh                     # cuda (default); or: ./build.sh vulkan
ln -s "$PWD/mattemodel.service" ~/.config/systemd/user/mattemodel.service
systemctl --user daemon-reload && systemctl --user enable --now mattemodel.service
```

**Prerequisites** — `cuda`: system ONNX Runtime (`pkg-config libonnxruntime`) +
OpenCV. `vulkan`: ncnn built `-DNCNN_VULKAN=ON` (sibling `~/programming/ncnn`;
`build.sh` builds it if missing) + Vulkan toolchain + OpenCV.

## Use

```bash
client/mattemodel.sh frame.png out.png cutout
client/mattemodel.sh --status
client/mattemodel.sh --park            # free GPU; --activate to reload
python client/mattemodel.py frame.png alpha out.png
curl -F image=@frame.png 'http://127.0.0.1:48460/matte?format=alpha' -o alpha.png
```

`format=cutout` → straight-alpha RGBA (subject over transparency); `format=alpha`
→ grayscale matte.

## License

MIT — see [LICENSE](LICENSE).
