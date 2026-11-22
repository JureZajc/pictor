# Pictor

Choose a picture, say what you want done, and an AI model running **on your own computer** does it with image tools. Nothing is uploaded anywhere.

![Pictor editing a photo in three steps](docs/screenshot.jpg)

You write "make it black and white, crop it to a square and add a thin white border". The model looks at the picture, decides which tools to call, and you watch each step happen. Every step is kept as a version you can go back to.

## What you need

| | |
|---|---|
| **[Ollama](https://ollama.com/download)** | The one program to download. It is free, open source (MIT) and runs the AI model on your machine. Install it and start it. |
| **[uv](https://docs.astral.sh/uv/)** | Runs the app and fetches Python and the libraries for you. |
| A reasonably recent computer | Apple Silicon Mac, a PC with an NVIDIA or AMD graphics card, or just a CPU with 8 GB of RAM or more. |
| **[mflux](https://github.com/filipstrand/mflux)**, optional | Only if you want new things painted into pictures. See [Adding things to pictures](#adding-things-to-pictures-optional). |

Pictor itself is Python only. The interface is a small web page bundled in the package and served locally, so there is no Node.js and no build step.

## Start

```bash
git clone https://github.com/JureZajc/pictor
cd pictor
uv run pictor
```

Your browser opens at `http://127.0.0.1:8765`. Without cloning:

```bash
uvx --from git+https://github.com/JureZajc/pictor pictor
```

## First start: the model is chosen for your machine

On first start Pictor looks at your computer and picks the largest model that fits:

1. It reads the CPU, the memory and the graphics card's memory.
2. It works out a memory budget: 70% of memory on Apple Silicon (the CPU and GPU share it), 95% of video memory on a PC with a graphics card, and at most 8 GB on a CPU-only machine, where speed is the limit.
3. It offers the largest model from its list that fits the budget, already selected. One click downloads it through Ollama and you are ready. You can pick any other model instead.

What that means in practice:

| Machine | Budget | Model chosen | Download |
|---|---|---|---|
| Mac with 8 GB | 5.6 GB | `qwen3.5:2b-mlx` | 3.1 GB |
| Mac with 16 or 24 GB | 11.2 / 16.8 GB | `gemma4:12b-mlx` | 7.7 GB |
| Mac with 32 GB | 22.4 GB | `gemma4:26b` | 16 GB |
| Mac with 48 GB or more | 33.6 GB | `qwen3.6:35b-mlx` | 23 GB |
| PC with a 6 GB card | 5.7 GB | `qwen3.5:2b` | 2.7 GB |
| PC with an 8 GB card | 7.6 GB | `qwen3.5:4b` | 3.3 GB |
| PC with a 12 or 16 GB card | 11.4 / 15.2 GB | `gemma4:12b` | 7.7 GB |
| PC with a 24 GB card | 22.8 GB | `gemma4:26b` | 16 GB |
| PC with a 32 GB card | 30.4 GB | `qwen3.6:35b` | 23 GB |
| No graphics card, 8 GB RAM | 4 GB | `qwen3.5:0.8b` | 1.2 GB |
| No graphics card, 16 GB RAM or more | 8 GB | `qwen3.5:4b` | 3.3 GB |

A model needs more memory than its download size, because the picture, the tool list and the conversation take room too. Pictor estimates `download × 1.2 + 1.8 GB`, calibrated on a real run where a 7.7 GB model peaked at 11.1 GB.

A model you already have in Ollama is preferred when it is nearly as good as the best download. On Apple Silicon, Ollama's MLX builds are used when they fit. Models must support tool calling; one that cannot see pictures still works but edits blind.

To see what Pictor detects and would choose, without starting the app:

```bash
uv run pictor doctor
```

Change the model later by clicking its name in the top right corner.

## Using it

1. **Choose a picture**: click, drag it in, or paste from the clipboard. JPEG, PNG, WebP, HEIC, GIF, BMP and TIFF work.
2. **Say what to do**, in your own words and your own language. Some examples:
   - "Remove the background"
   - "Blur all the faces"
   - "Crop to 16:9 keeping the top, then make it 1920 pixels wide"
   - "Make it warmer and a bit brighter"
   - "Write 'Summer 2026' in the bottom right corner in yellow"
   - "Blur the background like a portrait photo"
   - "What is in this picture?"
   - "Add a small brown dog sitting on the grass on the right" (with the [optional image generator](#adding-things-to-pictures-optional))
3. **Watch the steps.** Each tool call appears in the conversation; click one to see exactly what was called.
4. **Undo, compare, download.** Every edit is a version in the strip under the picture. Hold *Compare* to see the original. *Download* saves the current version.

The conversation is remembered, so "actually undo that and add a border instead" works.

## What the model can do

| Tool | What it does |
|---|---|
| Crop, crop to ratio | Cut to a region or to an aspect ratio such as 1:1 or 16:9 |
| Resize | By percent, or to a width and height (fit, fill or stretch) |
| Rotate, flip | Any angle, optionally trimming the empty corners |
| Border, pad to ratio | Add a frame, or reach an aspect ratio without cropping |
| Adjust | Brightness, contrast, saturation, sharpness, warmth |
| Filter | Grayscale, sepia, invert, blur, sharpen, vintage, posterize, pixelate, vignette, sketch, emboss |
| Auto enhance | One-step fix for a dull photo |
| Blur region | Blur or pixelate a rectangle or ellipse |
| Detect faces, anonymize faces | Find faces and blur or pixelate them |
| Remove background | Cut out the subject, leaving transparency |
| Change background | Solid colour, blur, black and white, or darken, keeping the subject |
| Add text, draw shape | Captions and outline boxes or ellipses |
| Inspect, look | Read the picture's facts, or look at it again after an edit |
| Output format | Save as PNG, JPEG or WebP |
| Undo, reset | The model can take back its own mistakes |
| Add object, remove object | Only with the optional image generator: paint something new into a region, or take something out and fill the gap |

Without the optional image generator it cannot generate new content: it will not paint in something that is not there or erase an object and invent what was behind it. It says so when asked, and tells you where to switch the generator on. Even with it, the whole picture is never repainted or restyled.

## Adding things to pictures (optional)

"Add a small brown dog sitting on the grass on the right" needs a second AI model that can paint. Ollama cannot run one (it dropped image generation in version 0.32.6), so Pictor uses [mflux](https://github.com/filipstrand/mflux) with the FLUX.2 klein 4B model. It is off until you turn it on.

**Setting it up**

1. Install mflux the way its own README recommends, with uv (Homebrew had no formula for it in November 2026):

   ```bash
   uv tool install mflux==0.24.0
   ```

   This takes 1.1 GB. Pictor finds it in `~/.local/bin` even when that folder is not on your `PATH`.
2. In Pictor, click the model name in the top right corner. Under *Add things to pictures*, click *Download (16 GB) and turn on*. Nothing this large is downloaded without that click.

| What | Licence | Size |
|---|---|---|
| mflux 0.24.0, with PyTorch and MLX | MIT | 1.1 GB |
| [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) by Black Forest Labs, from Hugging Face, no account needed | Apache-2.0 | 16 GB download |
| An 8-bit copy of that model, which Pictor has mflux prepare once | Apache-2.0 | 8.6 GB |

Edits load the 8-bit copy and ran without the 16 GB download present, so that download can be deleted afterwards. It is needed again only if you change `image_quantize`. `uv run pictor doctor` shows where everything is.

**How it works**

The image model redraws everything it is shown and shifts every pixel slightly while doing so. Pictor therefore shows it only a window around the region the planning model chose (1.6 times the region, at up to 1024 pixels), compares its answer with the original, and puts back just the part that really changed: the new thing and its shadow, with a soft edge. Every other pixel of your picture stays bit for bit what it was, and the picture keeps its size. If the image model changed nothing in the region, or repainted the whole window, Pictor asks it again with another seed, up to three times, and otherwise leaves the picture alone.

**Measured on the test machine** (Apple Silicon Mac, 24 GB, mflux 0.24.0, `gemma4:12b-mlx` as the planning model)

| | |
|---|---|
| One "add …" instruction, start to finish | 30 to 38 seconds |
| of which painting | 10 to 16 seconds (4 steps) |
| Peak memory of the image model | 8.1 GB |
| Stopping a running painting | The process was gone 0.12 seconds after *Stop* |

The planning model (8.4 GB) and the image model do not fit together in that machine's 16.8 GB budget, so Pictor has Ollama unload the planning model while the image model paints. It loads again by itself to look at the result and answer, and that last step is about a third of those 30 to 38 seconds. On a machine where both fit, nothing is unloaded. Five additions in a row ran without a memory failure.

Painting works on Apple Silicon Macs and, according to mflux, on Linux with an NVIDIA card. Only the Mac was tested. It does not work on Windows.

## What gets downloaded, and where

Everything below is open source and stays on your machine.

| What | Size | When | Where |
|---|---|---|---|
| The AI model | 1 to 23 GB, see the table above | First start, after you confirm | Ollama's model folder |
| Background model (IS-Net, through [rembg](https://github.com/danielgatis/rembg)) | 180 MB | First time a background tool runs | `~/.rembg` |
| Face detector ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), MIT) | 230 KB | First time a face tool runs | Pictor's cache folder |
| Image model (FLUX.2 klein 4B, Apache-2.0), optional | 16 GB | When you click *Download and turn on* | Hugging Face's cache, `~/.cache/huggingface` |
| 8-bit copy of the image model, optional | 8.6 GB, made locally | Right after that download | Pictor's cache folder, under `image-models` |

Each AI model has its own licence, shown on its page in the [Ollama library](https://ollama.com/library). After these downloads Pictor works offline. The image model is started with Hugging Face's offline switch set, so an edit can never begin a download.

## Settings

`uv run pictor doctor` prints where the settings file is. It is JSON:

| Key | Default | Meaning |
|---|---|---|
| `model` | chosen on first start | Ollama tag of the model |
| `think` | `"auto"` | `"auto"` keeps reasoning brief when the model allows it, `"off"` is fastest, `"default"` leaves the model's own setting |
| `num_ctx` | from the memory budget | Context window in tokens (8192 to 16384) |
| `vision_max_side` | `1024` | Longest side of the copy of the picture the model looks at |
| `max_steps` | `12` | Most tool rounds for one instruction |
| `background_model` | `"isnet-general-use"` | Any [rembg model name](https://github.com/danielgatis/rembg#models) |
| `ollama_host` | `OLLAMA_HOST` or port 11434 | Where Ollama listens |
| `image_generator` | `"off"` | `"mflux"` once you turn on [adding things to pictures](#adding-things-to-pictures-optional) |
| `image_model` | `"flux2-klein-4b"` | The model name passed to mflux. The 9B variants have a non-commercial licence |
| `image_quantize` | `8` | Bits per weight: 3, 4, 5, 6 or 8. Fewer bits need less memory and lose some quality |
| `image_steps` | `4` | Painting steps. Eight took 19 seconds instead of 11 and looked no better |
| `image_max_side` | `1024` | Longest side of the part of the picture handed to the image model |
| `image_small_text_encoder` | `false` | mflux's experimental small text encoder: a 10 GB download instead of 16 GB, weaker at following descriptions. Not tested |

Command line:

```
pictor [run|doctor] [--port 8765] [--host 127.0.0.1] [--model TAG] [--no-browser]
```

Environment: `OLLAMA_HOST` points at another Ollama server, `PICTOR_HOME` keeps settings and cache in a folder of your choice.

### Updating the model list

Local models change quickly. The list Pictor recommends from is data, in [`src/pictor/data/models.json`](src/pictor/data/models.json), last updated 2026-11-22 from the [Ollama library](https://ollama.com/search?c=vision&c=tools). To use your own list, put a `models.json` of the same shape next to the settings file. You can also type any Ollama tag under *More options* in the model dialog.

The tiers in that list are ordered by model size, not by a benchmark of how well each model edits pictures.

## Privacy and safety

- The model, the tools and the pictures all stay on your computer.
- The server listens on `127.0.0.1` only and refuses requests that come from other websites.
- The model's tools can only change the picture you opened. They cannot read or write other files, run commands or reach the network.
- The one exception is the optional image generator. Its tools start a single fixed program, mflux, and hand it a copy of part of the picture and a description. The description is passed as one argument, never through a shell, and mflux runs with network access to Hugging Face switched off.
- Working copies live in Pictor's cache folder and are cleared out on a later start, once they are a day old.

## Limits worth knowing

- **Tested on one machine.** Everything here was run on an Apple Silicon Mac with 24 GB of memory, Ollama 0.40.16 and `gemma4:12b-mlx`, where a two-step edit (black and white, then a square crop) took 14 seconds. The graphics card detection for Windows and Linux is covered by unit tests but has not been run on real hardware.
- **Small models make mistakes.** Below about 9B parameters, expect wrong tools or clumsy coordinates. Undo is one click.
- **Positions are approximate.** The model estimates regions by eye, as percentages of the picture.
- **A model that does not fit will not run well.** On that 24 GB Mac an 18 GB model failed with an out-of-memory error from Ollama. Pictor marks such models "Too large" and explains the failure if you pick one anyway.
- **Painted things land roughly where asked, and are not always right.** In 25 test additions and removals the first attempt was accepted every time, with the new thing in or next to its region. But a cat asked for "on the bench" ended up beside it more than once, and the image model sometimes paints a second copy elsewhere, which Pictor leaves out. Nothing checks that what was painted looks right: one early prompt wording produced a horse with two heads. Undo and ask again.
- **Painting was tested less than the rest.** It ran on one Mac and two pictures: a generated park scene of 1024 × 768 pixels, also enlarged to 4096 × 3072 and with part of it made see-through, and a 600 × 400 photo of a coffee cup. Not run: Linux with an NVIDIA card, the 9B models, the small text encoder, and the progress display of the 16 GB download inside the app (the download itself was done with mflux directly). On see-through areas a painted thing can come with a patch of invented background.

## Development

```bash
uv sync
uv run pytest          # needs neither Ollama nor a network connection
uv run ruff check .
uv run ruff format .
```

| Path | What is there |
|---|---|
| `src/pictor/hardware.py` | CPU, memory and GPU detection, and the memory budget |
| `src/pictor/catalog.py`, `data/models.json` | The model list and the rule that picks one |
| `src/pictor/ollama.py` | Client for Ollama's HTTP API |
| `src/pictor/agent.py` | The loop: show the picture, run tool calls, feed results back |
| `src/pictor/tools/` | The tools, one plain function each |
| `src/pictor/generation.py` | The optional image generator: runs mflux as a separate process, sets it up, stops it |
| `src/pictor/tools/generative.py` | Painting into a region: the window shown to the image model and what is kept of its answer |
| `src/pictor/session.py` | A picture's versions, undo history and export |
| `src/pictor/server.py` | Local web server and JSON API |
| `src/pictor/static/` | The interface: HTML, CSS and one JavaScript file |

### Adding a tool

A tool is a function in `src/pictor/tools/`. The docstring is what the model reads, so write it for the model:

```python
from typing import Annotated

from PIL import Image, ImageOps

from pictor.tools.registry import P, ToolContext, ToolResult, tool


@tool("Mirror tile")
def mirror_tile(
    ctx: ToolContext,
    copies: Annotated[float, P("How many copies side by side", 2, 6)] = 2,
) -> ToolResult:
    """Repeat the picture side by side, mirroring every second copy."""
    image = ctx.image
    count = max(2, min(6, round(copies)))
    result = Image.new(image.mode, (image.width * count, image.height))
    for index in range(count):
        tile = ImageOps.mirror(image) if index % 2 else image
        result.paste(tile, (index * image.width, 0))
    return ToolResult(f"Tiled the picture {count} times.", result, detail=f"{count}×")
```

Import the module in `src/pictor/tools/__init__.py` and the model can call it. Returning an image creates a new version; returning only a message leaves the picture alone.

## Licence

MIT. See [LICENSE](LICENSE).
