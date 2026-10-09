"""Render the packaged logo using real terminal graphics, never character art."""

from pathlib import Path

from textual.widget import Widget
from textual.widgets import Static
from textual_image.renderable import Image as AutoImageRenderable
from textual_image.renderable.sixel import Image as SixelRenderable
from textual_image.renderable.tgp import Image as TGPRenderable
from textual_image.widget import SixelImage, TGPImage

LOGO_PATH = Path(__file__).with_name("assets") / "logo.png"


def logo_widget() -> Widget:
    # Importing textual-image above probes the terminal before Textual owns stdin.
    # Do not use its automatic widget: that falls back to Unicode / block art.
    if AutoImageRenderable is SixelRenderable:
        return SixelImage(LOGO_PATH, id="brand-logo")
    if AutoImageRenderable is TGPRenderable:
        return TGPImage(LOGO_PATH, id="brand-logo")
    return Static("", id="brand-logo", classes="unavailable")
