from __future__ import annotations

from PIL import Image
from textual._compositor import CompositorUpdate
from textual_image.widget import SixelImage

from agent_tui import branding
from agent_tui.tui import AgentApp, ApprovalScreen


def screen_output(app: AgentApp) -> str:
    update = app.screen._compositor.render_update(full=True)
    assert isinstance(update, CompositorUpdate)
    return update.render_segments(app.console)


async def test_png_renders_as_sixel_and_stays_clear_of_approval_dialog(settings, monkeypatch):
    monkeypatch.setattr(branding, "AutoImageRenderable", branding.SixelRenderable)
    with Image.open(branding.LOGO_PATH) as image:
        assert image.format == "PNG"
        assert image.size == (400, 400)
    app = AgentApp(settings)
    async with app.run_test(size=(150, 28)) as pilot:
        logo = app.query_one("#brand-logo", SixelImage)
        assert logo.image == branding.LOGO_PATH
        assert "\x1bP" in screen_output(app)
        await pilot.resize_terminal(40, 20)
        assert logo.region.height == 3
        assert logo.region.right < app.query_one("#brand").region.x
        assert "\x1bP" in screen_output(app)
        await app.push_screen(ApprovalScreen("write_file", "Review this file change"))
        await pilot.pause()
        assert "\x1bP" not in screen_output(app)
        await pilot.press("escape")
        assert "\x1bP" in screen_output(app)


async def test_unsupported_terminal_keeps_wordmark_without_character_art(settings, monkeypatch):
    monkeypatch.setattr(branding, "AutoImageRenderable", None)
    app = AgentApp(settings)
    async with app.run_test():
        assert not app.query_one("#brand-logo").display
        assert "PyLattice" in str(app.query_one("#brand").render())
