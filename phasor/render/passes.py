"""The three render passes (SPEC "3D spatial awareness"), on ModernGL.

| pass  | target        | content                                               |
| color | RGBA16F       | the picture, linear light                             |
| id    | R32UI + depth | object index per pixel, nearest occluding surface     |
| solo  | R8, no depth  | one object's full silhouette as if nothing hid it     |

All three run the same object ``draw`` code and the same vertex shader; only the fragment
shader and the GL state differ, so what the inspector measures is what the picture shows.
Coplanar objects (every 2D scene) are separated by a tiny per-object depth bias in draw
order, which makes "nearest surface" well defined without z-fighting.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from phasor.core import text as text_metrics
from phasor.core.state import ObjectState, SceneState
from phasor.inspect.inspector import occludes
from phasor.render.glcontext import shared_context

DEPTH_BIAS = 1e-5  # NDC depth per draw-order step
MAX_TEXT_PX = 512

_VS = """
#version 430
uniform mat4 u_vp;
uniform float u_bias;
uniform float u_zoom;
in vec3 in_pos;
#ifdef TEXTURED
in vec2 in_uv;
out vec2 v_uv;
#endif
void main() {
    gl_Position = u_vp * vec4(in_pos, 1.0);
    gl_Position.xy *= u_zoom;
    gl_Position.z -= u_bias * gl_Position.w;
#ifdef TEXTURED
    v_uv = in_uv;
#endif
}
"""
_FS = {
    "color": """
#version 430
uniform vec4 u_color;
#ifdef TEXTURED
uniform sampler2D u_tex;
in vec2 v_uv;
#endif
out vec4 frag;
void main() {
    float a = u_color.a;
#ifdef TEXTURED
    a *= texture(u_tex, v_uv).r;
#endif
    frag = vec4(u_color.rgb * a, a);  // premultiplied
}
""",
    "id": """
#version 430
uniform uint u_id;
#ifdef TEXTURED
uniform sampler2D u_tex;
in vec2 v_uv;
#endif
layout(location = 0) out uint frag;
void main() {
#ifdef TEXTURED
    if (texture(u_tex, v_uv).r < 0.5) discard;
#endif
    frag = u_id;
}
""",
    "solo": """
#version 430
#ifdef TEXTURED
uniform sampler2D u_tex;
in vec2 v_uv;
#endif
out float frag;
void main() {
#ifdef TEXTURED
    if (texture(u_tex, v_uv).r < 0.5) discard;
#endif
    frag = 1.0;
}
""",
}


def _define(src: str, textured: bool) -> str:
    head, _, rest = src.strip().partition("\n")
    return f"{head}\n{'#define TEXTURED' if textured else ''}\n{rest}"


class _Collector:
    """Records an object's draw calls; the renderer replays them for the current pass."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []

    def triangles(self, positions, indices=None, *, color, emission=0.0) -> None:
        self.commands.append(("mesh", {"positions": positions, "indices": indices, "color": color}))

    def text(self, text, origin, height, *, color, anchor="center") -> None:
        self.commands.append(
            ("text", {"text": text, "origin": origin, "height": height, "color": color,
                      "anchor": anchor})
        )  # fmt: skip


class GLRenderer:
    """Renders SceneStates. Also satisfies ``phasor.inspect.passes.PassRunner``."""

    def __init__(self, ctx=None) -> None:
        self.ctx = ctx or shared_context()
        self._programs: dict[tuple[str, bool], Any] = {}
        self._targets: dict[tuple[str, tuple[int, int]], Any] = {}
        self._text_textures: dict[tuple[str, int], Any] = {}
        self._cache: dict[tuple, np.ndarray] = {}
        self._cache_state: SceneState | None = None

    # ------------------------------------------------------------ resources

    def renderer_class(self) -> str:
        """``"software"`` (Mesa llvmpipe and friends) or ``"gpu"``. Golden images are kept
        per class because the two do not antialias identically. The query must run with
        the context current, like every other GL call."""
        with self.ctx:
            name = str(self.ctx.info["GL_RENDERER"]).lower()
        return "software" if any(k in name for k in ("llvmpipe", "softpipe", "swrast")) else "gpu"

    def _program(self, pass_: str, textured: bool):
        key = (pass_, textured)
        if key not in self._programs:
            self._programs[key] = self.ctx.program(
                vertex_shader=_define(_VS, textured), fragment_shader=_define(_FS[pass_], textured)
            )
        return self._programs[key]

    def _target(self, pass_: str, size: tuple[int, int]):
        key = (pass_, size)
        if key not in self._targets:
            ctx = self.ctx
            if pass_ == "color":
                samples = min(4, ctx.max_samples)
                ms = ctx.framebuffer(
                    [ctx.texture(size, 4, samples=samples, dtype="f2")],
                    ctx.depth_renderbuffer(size, samples=samples),
                )
                self._targets[key] = (ms, ctx.framebuffer([ctx.texture(size, 4, dtype="f2")]))
            elif pass_ == "id":
                fbo = ctx.framebuffer(
                    [ctx.texture(size, 1, dtype="u4")], ctx.depth_renderbuffer(size)
                )
                self._targets[key] = (fbo, fbo)
            else:
                fbo = ctx.framebuffer([ctx.texture(size, 1, dtype="f1")])
                self._targets[key] = (fbo, fbo)
        return self._targets[key]

    def _text_texture(self, text: str, px: int):
        key = (text, px)
        if key not in self._text_textures:
            bitmap = text_metrics.rasterize(text, px)
            tex = self.ctx.texture(bitmap.shape[::-1], 1, bitmap.tobytes(), alignment=1)
            self._text_textures[key] = tex
        return self._text_textures[key]

    # ------------------------------------------------------------ drawing

    def _draw_object(self, state, obj: ObjectState, pass_, size, order: int, zoom: float) -> None:
        ctx = self.ctx
        cam = state.camera_for(obj)
        collector = _Collector()
        obj.impl.draw(collector, state, pass_)
        for kind, c in collector.commands:
            textured = kind == "text"
            prog = self._program(pass_, textured)
            prog["u_vp"].write(cam.view_proj.astype("f4").T.tobytes())
            prog["u_bias"].value = order * DEPTH_BIAS
            prog["u_zoom"].value = 1.0 / zoom
            if pass_ == "color":
                prog["u_color"].value = tuple(float(x) for x in c["color"])
            elif pass_ == "id":
                prog["u_id"].value = obj.index
            buffers = []
            if textured:
                corner, wv, hv = text_metrics.quad(
                    c["text"], c["origin"], c["height"], c["anchor"], cam.right, cam.up
                )
                px = round(c["height"] / cam.units_per_height() * size[1])
                tex = self._text_texture(c["text"], int(np.clip(px, 2, MAX_TEXT_PX)))
                tex.use(0)
                prog["u_tex"].value = 0
                # Bitmap row 0 is the top of the text, so the top corners take v = 0.
                quad = np.array(
                    [
                        [*corner, 0, 1],
                        [*(corner + wv), 1, 1],
                        [*(corner + hv), 0, 0],
                        [*(corner + wv), 1, 1],
                        [*(corner + wv + hv), 1, 0],
                        [*(corner + hv), 0, 0],
                    ],
                    dtype="f4",
                )
                vbo = ctx.buffer(quad.tobytes())
                vao = ctx.vertex_array(prog, [(vbo, "3f 2f", "in_pos", "in_uv")])
                buffers = [vbo]
            else:
                vbo = ctx.buffer(np.ascontiguousarray(c["positions"], dtype="f4").tobytes())
                buffers = [vbo]
                ibo = None
                if c["indices"] is not None:
                    ibo = ctx.buffer(np.ascontiguousarray(c["indices"], dtype="i4").tobytes())
                    buffers.append(ibo)
                vao = ctx.vertex_array(prog, [(vbo, "3f", "in_pos")], index_buffer=ibo)
            vao.render()
            vao.release()
            for b in buffers:
                b.release()

    def _render(self, state, pass_, size, **kwargs) -> np.ndarray:
        with self.ctx:  # make our context current: another one may have been used since
            return self._render_current(state, pass_, size, **kwargs)

    def _render_current(
        self, state, pass_, size, *, only=None, force=frozenset(), zoom=1.0
    ) -> np.ndarray:
        import moderngl

        ctx = self.ctx
        target, resolved = self._target(pass_, size)
        target.use()
        ctx.viewport = (0, 0, *size)
        if pass_ == "color":
            target.clear(*state.background)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        else:
            target.clear(0, 0, 0, 0)
            ctx.disable(moderngl.BLEND)
        if pass_ == "solo":
            ctx.disable(moderngl.DEPTH_TEST)
        else:
            ctx.enable(moderngl.DEPTH_TEST)
        for order, obj in enumerate(state.draw_order()):
            if only is not None and obj.id != only:
                continue
            if obj.params.opacity <= 0:
                continue
            if pass_ == "id" and not occludes(obj) and obj.id not in force:
                continue
            # The overlay layer always sits on top of the world: it ignores world depth.
            ctx.depth_func = "1" if obj.layer == "overlay" else "<="
            self._draw_object(state, obj, pass_, size, order, zoom)
        if resolved is not target:
            ctx.copy_framebuffer(resolved, target)
        w, h = size
        if pass_ == "color":
            raw = np.frombuffer(resolved.read(components=4, dtype="f2"), dtype=np.float16)
            return raw.reshape(h, w, 4)[::-1].astype(np.float32)
        if pass_ == "id":
            raw = np.frombuffer(resolved.read(components=1, dtype="u4"), dtype=np.uint32)
            return raw.reshape(h, w)[::-1].copy()
        raw = np.frombuffer(resolved.read(components=1, dtype="f1"), dtype=np.uint8)
        return raw.reshape(h, w)[::-1] > 127

    def _cached(self, state: SceneState, key: tuple, make) -> np.ndarray:
        if self._cache_state is not state:
            self._cache, self._cache_state = {}, state
        if key not in self._cache:
            out = make()
            out.setflags(write=False)
            self._cache[key] = out
        return self._cache[key]

    # ------------------------------------------------------------ public

    def color(self, state: SceneState, size: tuple[int, int] | None = None) -> np.ndarray:
        """(h, w, 4) float32 linear RGBA, row 0 = top."""
        size = size or (state.width, state.height)
        return self._render(state, "color", size)

    def id_buffer(self, state, size, force: frozenset[str] = frozenset()) -> np.ndarray:
        return self._cached(
            state, ("id", size, force), lambda: self._render(state, "id", size, force=force)
        )

    def solo_mask(self, state, size, object_id: str, zoom_out: float = 1.0) -> np.ndarray:
        return self._cached(
            state,
            ("solo", size, object_id, round(zoom_out, 6)),
            lambda: self._render(state, "solo", size, only=object_id, zoom=zoom_out),
        )


@functools.cache
def default_renderer() -> GLRenderer:
    """The process-wide renderer on the shared context. Raises GLUnavailableError on tier C."""
    return GLRenderer()
