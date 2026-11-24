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

import dataclasses
import functools
from collections import OrderedDict
from typing import Any

import numpy as np

from phasor.core import batches
from phasor.core import text as text_metrics
from phasor.core.state import ObjectState, SceneState
from phasor.core.timing import stage
from phasor.layout.geometry import occludes
from phasor.render.glcontext import on_gl_thread, shared_context

DEPTH_BIAS = 1e-5  # NDC depth per draw-order step
MAX_TEXT_PX = 512
MAX_FIELD_PX = 8192  # largest side of a kernel-drawn quad's texture
FIELD_CACHE = 16  # kernel results kept, keyed by object parameters and view

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
#ifdef VCOLOR
in vec4 in_color;
out vec4 v_color;
#endif
void main() {
    gl_Position = u_vp * vec4(in_pos, 1.0);
    gl_Position.xy *= u_zoom;
    gl_Position.z -= u_bias * gl_Position.w;
#ifdef TEXTURED
    v_uv = in_uv;
#endif
#ifdef VCOLOR
    v_color = in_color;
#endif
}
"""
_FS = {
    "color": """
#version 430
uniform vec4 u_color;
uniform float u_glow;
#ifdef BATCH
uniform float u_tint;
#endif
#ifdef TEXTURED
uniform sampler2D u_tex;
in vec2 v_uv;
#endif
#ifdef VCOLOR
in vec4 v_color;
#endif
layout(location = 0) out vec4 frag;
layout(location = 1) out vec4 emit;  // only the light above plain paint: the bloom source
void main() {
    float a = u_color.a;
    vec3 rgb = u_color.rgb;
#ifdef VCOLOR
    a *= v_color.a;   // a color per vertex (a hue running along a curve), times opacity
    rgb = v_color.rgb;
#endif
#ifdef IMAGE
    // Kernel pixels, premultiplied. Their alpha is a per-pixel glow weight w (0-1): with
    // the object's glow g, a pixel shines (1 + w (g - 1)) times brighter and sends
    // w (g - 1) of its light to the bloom. g = 1 (the default) emits nothing and hides
    // what does, as before.
    vec4 t = texture(u_tex, v_uv);
    float lit = t.a * max(u_glow - 1.0, 0.0);
    frag = vec4(t.rgb * (1.0 + lit) * a, a);
    emit = vec4(t.rgb * lit * a, a);
    return;
#endif
#ifdef TEXTURED
    a *= texture(u_tex, v_uv).r;
    // Empty texels of a text quad write nothing, depth included: otherwise glass drawn
    // later is cut out behind the whole rectangle of a label.
    if (a < 0.004) discard;
#endif
#ifdef BATCH
    rgb *= u_tint;  // the flash verb on a batch drawn in one colour (as on a plain mesh)
#endif
    frag = vec4(rgb * u_glow * a, a);  // premultiplied; glow > 1 is HDR light
    emit = vec4(rgb * max(u_glow - 1.0, 0.0) * a, a);
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


# Lit surfaces (SPEC "Look": PBR-lite). World-space lighting from a three-point rig placed
# relative to the camera, two-sided, Blinn-Phong highlights, and depth fog toward the
# background. Materials: 0 solid, 1 glass (brighter at grazing angles), 2 wire (a grid
# drawn along the surface coordinates), 3 emissive (unlit, glowing).
_VS_LIT = """
#version 430
uniform mat4 u_vp;
uniform float u_bias;
uniform float u_zoom;
in vec3 in_pos;
in vec3 in_normal;
in vec4 in_rgba;
in vec2 in_uv;
out vec3 v_pos;
out vec3 v_normal;
out vec4 v_rgba;
out vec2 v_uv;
void main() {
    gl_Position = u_vp * vec4(in_pos, 1.0);
    gl_Position.xy *= u_zoom;
    gl_Position.z -= u_bias * gl_Position.w;
    v_pos = in_pos; v_normal = in_normal; v_rgba = in_rgba; v_uv = in_uv;
}
"""
_FS_LIT = """
#version 430
uniform vec3 u_eye;
uniform vec3 u_light_dir[3];
uniform float u_light[3];
uniform float u_ambient;
uniform float u_specular;
uniform float u_shininess;
uniform float u_fog;
uniform float u_fog_ref;
uniform vec4 u_bg;
uniform float u_opacity;
uniform float u_glow;
uniform int u_material;
uniform vec2 u_grid;
in vec3 v_pos;
in vec3 v_normal;
in vec4 v_rgba;
in vec2 v_uv;
layout(location = 0) out vec4 frag;
layout(location = 1) out vec4 emit;
void main() {
    vec3 view = normalize(u_eye - v_pos);
    vec3 n = normalize(v_normal);
    if (dot(n, view) < 0.0) n = -n;  // two-sided
    vec3 base = v_rgba.rgb;
    vec3 c;
    if (u_material == 3) {
        c = base * u_glow;
    } else {
        float diffuse = u_ambient;
        float spec = 0.0;
        for (int i = 0; i < 3; i++) {
            vec3 l = normalize(u_light_dir[i]);
            diffuse += u_light[i] * max(dot(n, l), 0.0);
            vec3 h = normalize(l + view);
            spec += u_light[i] * pow(max(dot(n, h), 0.0), u_shininess);
        }
        c = base * diffuse + vec3(u_specular * spec);
        if (u_material == 2) {
            vec2 g = v_uv * u_grid;
            vec2 d = abs(fract(g - 0.5) - 0.5) / fwidth(g);
            float line = 1.0 - min(min(d.x, d.y), 1.0);
            c = mix(c, c * 1.8 + 0.08, line);
        }
    }
    float a = v_rgba.a * u_opacity;
    if (u_material == 1) {
        float fres = pow(1.0 - max(dot(n, view), 0.0), 3.0);
        a = clamp(a * (0.75 + 0.9 * fres), 0.0, 1.0);
    }
    float dist = length(u_eye - v_pos);
    float f = 1.0 - exp(-u_fog * max(dist - u_fog_ref, 0.0) / u_fog_ref);
    c = mix(c, u_bg.rgb, f);
    frag = vec4(c * a, a);
    float g = u_material == 3 ? max(u_glow - 1.0, 0.0) * (1.0 - f) : 0.0;
    emit = vec4(base * g * a, a);
}
"""
# World-space batches (Milestone 28, ``core/batches.py``). The arrays live in shader storage
# buffers that depend only on the batch (never on the camera); each instance (a dot, a line
# segment, an arrow) pulls what it needs by ``gl_InstanceID`` and its corner by
# ``gl_VertexID``, and is expanded to face the camera here. ``core.batches.instances`` is
# the same arithmetic in NumPy (the reference: tests compare the two through transform
# feedback). One of POINTS, LINES, FLAT, CONE is defined.
_VS_BATCH = """
#version 430
uniform mat4 u_vp;
uniform float u_bias;
uniform float u_zoom;
uniform vec3 u_eye;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_forward;
uniform int u_persp;
uniform float u_height;   // ortho: world units per frame height
uniform float u_tan;      // perspective: tan(vertical fov / 2)
uniform int u_units;      // 0 world, 1 share of the frame height, 2 pixels
uniform float u_px;       // 1 / the picture's height in pixels
uniform int u_anchored;
uniform vec3 u_anchor;
uniform float u_head;
uniform int u_n;          // items in the batch
uniform int u_each;       // bit 1: a colour per item (else col[0]), bit 2: a size per item
uniform int u_strips;     // lines: 1 when strip[] says where strips start, 0: all one strip
layout(std430, binding = 0) readonly buffer B0 { float pos[]; };
layout(std430, binding = 1) readonly buffer B1 { vec4 col[]; };
layout(std430, binding = 2) readonly buffer B2 { float siz[]; };
#ifdef LINES
layout(std430, binding = 3) readonly buffer B3 { int strip[]; };
#endif
#if defined(FLAT) || defined(CONE)
layout(std430, binding = 3) readonly buffer B3 { float tipv[]; };
#endif
out vec4 v_color;
out vec3 v_world;
const float TAU = 6.283185307179586;

vec3 P(int i) { return vec3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]); }
vec4 C(int i) { return (u_each & 1) != 0 ? col[i] : col[0]; }
float S(int i) { return (u_each & 2) != 0 ? siz[i] : siz[0]; }

float world_size(float s, vec3 at) {
    if (u_units == 0) return s;
    float share = u_units == 1 ? s : s * u_px;
    if (u_persp == 0) return share * u_height;
    vec3 ref = u_anchored == 1 ? u_anchor : at;
    return share * max(dot(ref - u_eye, u_forward), 1e-6) * 2.0 * u_tan;
}

void emit(vec3 p) {
    v_world = p;
    gl_Position = u_vp * vec4(p, 1.0);
    gl_Position.xy *= u_zoom;
    gl_Position.z -= u_bias * gl_Position.w;
}

#ifdef LINES
vec2 flat2(vec3 p) { return vec2(dot(p, u_right), dot(p, u_up)); }
// the normal (in the camera plane) of the segment from point i to i + 1
vec2 seg_normal(int i) {
    vec3 dp = P(i + 1) - P(i);  // project the difference: no cancellation
    vec2 d = flat2(dp);
    float n = sqrt(d.x * d.x + d.y * d.y);
    if (u_persp == 1 && n <= 1e-3 * sqrt(dot(dp, dp))) {  // along the view: its screen direction
        vec3 r0 = P(i) - u_eye;
        vec3 r1 = P(i + 1) - u_eye;
        d = flat2(r1) / max(dot(r1, u_forward), 1e-6) - flat2(r0) / max(dot(r0, u_forward), 1e-6);
        n = sqrt(d.x * d.x + d.y * d.y);
    }
    vec2 u = d / (n == 0.0 ? 1.0 : n);
    return vec2(-u.y, u.x);
}
#endif

void main() {
    int j = gl_InstanceID;
    int k = gl_VertexID;
#ifdef POINTS
    vec3 c = P(j);
    float r = world_size(S(j), c);
    vec2 corner = vec2(0.0);
    if (k > 0) {
        float a = TAU * float(k - 1) / 12.0;
        corner = vec2(cos(a), sin(a));
    }
    emit(c + r * (corner.x * u_right + corner.y * u_up));
    v_color = C(j);
#endif
#ifdef LINES
    // instance j: the segment j -> j + 1; vertices 0, 1 at point j, 2, 3 at j + 1 (+, -)
    int i = j + k / 2;
    float side = (k % 2 == 0) ? 1.0 : -1.0;
    vec3 p = P(i);
    if (u_strips == 1 && strip[j] != strip[j + 1]) {  // across two strips: nothing
        v_world = p;
        v_color = vec4(0.0);
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    bool has_prev = i > 0 && (u_strips == 0 || strip[i - 1] == strip[i]);
    bool has_next = i + 1 < u_n && (u_strips == 0 || strip[i + 1] == strip[i]);
    vec2 before = has_prev ? seg_normal(i - 1) : vec2(0.0);
    vec2 after = has_next ? seg_normal(i) : vec2(0.0);
    vec2 n_in = has_prev ? before : after;
    vec2 n_out = has_next ? after : before;
    vec2 m = n_in + n_out;
    m = m / max(sqrt(m.x * m.x + m.y * m.y), 1e-9);
    float scale = 1.0 / max(m.x * n_in.x + m.y * n_in.y, 0.25);
    float half_w = world_size(S(i), p) / 2.0;
    vec2 off = m * (scale * half_w);
    emit(p + side * (off.x * u_right + off.y * u_up));
    v_color = C(i);
#endif
#ifdef FLAT
    vec3 a = P(j);
    vec3 b = vec3(tipv[3 * j], tipv[3 * j + 1], tipv[3 * j + 2]);
    vec3 d = b - a;
    float len = sqrt(dot(d, d));
    vec3 mid = (a + b) / 2.0;
    vec3 view = u_persp == 1 ? mid - u_eye : u_forward;
    vec3 s = cross(d, view);
    float sn = sqrt(dot(s, s));
    s = sn > 1e-12 ? s / max(sn, 1e-12) : u_right;
    float w = world_size(S(j), mid) / 2.0;
    float hw = max(w * 2.2, len * u_head * 0.45);
    vec3 base = b - d * u_head;
    vec3 q;
    if (k == 0) q = a - s * w;
    else if (k == 1) q = a + s * w;
    else if (k == 2) q = base + s * w;
    else if (k == 3) q = base - s * w;
    else if (k == 4) q = base - s * hw;
    else if (k == 5) q = base + s * hw;
    else q = b;
    emit(q);
    v_color = C(j);
#endif
#ifdef CONE
    vec3 a = P(j);
    vec3 b = vec3(tipv[3 * j], tipv[3 * j + 1], tipv[3 * j + 2]);
    vec3 d = b - a;
    d = d / max(sqrt(dot(d, d)), 1e-12);
    vec3 pick = abs(d.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 u = cross(d, pick);
    u = u / max(sqrt(dot(u, u)), 1e-12);
    vec3 v = cross(d, u);
    float r = world_size(S(j), a) / 2.0;
    vec3 q;
    float shade;
    if (k < 8) {
        float t = TAU * float(k) / 8.0;
        q = a + r * (cos(t) * u + sin(t) * v);
        shade = 0.55;
    } else if (k == 8) {
        q = b;
        shade = 1.0;
    } else {
        q = a;
        shade = 0.4;
    }
    emit(q);
    v_color = vec4(C(j).rgb * shade, C(j).a);
#endif
}
"""
BATCH_KINDS = {("points", "disc"): "b_points", ("lines", "strip"): "b_lines",
               ("arrows", "flat"): "b_flat", ("arrows", "cone"): "b_cone"}  # fmt: skip
BATCH_BUFFERS = 512  # storage buffers kept by content (``GLRenderer._batch_buffer``)
BATCH_BYTES = 768 * 2**20  # ... and their bytes
BATCH_SPARE = 8  # evicted buffers kept for reuse instead of released
BATCH_SPARE_BYTES = 4 * 2**20  # ... only those this small (a bigger one is released)
BATCH_KEEP_RENDERS = 32  # renders (any pass) a batch buffer outlives its last use
QUALITIES = ("full", "interactive")
MATERIALS = {"solid": 0, "glass": 1, "wire": 2, "emissive": 3}
WIRE_LINES = (16.0, 8.0)  # grid lines across the surface's u and v ranges


def _light_dirs(cam, light) -> list[tuple[float, float, float]]:
    """Key, fill and rim directions (toward the light), placed relative to the camera so a
    surface is lit the same way however the camera moves."""
    out = []
    for name in ("key", "fill", "rim"):
        az = np.radians(light[f"{name}_azimuth"])
        el = np.radians(light[f"{name}_elevation"])
        d = np.cos(el) * (np.cos(az) * -cam.forward + np.sin(az) * cam.right) + np.sin(el) * cam.up
        out.append(tuple(float(x) for x in d))
    return out


def _define(src: str, kind: str) -> str:
    """``kind``: ``mesh`` (flat color), ``vmesh`` (a color per vertex, color pass only),
    ``text`` (coverage bitmap) or ``image`` (RGBA)."""
    head, _, rest = src.strip().partition("\n")
    defines = {"mesh": "", "vmesh": "#define VCOLOR", "text": "#define TEXTURED",
               "image": "#define TEXTURED\n#define IMAGE"}  # fmt: skip
    for k in BATCH_KINDS.values():  # world-space batches: colour per vertex, from storage
        defines[k] = f"#define VCOLOR\n#define BATCH\n#define {k[2:].upper()}"
    return f"{head}\n{defines[kind]}\n{rest}"


def _stroke_hue(state, pass_) -> tuple | None:
    """The theme's ``[strokes]`` hue for the color pass: (lut, cycles, shift at t)."""
    st = getattr(state, "strokes", None) or {}
    if pass_ != "color" or not st.get("hue_along") or not st.get("lut"):
        return None
    return (np.asarray(st["lut"], dtype=np.float64), float(st["hue_along"]),
            (float(st.get("flow", 0.0)) * state.t) % 1.0)  # fmt: skip


class _Collector:
    """Records an object's draw calls; the renderer replays them for the current pass.

    ``quality`` is a render hint objects may read (``ctx.quality``): ``"full"``, or
    ``"interactive"`` while a person drags (an arrow field thinned to a budget, point sets
    strided). It never changes the document; the inspector and lint always see ``"full"``."""

    def __init__(self, stroke_hue: tuple | None = None, quality: str = "full") -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        # (lut, cycles per stroke, shift): the theme's hue along glowing strokes, or None.
        self.stroke_hue = stroke_hue
        self.quality = quality

    # World-space batches (``core/batches.py``): sanitised once here; what was repaired is
    # reported by lint from the recorder, which sees the same calls.

    def _batch(self, made) -> None:
        b, _ = made
        if b is not None:
            self.commands.append(("batch", {"batch": b}))

    def points3d(self, centers, **kw) -> None:
        self._batch(batches.make_points(centers, **kw))

    def lines3d(self, points, **kw) -> None:
        if self.stroke_hue is not None and _glows(kw) and np.ndim(kw.get("colors", 0)) <= 1:
            kw = {**kw, "colors": _hue_along(points, kw.get("lengths"), self.stroke_hue,
                                             kw.get("colors", (1.0, 1.0, 1.0, 1.0)))}  # fmt: skip
        self._batch(batches.make_lines(points, **kw))

    def arrows3d(self, tails, tips, **kw) -> None:
        self._batch(batches.make_arrows(tails, tips, **kw))

    def triangles(self, positions, indices=None, *, color, glow=1.0, colors=None) -> None:
        self.commands.append(
            ("mesh", {"positions": positions, "indices": indices, "color": color, "glow": glow,
                      "colors": colors})
        )  # fmt: skip

    def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0) -> None:
        self.commands.append(
            ("field", {"corner": np.asarray(corner, dtype=np.float64), "right": np.asarray(right),
                       "up": np.asarray(up), "pixels": pixels, "opacity": opacity,
                       "glow": float(glow)})
        )  # fmt: skip

    def surface(self, positions, indices, *, normals, colors, uv=None, material="solid",
                opacity=1.0, glow=1.0) -> None:  # fmt: skip
        n = len(positions)
        self.commands.append(
            ("surface", {"positions": np.asarray(positions, dtype=np.float64),
                         "indices": np.asarray(indices), "normals": np.asarray(normals),
                         "colors": np.asarray(colors), "material": material,
                         "uv": np.zeros((n, 2)) if uv is None else np.asarray(uv),
                         "opacity": float(opacity), "glow": float(glow)})
        )  # fmt: skip

    def text(self, text, origin, height, *, color, anchor="center", glow=1.0,
             reveal=1.0) -> None:  # fmt: skip
        if reveal <= 0:
            return
        self.commands.append(
            ("text", {"text": text, "origin": origin, "height": height, "color": color,
                      "anchor": anchor, "glow": glow, "reveal": min(float(reveal), 1.0)})
        )  # fmt: skip


def _glows(kw: dict) -> bool:
    try:
        return float(kw.get("glow", 1.0)) > 1.0
    except (TypeError, ValueError):
        return False


def _hue_along(points, lengths, hue: tuple, color) -> Any:
    """The theme's ``[strokes]`` hue run along each strip of a glowing ``lines3d`` (as
    ``strokes.polylines`` runs it along a glowing stroke), ``color``'s alpha kept. Input that
    is not clean points is left as it was, for ``make_lines`` to judge."""
    from phasor.core.color import hue_colors

    try:
        pts = np.asarray(points, dtype=np.float64)
        flat = pts.reshape(-1, 3)
        if lengths is not None:
            cuts = np.cumsum(np.asarray(lengths, dtype=np.int64))[:-1]
        elif pts.ndim == 3:
            cuts = np.arange(1, pts.shape[0]) * pts.shape[1]
        else:
            cuts = []
        alpha = float(np.asarray(color, dtype=np.float64).reshape(-1)[3])
        lut, cycles, shift = hue
        runs = [hue_colors(r, lut, cycles, shift) for r in np.split(flat, cuts) if len(r)]
        out = np.concatenate(runs)
        out[:, 3] = alpha
        return out
    except (TypeError, ValueError, IndexError):
        return color


class PendingRead:
    """A color pass on its way off the card (``GLRenderer.start_color_raw``)."""

    def __init__(self, renderer: GLRenderer, packs: list, size: tuple[int, int],
                 targets: int = 2) -> None:  # fmt: skip
        self._renderer, self._packs, self.size = renderer, packs, size
        self._targets = targets  # 1: the picture only (no bloom wants the emission)

    def collect(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Wait for the copy; the picture and emission as (h, w, 4) float16, rows
        bottom-up, read-only (the emission None when it was not read). The pack buffers go
        back to the renderer's pool."""
        if self._packs is None:
            raise RuntimeError("this read was already collected or released")
        packs, self._packs = self._packs, None
        w, h = self.size
        with stage("readback"):
            data = self._renderer._finish_read(packs, self.size, read=True, targets=self._targets)
        pic, *glow = (np.frombuffer(d, dtype=np.float16).reshape(h, w, 4) for d in data)
        return pic, (glow[0] if glow else None)

    def release(self) -> None:
        """Give the buffers back unread (after an error). Harmless once collected."""
        if self._packs is not None:
            packs, self._packs = self._packs, None
            self._renderer._finish_read(packs, self.size, read=False)


ARENA_MIN_BYTES = 64 * 1024  # first size of a reused vertex or index buffer; it doubles


def _refill(buffer, data: np.ndarray) -> None:
    """Replace a reused buffer's contents with ``data``: fresh storage (orphaned, so no draw
    still reading the old contents makes us wait), grown by doubling when too small."""
    size = buffer.size
    while size < data.nbytes:
        size *= 2
    buffer.orphan(size)
    buffer.write(data)


class GLRenderer:
    """Renders SceneStates. Also satisfies ``phasor.inspect.passes.PassRunner``."""

    def __init__(self, ctx=None) -> None:
        self.ctx = ctx or shared_context()
        self._programs: dict[tuple[str, str], Any] = {}
        self._targets: dict[tuple[str, tuple[int, int]], Any] = {}
        self._text_textures: dict[tuple[str, int], Any] = {}
        self._cache: dict[tuple, np.ndarray] = {}
        self._cache_state: SceneState | None = None
        self._fields: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._emit_clear: dict[tuple[int, int], Any] = {}
        self._bridge = None
        self._arena: dict[tuple, tuple] = {}  # (program, layout, indexed) -> (vao, vbos, ibo)
        self._free_packs: dict[tuple[int, int], list] = {}  # size -> free pack-buffer pairs
        # World-space batches: storage buffers by content (dtype, bytes, hash), most recently
        # used last; spare buffers to refill; one empty vertex array per program.
        self._batch_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._batch_spare: list = []
        self._batch_used: set[tuple] = set()  # keys drawn in the current render: never evicted
        self._batch_seen: dict[tuple, int] = {}  # key -> the render that last drew it
        self._renders = 0
        self._batch_vaos: dict[Any, Any] = {}
        # Draw batches as plain triangles expanded on the CPU (``core.batches.expand``)
        # instead of in the vertex shader: the reference picture the tests compare against.
        self.expand_on_cpu = False
        self._limits: dict[str, int] | None = None  # storage-buffer limits, read once
        # Uploads, for tests and ``phasor bench``: "batch" storage buffers written (and their
        # bytes), "arena" vertex and index buffers refilled for the other draw calls.
        self.uploads = {"batch": 0, "batch_bytes": 0, "arena": 0}

    # ------------------------------------------------------------ resources

    def renderer_class(self) -> str:
        """``"software"`` (Mesa llvmpipe and friends) or ``"gpu"``. Golden images are kept
        per class because the two do not antialias identically. The query must run with
        the context current, like every other GL call."""
        name = self.on_gl_info("GL_RENDERER").lower()
        return "software" if any(k in name for k in ("llvmpipe", "softpipe", "swrast")) else "gpu"

    def on_gl_info(self, key: str) -> str:
        """A value of ``ctx.info`` (``"GL_RENDERER"``, ``"GL_VERSION"``...), read on the GL
        thread."""

        def read() -> str:
            with self.ctx:
                return str(self.ctx.info[key])

        return on_gl_thread(read)

    def _program(self, pass_: str, kind: str):
        key = (pass_, kind)
        if key == ("color", "lit") and key not in self._programs:
            self._programs[key] = self.ctx.program(vertex_shader=_VS_LIT, fragment_shader=_FS_LIT)
        if key not in self._programs:
            vs = _VS_BATCH if kind.startswith("b_") else _VS
            self._programs[key] = self.ctx.program(
                vertex_shader=_define(vs, kind), fragment_shader=_define(_FS[pass_], kind)
            )
        return self._programs[key]

    def _target(self, pass_: str, size: tuple[int, int]):
        key = (pass_, size)
        if key not in self._targets:
            ctx = self.ctx
            if pass_ == "color":
                # Two targets: the picture, and the emission that feeds the bloom.
                samples = min(4, ctx.max_samples)
                textures = [ctx.texture(size, 4, samples=samples, dtype="f2") for _ in range(2)]
                ms = ctx.framebuffer(textures, ctx.depth_renderbuffer(size, samples=samples))
                resolved = ctx.framebuffer([ctx.texture(size, 4, dtype="f2") for _ in range(2)])
                self._emit_clear[size] = ctx.framebuffer([textures[1]])
                self._targets[key] = (ms, resolved)
            elif pass_ == "id":
                fbo = ctx.framebuffer([ctx.texture(size, 1, dtype="u4")], ctx.depth_texture(size))
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

    def _vertex_array(self, prog, layout: tuple, arrays: list, indices=None):
        """A vertex array over ``arrays`` (one per ``layout`` entry ``(format, *names)``)
        and ``indices``, ready to draw, and its vertex count (GPU_SPEED.md step 3a).

        One vertex array per (program, layout, indexed) lives as long as the renderer; its
        buffers grow to the largest draw seen and are orphaned and refilled for each draw,
        so the card never waits for a draw still reading the old contents. The draw is the
        same as with fresh buffers: same data, same order, same count."""
        key = (prog, layout, indices is not None)
        if key not in self._arena:
            ctx = self.ctx
            vbos = [ctx.buffer(reserve=ARENA_MIN_BYTES) for _ in layout]
            ibo = ctx.buffer(reserve=ARENA_MIN_BYTES) if indices is not None else None
            vao = ctx.vertex_array(prog, [(b, *f) for b, f in zip(vbos, layout, strict=True)],
                                   index_buffer=ibo, index_element_size=4)  # fmt: skip
            self._arena[key] = (vao, vbos, ibo)
        vao, vbos, ibo = self._arena[key]
        for b, a in zip(vbos, arrays, strict=True):
            _refill(b, np.ascontiguousarray(a, dtype="f4"))
        self.uploads["arena"] += len(vbos) + (ibo is not None)
        if ibo is None:
            return vao, len(arrays[0])
        idx = np.ascontiguousarray(indices, dtype="i4")
        _refill(ibo, idx)
        return vao, idx.size

    def _batch_buffer(self, b: batches.Batch, name: str):
        """The storage buffer holding ``b``'s array ``name``: found by content, so an array
        drawn before (whatever the camera) is not uploaded again."""
        a = b.arrays()[name]
        key = (a.dtype.str, a.nbytes, b.digest(name))
        self._batch_used.add(key)
        self._batch_seen[key] = self._renders
        buf = self._batch_cache.get(key)
        if buf is not None:
            self._batch_cache.move_to_end(key)
            return buf
        buf = self._spare_for(a.nbytes) or self.ctx.buffer(reserve=max(a.nbytes, 16))
        _refill(buf, a)
        self.uploads["batch"] += 1
        self.uploads["batch_bytes"] += a.nbytes
        self._batch_cache[key] = buf
        self._evict_batches()
        return buf

    def _spare_for(self, nbytes: int):
        """A spare buffer that fits ``nbytes`` without being much larger (audit S2: a small
        array must not sit in a buffer of megabytes), or None."""
        fits = [b for b in self._batch_spare if b.size <= 4 * max(nbytes, ARENA_MIN_BYTES)]
        if not fits:
            return None
        best = max(fits, key=lambda b: b.size)
        self._batch_spare.remove(best)
        return best

    def _evict_batches(self) -> None:
        """Let go of buffers not drawn in the last ``BATCH_KEEP_RENDERS`` renders (a flow
        that moves makes new positions every frame: the old ones are dead), and past the
        caps the least recently used; never one drawn in the current render."""
        total = sum(x.size for x in self._batch_cache.values())
        total += sum(x.size for x in self._batch_spare)  # spares count against the cap
        for old in list(self._batch_cache):
            stale = self._renders - self._batch_seen.get(old, -1) > BATCH_KEEP_RENDERS
            over = len(self._batch_cache) > BATCH_BUFFERS or total > BATCH_BYTES
            if old in self._batch_used or not (stale or over):
                continue
            gone = self._batch_cache.pop(old)
            self._batch_seen.pop(old, None)
            total -= gone.size
            if len(self._batch_spare) < BATCH_SPARE and gone.size <= BATCH_SPARE_BYTES:
                self._batch_spare.append(gone)
                total += gone.size
            else:
                gone.release()

    def _limits_current(self) -> dict[str, int]:
        """The storage-buffer limits of this context (the context must be current): the
        largest block in bytes, and how many blocks a vertex shader may read. GL 4.3 only
        promises 16 MB and none in the vertex stage (audit S5)."""
        if self._limits is None:
            info = self.ctx.info
            self._limits = {
                "block_bytes": int(info.get("GL_MAX_SHADER_STORAGE_BLOCK_SIZE", 0) or 0),
                "vertex_blocks": int(info.get("GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS", 0) or 0),
            }
        return self._limits

    def storage_limits(self) -> dict[str, int]:
        """``{"block_bytes", "vertex_blocks"}``, read on the GL thread."""

        def read() -> dict[str, int]:
            with self.ctx:
                return dict(self._limits_current())

        return on_gl_thread(read)

    def _fits(self, b: batches.Batch) -> bool:
        """Whether the shader can draw ``b``: four storage blocks in the vertex stage and every
        array within the largest block. Otherwise it is drawn from the CPU expansion (the
        same triangles, built each frame)."""
        lim = self._limits_current()
        biggest = max(a.nbytes for a in b.arrays().values())
        return lim["vertex_blocks"] >= 4 and biggest <= lim["block_bytes"]

    def forget_batches(self) -> None:
        """Let go of every batch buffer (a benchmark starting each run from nothing)."""

        def run() -> None:
            with self.ctx:
                for buf in [*self._batch_cache.values(), *self._batch_spare]:
                    buf.release()
            self._batch_cache.clear()
            self._batch_spare.clear()
            self._batch_seen.clear()

        on_gl_thread(run)

    def batch_memory(self) -> dict[str, int]:
        """Storage buffers kept for batches (and spares) and their bytes on the card."""
        return {"buffers": len(self._batch_cache), "spares": len(self._batch_spare),
                "bytes": sum(x.size for x in self._batch_cache.values())
                + sum(x.size for x in self._batch_spare)}  # fmt: skip

    def _draw_batch(self, prog, b: batches.Batch, cam, size, flash: float) -> None:
        """One batch, instanced: its arrays bound as storage buffers, the camera as uniforms."""
        for k, name in enumerate(b.arrays()):
            self._batch_buffer(b, name).bind_to_storage_buffer(k)
        self._set_camera_uniforms(prog, b, cam, size)
        if "u_tint" in prog:
            prog["u_tint"].value = (1.0 + 0.8 * flash) if b.tint else 1.0
        vao = self._batch_vao(prog, (b.kind, b.shape))
        instances = b.n - 1 if b.kind == "lines" else b.n
        if instances > 0:
            vao.render(instances=instances)

    @staticmethod
    def _set_camera_uniforms(prog, b: batches.Batch, cam, size) -> None:
        anchor = b.anchor if b.anchor is not None else (0.0, 0.0, 0.0)
        values = {"u_eye": tuple(float(x) for x in cam.eye),
                  "u_right": tuple(float(x) for x in cam.right),
                  "u_up": tuple(float(x) for x in cam.up),
                  "u_forward": tuple(float(x) for x in cam.forward),
                  "u_persp": int(cam.perspective), "u_height": float(cam.height),
                  "u_tan": float(cam.tan_half_fov), "u_units": batches.UNITS.index(b.units),
                  "u_px": 1.0 / max(size[1], 1), "u_anchored": int(b.anchor is not None),
                  "u_anchor": anchor, "u_head": float(b.head), "u_n": int(b.n),
                  "u_each": b.each, "u_strips": int(b.strips is not None)}  # fmt: skip
        for name, value in values.items():
            if name in prog:  # (a uniform the kind does not use is compiled away)
                prog[name].value = value

    def _batch_vao(self, prog, kind: tuple[str, str]):
        """An empty vertex array (the shader pulls its data) with the kind's triangles."""
        if prog not in self._batch_vaos:
            tri = np.ascontiguousarray(batches.TEMPLATES[kind], dtype="i4")
            ibo = self.ctx.buffer(tri.tobytes())
            self._batch_vaos[prog] = self.ctx.vertex_array(prog, [], index_buffer=ibo,
                                                           index_element_size=4)  # fmt: skip
        return self._batch_vaos[prog]

    def batch_vertices(self, b: batches.Batch, cam, size: tuple[int, int]) -> np.ndarray:
        """What the vertex shader makes of ``b``: each instance's vertices in world units
        (k, v, 3), read back through transform feedback. For tests: it must equal
        ``core.batches.instances`` up to float32 rounding."""

        def run() -> np.ndarray:
            import moderngl

            with self.ctx:
                kind = (b.kind, b.shape)
                key = ("feedback", BATCH_KINDS[kind])
                if key not in self._programs:
                    self._programs[key] = self.ctx.program(
                        vertex_shader=_define(_VS_BATCH, BATCH_KINDS[kind]),
                        varyings=["v_world"])  # fmt: skip
                prog = self._programs[key]
                fbo, _ = self._target("solo", size)  # (a framebuffer must be bound)
                fbo.use()
                prog["u_vp"].write(cam.view_proj.astype("f4").T.tobytes())
                prog["u_bias"].value = 0.0
                prog["u_zoom"].value = 1.0
                self._batch_used.clear()
                for k, name in enumerate(b.arrays()):
                    self._batch_buffer(b, name).bind_to_storage_buffer(k)
                self._set_camera_uniforms(prog, b, cam, size)
                v = batches.VERTICES[kind]
                count = b.n - 1 if b.kind == "lines" else b.n
                out = self.ctx.buffer(reserve=max(count * v * 12, 12))
                vao = self.ctx.vertex_array(prog, [])
                vao.transform(out, moderngl.POINTS, vertices=v, instances=count)
                data = np.frombuffer(out.read(), dtype=np.float32)[: count * v * 3].copy()
                vao.release()
                out.release()
                return data.reshape(count, v, 3)

        return on_gl_thread(run)

    # ------------------------------------------------------------ drawing

    def _commands(self, state, obj: ObjectState, pass_, quality: str = "full") -> list:
        # id and solo passes are what the inspector measures: always the full drawing
        collector = _Collector(_stroke_hue(state, pass_), quality if pass_ == "color" else "full")
        obj.impl.draw(collector, state, pass_)
        return collector.commands

    def _draw_object(self, state, obj: ObjectState, pass_, size, order: int, zoom: float,
                     commands=None, fog: bool = True) -> None:  # fmt: skip
        cam = state.camera_for(obj)
        for kind, c in self._commands(state, obj, pass_) if commands is None else commands:
            if kind == "batch":
                b = c["batch"]
                if not self.expand_on_cpu and self._fits(b):
                    prog = self._program(pass_, BATCH_KINDS[(b.kind, b.shape)])
                    self._object_uniforms(prog, obj, pass_, cam, order, zoom,
                                          (1.0, 1.0, 1.0, b.opacity), b.glow)  # fmt: skip
                    self._draw_batch(prog, b, cam, size, float(obj.params.flash))
                    continue
                # the reference: the same triangles, expanded here, drawn as a plain mesh
                pos, idx, cols = batches.expand(b, cam, size[1])
                if b.tint and pass_ == "color":
                    cols = cols.copy()
                    cols[:, :3] *= 1.0 + 0.8 * float(obj.params.flash)
                kind, c = "mesh", {"positions": pos, "indices": idx, "glow": b.glow,
                                   "color": (1.0, 1.0, 1.0, b.opacity), "colors": cols}  # fmt: skip
                if not len(idx):
                    continue
            if kind == "surface" and pass_ == "color":
                if obj.params.flash > 0:  # the flash verb lights surfaces too
                    c = {**c, "glow": float(c.get("glow", 1.0)) * (1.0 + 1.5 * obj.params.flash)}
                self._draw_lit(state, cam, c, order, zoom, fog)
                continue
            if kind == "surface":  # id and solo: plain triangles
                kind, c = "mesh", {"positions": c["positions"], "indices": c["indices"],
                                   "color": (0, 0, 0, 1)}  # fmt: skip
            if kind == "field" and pass_ == "color":
                quad = self._field_quad(state, obj, cam, c, size, zoom)
                if quad is None:
                    continue
                kind, c = (
                    "image",
                    {
                        "quad": quad,
                        "color": (0.0, 0.0, 0.0, c["opacity"]),
                        "glow": c.get("glow", 1.0),
                    },
                )
            elif kind == "field":  # id and solo: a plain quad
                p0, r, u = c["corner"], c["right"], c["up"]
                c = {
                    "positions": np.array([p0, p0 + r, p0 + r + u, p0 + u]),
                    "indices": np.array([[0, 1, 2], [0, 2, 3]]),
                    "color": (0, 0, 0, 1),
                }
                kind = "mesh"
            if kind == "mesh" and pass_ == "color" and c.get("colors") is not None:
                kind = "vmesh"
            prog = self._program(pass_, kind)
            self._object_uniforms(prog, obj, pass_, cam, order, zoom, c["color"],
                                  c.get("glow", 1.0))  # fmt: skip
            if kind == "image":
                c["quad"][1].use(0)
                prog["u_tex"].value = 0
                vao, n = self._vertex_array(prog, (("3f 2f", "in_pos", "in_uv"),), [c["quad"][0]])
            elif kind == "text":
                corner, wv, hv = text_metrics.quad(
                    c["text"], c["origin"], c["height"], c["anchor"], cam.right, cam.up
                )
                px = round(c["height"] / cam.units_per_height(c["origin"]) * size[1])
                tex = self._text_texture(c["text"], int(np.clip(px, 2, MAX_TEXT_PX)))
                tex.use(0)
                prog["u_tex"].value = 0
                # Bitmap row 0 is the top of the text, so the top corners take v = 0. A
                # partial ``reveal`` (writing it in) shows the left share of the quad.
                r = float(c.get("reveal", 1.0))
                wr = wv * r
                quad = np.array(
                    [
                        [*corner, 0, 1],
                        [*(corner + wr), r, 1],
                        [*(corner + hv), 0, 0],
                        [*(corner + wr), r, 1],
                        [*(corner + wr + hv), r, 0],
                        [*(corner + hv), 0, 0],
                    ],
                    dtype="f4",
                )
                vao, n = self._vertex_array(prog, (("3f 2f", "in_pos", "in_uv"),), [quad])
            elif kind == "vmesh":
                layout = (("3f", "in_pos"), ("4f", "in_color"))
                vao, n = self._vertex_array(prog, layout, [c["positions"], c["colors"]],
                                            c["indices"])  # fmt: skip
            else:
                vao, n = self._vertex_array(prog, (("3f", "in_pos"),), [c["positions"]],
                                            c["indices"])  # fmt: skip
            vao.render(vertices=n)

    @staticmethod
    def _object_uniforms(prog, obj, pass_, cam, order: int, zoom: float, color, glow) -> None:
        """The uniforms every flat draw shares: camera, depth bias, zoom, and the color
        pass's color and glow (with the flash verb) or the id pass's object index."""
        prog["u_vp"].write(cam.view_proj.astype("f4").T.tobytes())
        prog["u_bias"].value = order * DEPTH_BIAS
        prog["u_zoom"].value = 1.0 / zoom
        if pass_ == "color":
            flash = float(obj.params.flash)  # the flash verb: more light, more glow
            rgb = [float(x) * (1.0 + 0.8 * flash) for x in color[:3]]
            prog["u_color"].value = (*rgb, float(color[3]))
            prog["u_glow"].value = float(glow) * (1.0 + 1.5 * flash)
        elif pass_ == "id":
            prog["u_id"].value = obj.index

    def _draw_lit(self, state, cam, c, order: int, zoom: float, fog: bool = True) -> None:
        prog = self._program("color", "lit")
        light = state.light
        prog["u_vp"].write(cam.view_proj.astype("f4").T.tobytes())
        prog["u_bias"].value = order * DEPTH_BIAS
        prog["u_zoom"].value = 1.0 / zoom
        prog["u_eye"].value = tuple(float(x) for x in cam.eye)
        prog["u_light_dir"].write(np.array(_light_dirs(cam, light), dtype="f4").tobytes())
        prog["u_light"].write(np.array([light["key"], light["fill"], light["rim"]],
                                       dtype="f4").tobytes())  # fmt: skip
        for name in ("ambient", "specular", "shininess", "fog"):
            prog[f"u_{name}"].value = float(light[name])
        if not fog:  # a draft: no depth fog (render.presets.Effects)
            prog["u_fog"].value = 0.0
        prog["u_fog_ref"].value = float(cam.height if cam.perspective else 100.0)
        prog["u_bg"].value = tuple(float(x) for x in state.background)
        prog["u_opacity"].value = c["opacity"]
        prog["u_glow"].value = c["glow"]
        prog["u_material"].value = MATERIALS.get(c["material"], 0)
        prog["u_grid"].value = WIRE_LINES
        n = len(c["positions"])
        data = np.empty((n, 12), dtype="f4")
        data[:, 0:3], data[:, 3:6] = c["positions"], c["normals"]
        data[:, 6:10], data[:, 10:12] = c["colors"], c["uv"]
        layout = (("3f 3f 4f 2f", "in_pos", "in_normal", "in_rgba", "in_uv"),)
        vao, count = self._vertex_array(prog, layout, [data], c["indices"])
        vao.render(vertices=count)

    def _field_quad(self, state, obj, cam, c, size, zoom):
        """Compute a kernel-drawn quad's visible pixels and upload them through the bridge.
        Returns (vertices with uv, texture), or None when none of it is on screen."""
        from phasor.render.gpu.bridge import CopyBridge

        p0, r, u = c["corner"], c["right"], c["up"]
        w, h = size
        corners = np.array([p0, p0 + r, p0 + u])
        ndc = cam.project(corners)[:, :2] / zoom
        px = np.c_[(ndc[:, 0] + 1) / 2 * w, (1 - ndc[:, 1]) / 2 * h]
        a, b = px[1] - px[0], px[2] - px[0]  # screen motion along right and up
        s0, s1, t0, t1 = 0.0, 1.0, 0.0, 1.0
        if abs(a[1]) < 1e-9 and abs(b[0]) < 1e-9 and a[0] != 0 and b[1] != 0:
            # Axis-aligned on screen (every 2D scene): render only the part in the frame.
            sa = sorted(((0 - px[0][0]) / a[0], (w - px[0][0]) / a[0]))
            tb = sorted(((0 - px[0][1]) / b[1], (h - px[0][1]) / b[1]))
            s0, s1 = max(0.0, sa[0]), min(1.0, sa[1])
            t0, t1 = max(0.0, tb[0]), min(1.0, tb[1])
            if s1 <= s0 or t1 <= t0:
                return None
        # ceil, but a size that is a whole number up to rounding noise stays that number
        pw = int(np.clip(np.ceil(np.hypot(*a) * (s1 - s0) - 1e-6), 1, MAX_FIELD_PX))
        ph = int(np.clip(np.ceil(np.hypot(*b) * (t1 - t0) - 1e-6), 1, MAX_FIELD_PX))
        rl, ul = float(np.linalg.norm(r)), float(np.linalg.norm(u))
        center = ((s0 + s1) / 2 * rl, (t0 + t1) / 2 * ul)
        height = (t1 - t0) * ul
        # The background is in the key: fields fill their empty parts with it (Fractal,
        # cloud, image), and a scene may choose its own (M18).
        # ``cache_key`` on the pixels function: what else its picture depends on (a video's
        # file and frame, Milestone 24), so a clip that plays is not served one frame.
        key = (obj.type, obj.params, tuple(sorted(obj.roles.items())), pw, ph,
               round(center[0], 12), round(center[1], 12), round(height, 15),
               tuple(state.background), getattr(c["pixels"], "cache_key", None))  # fmt: skip
        # ``uncached`` set on it by the call: these pixels stand in for a picture that could
        # not be had this time (a frame a decoder failed to give): drawn, never kept.
        pixels = self._fields.get(key)
        if pixels is None:
            pixels = np.asarray(c["pixels"](pw, ph, center, height), dtype=np.float32)
            if pixels.shape != (ph, pw, 4):
                raise ValueError(
                    f"{obj.id}: field pixels must be {(ph, pw, 4)}, got {pixels.shape}"
                )
            if not getattr(c["pixels"], "uncached", False):
                self._fields[key] = pixels
                if len(self._fields) > FIELD_CACHE:
                    self._fields.popitem(last=False)
        if key in self._fields:
            self._fields.move_to_end(key)
        if self._bridge is None:
            self._bridge = CopyBridge(self.ctx)
        tex = self._bridge.upload(pixels)
        q = [p0 + s * r + t * u for s, t in ((s0, t0), (s1, t0), (s0, t1), (s1, t1))]
        uv = [(0, 0), (1, 0), (0, 1), (1, 1)]  # the bridge stores rows bottom-up: v = 0 is low
        order_ = (0, 1, 2, 1, 3, 2)
        verts = np.array([[*q[i], *uv[i]] for i in order_], dtype="f4")
        return verts, tex

    def _render(self, state, pass_, size, **kwargs) -> np.ndarray:
        with stage("draw"):
            return on_gl_thread(self._render_on_gl_thread, state, pass_, size, **kwargs)

    def _render_on_gl_thread(self, state, pass_, size, **kwargs) -> np.ndarray:
        with self.ctx:  # make our context current: another one may have been used since
            return self._render_current(state, pass_, size, **kwargs)

    def _render_current(
        self, state, pass_, size, *, only=None, force=frozenset(), zoom=1.0, raw=False,
        fog=True, emission=True, plan=None, quality="full",
    ) -> np.ndarray:  # fmt: skip
        import moderngl

        ctx = self.ctx
        self._batch_used.clear()
        self._renders += 1
        if self._renders % 8 == 0:
            self._evict_batches()
        target, resolved = self._target(pass_, size)
        target.use()
        ctx.viewport = (0, 0, *size)
        if pass_ == "color":
            target.clear(*state.background)
            self._emit_clear[size].clear(0.0, 0.0, 0.0, 0.0)  # no emission until drawn
            target.use()
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        else:
            target.clear(0, 0, 0, 0)
            ctx.disable(moderngl.BLEND)
        if pass_ == "solo":
            ctx.disable(moderngl.DEPTH_TEST)
        else:
            ctx.enable(moderngl.DEPTH_TEST)
        deferred = []  # see-through surfaces: drawn after everything opaque, far to near
        for order, obj in enumerate(state.draw_order()):
            if only is not None and obj.id != only:
                continue
            if obj.params.opacity <= 0:
                continue
            if pass_ == "id" and not occludes(obj) and obj.id not in force:
                continue
            if plan is not None and pass_ == "color":  # recorded already (``frame_plan``)
                commands = plan.commands[order]
            else:
                commands = self._commands(state, obj, pass_, quality)
            if pass_ == "color" and obj.layer != "overlay":
                clear = [c for c in commands if c[0] == "surface"
                         and (c[1]["opacity"] < 0.999 or c[1]["material"] == "glass")]  # fmt: skip
                if clear:
                    cam = state.camera_for(obj)
                    for c in clear:
                        deferred.append((float(cam.depth(c[1]["positions"]).mean()), order, obj, c))
                    commands = [c for c in commands if all(c is not d for d in clear)]
            if obj.layer == "overlay" and deferred:
                self._draw_deferred(state, deferred, size, zoom, fog)
            # The overlay layer always sits on top of the world: it ignores world depth.
            ctx.depth_func = "1" if obj.layer == "overlay" else "<="
            self._draw_object(state, obj, pass_, size, order, zoom, commands, fog)
        if deferred:
            self._draw_deferred(state, deferred, size, zoom, fog)
        if resolved is not target:
            ctx.copy_framebuffer(resolved, target)
        if raw != "async":
            # Measured runs only: wait for the card to finish drawing here, so the read
            # below is charged for the copy alone (a read otherwise waits for the drawing
            # too). An asynchronous read must not wait: that is its point.
            with stage("gpu draw", sync=ctx.finish):
                pass
        return self._read(resolved, pass_, size, raw=raw, emission=emission)

    def _read(self, resolved, pass_, size, raw=False, emission=True):
        w, h = size
        if pass_ == "color" and raw:  # as stored: float16, rows bottom-up, not widened
            free = self._free_packs.setdefault(size, [])
            packs = free.pop() if free else [self.ctx.buffer(reserve=w * h * 8) for _ in (0, 1)]
            targets = 2 if emission else 1  # no bloom: the emission is not needed
            for buffer, attachment in zip(packs[:targets], (0, 1)[:targets], strict=True):
                resolved.read_into(buffer, components=4, dtype="f2", attachment=attachment)
            return PendingRead(self, packs, size, targets)
        if pass_ == "color":
            out = []
            for attachment in (0, 1):
                with stage("readback"):
                    raw = resolved.read(components=4, dtype="f2", attachment=attachment)
                with stage("widen"):  # float16 rows bottom-up -> float32 rows top-down
                    px = np.frombuffer(raw, dtype=np.float16).reshape(h, w, 4)[::-1]
                    out.append(px.astype(np.float32))
            with stage("widen"):
                return np.stack(out)
        if pass_ == "id":
            raw = np.frombuffer(resolved.read(components=1, dtype="u4"), dtype=np.uint32)
            return raw.reshape(h, w)[::-1].copy()
        raw = np.frombuffer(resolved.read(components=1, dtype="f1"), dtype=np.uint8)
        return raw.reshape(h, w)[::-1] > 127

    def _draw_deferred(self, state, deferred: list, size, zoom, fog: bool = True) -> None:
        """Translucent surfaces, tested against depth but not writing it, drawn far to near:
        objects by depth, and each surface's own triangles by depth, so a glass surface that
        folds over itself (two sheets) blends in the right order. Sorting replaces OIT; it is
        exact except where translucent triangles cut through each other."""
        target, _ = self._target("color", size)
        self.ctx.depth_func = "<="
        target.depth_mask = False
        for _, order, obj, cmd in sorted(deferred, key=lambda d: (-d[0], d[1])):
            if cmd[0] == "surface":  # its own triangles, far to near too
                c = dict(cmd[1])
                tris = np.asarray(c["indices"])
                centroid = c["positions"][tris].mean(axis=1)
                c["indices"] = tris[
                    np.argsort(-state.camera_for(obj).depth(centroid), kind="stable")
                ]
                cmd = ("surface", c)
            self._draw_object(state, obj, "color", size, order, zoom, [cmd], fog)
        target.depth_mask = True
        deferred.clear()

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
        """(h, w, 4) float32 linear RGBA, row 0 = top: the picture before bloom."""
        return self.color_and_emission(state, size)[0]

    def color_and_emission(
        self, state: SceneState, size: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The picture and its emission (the light glow roles add, which feeds the bloom),
        both (h, w, 4) float32 linear, row 0 = top."""
        size = size or (state.width, state.height)
        both = self._render(state, "color", size)
        return both[0], both[1]

    def color_and_emission_raw(
        self, state: SceneState, size: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``color_and_emission`` as the targets store it: (h, w, 4) float16, rows
        bottom-up (row 0 = the bottom of the frame), read-only. For sums made on the card
        (``Engine.frame_blurred``), which widen and flip there instead of on the host."""
        return self.start_color_raw(state, size).collect()

    def start_color_raw(
        self, state: SceneState, size: tuple[int, int] | None = None, *, fog: bool = True,
        emission: bool = True, plan: FramePlan | None = None, quality: str = "full",
    ) -> PendingRead:  # fmt: skip
        """Draw the color pass and start copying both targets off the card, without
        waiting: ``collect()`` on the result gives what ``color_and_emission_raw`` gives.
        Start several before collecting any, and the card copies one frame while the next
        is being built. Every started read must be collected (or ``release``d).
        A draft (``render.presets.DRAFT``) passes ``fog=False`` (no depth fog on lit
        surfaces) and ``emission=False`` (the emission target is not read: no bloom uses
        it); ``plan`` is this state's ``frame_plan``, so its objects are not drawn twice.
        ``quality``: ``"full"`` or ``"interactive"`` (a hint objects may read while a person
        drags: fewer arrows, strided point sets; ``_Collector``)."""
        size = size or (state.width, state.height)
        _check_quality(quality)
        if plan is not None and (plan.state is not state or plan.size != size
                                 or plan.quality != quality):  # fmt: skip
            raise ValueError("this frame plan was made for another state, size or quality")
        return self._render(state, "color", size, raw="async", fog=fog, emission=emission,
                            plan=plan, quality=quality)  # fmt: skip

    def frame_plan(self, state: SceneState, size: tuple[int, int] | None = None, *,
                   fog: bool = True, quality: str = "full") -> FramePlan:  # fmt: skip
        """Record what the color pass will draw for ``state`` at ``size`` (each object's draw
        calls) and a digest of everything that drawing depends on: the calls, the draw
        order, each object's layer, opacity, flash and camera, the light, the background,
        the theme's stroke hue, the size, ``fog`` and ``quality``. Two plans with the same
        ``key`` draw the same picture, so a render may repeat the first frame instead of the
        second (``render.job``'s de-duplication). Kernel pictures (``field``) are keyed the
        way the renderer caches them: by the object's type, parameters, roles and style,
        its quad, and the background, not by the pixels themselves. Needs no GL."""
        from phasor.render.framekey import digest

        size = size or (state.width, state.height)
        _check_quality(quality)
        commands: dict[int, list] = {}
        parts: list[Any] = [size, bool(fog), tuple(state.background), state.light,
                            _stroke_hue(state, "color"), quality]  # fmt: skip
        for order, obj in enumerate(state.draw_order()):
            if obj.params.opacity <= 0:
                continue
            cmds = self._commands(state, obj, "color", quality)
            commands[order] = cmds
            keyed = [(kind, {k: v for k, v in c.items() if k != "pixels"}) for kind, c in cmds]
            fields = any(kind == "field" for kind, _ in cmds)
            parts.append((order, obj.id, obj.type, obj.layer, float(obj.params.opacity),
                          float(obj.params.flash), state.camera_for(obj), keyed,
                          (obj.params, obj.roles, obj.style) if fields else None))  # fmt: skip
        return FramePlan(state, size, digest(*parts), commands, quality)

    def _finish_read(self, packs: list, size, read: bool,
                     targets: int = 2) -> list[bytes] | None:  # fmt: skip
        def run():
            with self.ctx:
                data = [b.read() for b in packs[:targets]] if read else None
            self._free_packs.setdefault(size, []).append(packs)
            return data

        return on_gl_thread(run)

    def id_buffer(self, state, size, force: frozenset[str] = frozenset()) -> np.ndarray:
        return self._cached(
            state, ("id", size, force), lambda: self._render(state, "id", size, force=force)
        )

    def depth_buffer(self, state, size) -> np.ndarray:
        def make():
            world = dataclasses.replace(
                state, objects=tuple(o for o in state.objects if o.layer != "overlay")
            )

            def draw():
                with self.ctx:
                    self._render_current(world, "id", size)
                    fbo, _ = self._target("id", size)
                    return fbo.read(components=1, attachment=-1, dtype="f4")

            raw = on_gl_thread(draw)
            w, h = size
            z = np.frombuffer(raw, dtype=np.float32).reshape(h, w)[::-1].astype(np.float64)
            return _view_depth(state.camera, z, size)

        return self._cached(state, ("depth", size), make)

    def solo_mask(self, state, size, object_id: str, zoom_out: float = 1.0) -> np.ndarray:
        return self._cached(
            state,
            ("solo", size, object_id, round(zoom_out, 6)),
            lambda: self._render(state, "solo", size, only=object_id, zoom=zoom_out),
        )


def _view_depth(cam, z01: np.ndarray, size) -> np.ndarray:
    """Depth-buffer values (0..1) -> distance along the view direction; inf where empty."""
    w, h = size
    xs = (np.arange(w) + 0.5) / w * 2 - 1
    ys = 1 - (np.arange(h) + 0.5) / h * 2
    xx, yy = np.meshgrid(xs, ys)
    ndc = np.stack([xx, yy, z01 * 2 - 1, np.ones_like(xx)], axis=-1).reshape(-1, 4)
    world = ndc @ np.linalg.inv(cam.view_proj).T
    world = world[:, :3] / world[:, 3:4]
    d = ((world - cam.eye) @ cam.forward).reshape(h, w)
    d[z01 >= 1.0] = np.inf
    return d


@dataclasses.dataclass(frozen=True, eq=False)
class FramePlan:
    """What ``GLRenderer.frame_plan`` recorded: the state and size it is for, the digest of
    the drawing (``key``) and each drawn object's commands by draw-order index."""

    state: SceneState
    size: tuple[int, int]
    key: bytes
    commands: dict[int, list]
    quality: str = "full"


def _check_quality(quality: str) -> None:
    if quality not in QUALITIES:
        raise ValueError(f"quality is one of {', '.join(QUALITIES)}, not {quality!r}")


@functools.cache
def default_renderer() -> GLRenderer:
    """The process-wide renderer on the shared context. Raises GLUnavailableError on tier C."""
    return GLRenderer()
