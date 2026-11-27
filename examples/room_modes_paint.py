"""A note's pressure squared painted on a room's walls (BEAUTY.md step 7, `Room3D(paint=)`).

The room is 5 x 4 x 3 m. The formula is the pressure squared of a room mode (l, m, n): bright
where the note's pressure swings most, dark on its nodal lines. The axial note (1, 0, 0) is
bright on the two end walls and fades along the four side walls: it touches two pairs of
walls only lightly, so it rings longer. The oblique note (1, 1, 1) is patchier on every wall.
Animating the mode numbers glides from one to the other with no flicker.

Run:  uv run python examples/room_modes_paint.py
"""

from phasor import Scene, ac

LX, LY, LZ = 5.0, 4.0, 3.0
NOTE = "(cos(pi*l*x/Lx) * cos(pi*m*y/Ly) * cos(pi*n*z/Lz))^2"

s = Scene("room_modes_paint", look="prism")
s.use_camera("orbit", azimuth=-52, elevation=24, distance=12.5, target=(LX / 2, LY / 2, LZ / 2))
s.add(ac.Room3D(id="room", size=(LX, LY, LZ), material="plaster", paint=NOTE, mode=(1, 0, 0)))
s.marker("axial", 0.0)
s.marker("oblique", 6.0)
s.marker("end", 10.0)
s.at("axial").to("room.mode", (1, 1, 1), duration=6.0)

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 3.0, 6.0):
        p = e.inspect(t, id="room")["objects"][0]["semantic"]["paint"]
        print(f"t={t:g}: mode {p['mode']}, mean per wall {p['mean']}")
