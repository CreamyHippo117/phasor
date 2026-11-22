# Waiting for the owner's OK

Changes made but not applied, so they are not live until approved.

- `prism_rest.patch`: the prism look for the remaining types (Cayley tables, clock diagrams,
  matrices, the room, the 3D surfaces as glass). Axes and photos stay plain. Before (left) and
  after (right): `prism_rest_before_after.png`. To apply: `git apply pending/prism_rest.patch`,
  then run the tests and delete this entry.
