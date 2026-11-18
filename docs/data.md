# Example data

The example pages use pushover curves of typical **concentrically braced
frames (CBFs) designed to the second-generation Eurocode 8 (EC8 gen2)**. They
are in the
[`examples/data`](https://github.com/nc-hsu/fitpo/tree/main/examples/data)
folder of the repository; they are not installed with the package.

## File format

- CSV, no header row.
- Column 1: roof displacement [mm].
- Column 2: base shear [N].

## Folders

| Folder | Contents | Used on |
|---|---|---|
| `cbf_cusum/` | Monotonic pushover curves with brace buckling and fracture, densely sampled. They start slightly away from (0, 0), so the examples use `origin="correct_disp"`. | {doc}`examples/cbf_cusum`, {doc}`examples/cbf_knots`, {doc}`examples/deluca_vs_cbf`, {doc}`examples/deluca_modes`, {doc}`examples/origin` |
| `cbf_knots/` | Sparse monotonic envelopes (about a dozen points each). | {doc}`examples/cbf_knots` |
| `bilinear/` | Monotonic pushover envelopes. | {doc}`examples/bilinear_modes` |
| `cyclic/` | One cyclic pushover curve. | {doc}`examples/envelopes` |

## Licence

The data are released under GPL-3.0-or-later, like the rest of the
repository.
