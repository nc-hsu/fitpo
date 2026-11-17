# Example data

The curves in this folder are pushover curves of typical **concentrically
braced frames (CBFs) designed to the second-generation Eurocode 8 (EC8 gen2)**.

## File format

- CSV, no header row.
- Column 1: roof displacement [mm].
- Column 2: base shear [N].

## Folders

- `cbf_cusum/`: monotonic pushover curves with brace buckling and fracture,
  used to show `fit_cbf_cusum_backbone` and to compare it with the De Luca fits.
- `cbf_knots/`: monotonic pushover curves used to show `fit_cbf_knots_backbone`.
- `bilinear/`: monotonic pushover curves used to show `fit_bilinear_backbone`.
- `cyclic/`: one cyclic pushover curve, used to show `fit_envelope`.

## Licence

These data are released under GPL-3.0-or-later, like the rest of this
repository.
