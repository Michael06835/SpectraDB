# B4.4-SUPPLEMENT — Test Representative Molecules (B0 §7.6)

> **Post-hoc descriptive supplement.** Selection uses the pre-specified B0 §7.6 rule with Seed 42 and threshold 0.5. It is qualitative only, is not XAI, and does not constitute confirmatory evidence.

## Selection rule

- Correct sample: all three modalities exactly match every label; choose the sample whose confidence is closest to the subgroup's median confidence (ties: row_index ascending).
- Error sample: at least one modality/label is wrong; choose the highest-confidence error sample (ties: row_index ascending).
- Confidence per sample = mean over 3 modalities × all task labels of max(p, 1−p).
- Subgroup definitions are the frozen D1/B0 §7.5 definitions; support tables are separate from case selection.

## Coverage

| Task | Non-empty subgroups | Correct cases | Error cases | Empty/non-ok statuses |
|---|---|---|---|---|
| Shared14 | 22 | 22 | 22 | 4 |
| UVSPEC15 | 22 | 22 | 22 | 4 |

## Examples

| Task | Subgroup | Case | row_index | Confidence | Subgroup median | Exact (IR/Raman/UV) |
|---|---|---|---|---|---|---|
| Shared14 | ring_count=0 | correct | 119771 | 0.9553 | 0.9553 | Y/Y/Y |
| Shared14 | ring_count=0 | error | 4 | 0.9973 | 0.9553 | N/N/N |
| Shared14 | ring_count=1 | correct | 29716 | 0.9548 | 0.9551 | Y/Y/Y |
| Shared14 | ring_count=1 | error | 112436 | 0.9915 | 0.9551 | Y/Y/N |
| Shared14 | ring_count=2 | correct | 28902 | 0.9560 | 0.9560 | Y/Y/Y |
| Shared14 | ring_count=2 | error | 13765 | 0.9919 | 0.9560 | Y/N/Y |
| Shared14 | ring_count=>=3 | correct | 66238 | 0.9488 | 0.9489 | Y/Y/Y |
| Shared14 | ring_count=>=3 | error | 41338 | 0.9923 | 0.9489 | Y/Y/N |
| Shared14 | aromatic_ring_count=0 | correct | 32039 | 0.9540 | 0.9540 | Y/Y/Y |
| Shared14 | aromatic_ring_count=0 | error | 4 | 0.9973 | 0.9540 | N/N/N |
| Shared14 | aromatic_ring_count=1 | correct | 122082 | 0.9528 | 0.9528 | Y/Y/Y |
| Shared14 | aromatic_ring_count=1 | error | 126811 | 0.9896 | 0.9528 | Y/Y/N |
| Shared14 | aromatic_ring_count=>=2 | correct | 121897 | 0.9555 | 0.9555 | Y/Y/Y |
| Shared14 | aromatic_ring_count=>=2 | error | 23665 | 0.9793 | 0.9555 | Y/Y/N |
| Shared14 | n_o_f=0 | correct | 87980 | 0.9650 | 0.9651 | Y/Y/Y |
| Shared14 | n_o_f=0 | error | 1787 | 0.9903 | 0.9651 | Y/N/Y |
| Shared14 | n_o_f=1 | correct | 86662 | 0.9606 | 0.9606 | Y/Y/Y |
| Shared14 | n_o_f=1 | error | 4 | 0.9973 | 0.9606 | N/N/N |
| Shared14 | n_o_f=2 | correct | 49151 | 0.9554 | 0.9554 | Y/Y/Y |
| Shared14 | n_o_f=2 | error | 41338 | 0.9923 | 0.9554 | Y/Y/N |
| Shared14 | n_o_f=>=3 | correct | 123035 | 0.9469 | 0.9470 | Y/Y/Y |
| Shared14 | n_o_f=>=3 | error | 1360 | 0.9933 | 0.9470 | Y/Y/N |
| Shared14 | heavy_atom_count=Q1 | correct | 43675 | 0.9537 | 0.9537 | Y/Y/Y |
| Shared14 | heavy_atom_count=Q1 | error | 4 | 0.9973 | 0.9537 | N/N/N |
| Shared14 | rotatable_bond_count=0 | correct | 36012 | 0.9510 | 0.9510 | Y/Y/Y |
| Shared14 | rotatable_bond_count=0 | error | 4 | 0.9973 | 0.9510 | N/N/N |
| Shared14 | rotatable_bond_count=1 | correct | 105983 | 0.9553 | 0.9554 | Y/Y/Y |
| Shared14 | rotatable_bond_count=1 | error | 1360 | 0.9933 | 0.9554 | Y/Y/N |
| Shared14 | rotatable_bond_count=>=2 | correct | 94746 | 0.9585 | 0.9585 | Y/Y/Y |
| Shared14 | rotatable_bond_count=>=2 | error | 5627 | 0.9952 | 0.9585 | Y/Y/N |
| Shared14 | fraction_csp3=<0.5 | correct | 30344 | 0.9507 | 0.9505 | Y/Y/Y |
| Shared14 | fraction_csp3=<0.5 | error | 4 | 0.9973 | 0.9505 | N/N/N |
| Shared14 | fraction_csp3=>=0.5 | correct | 108838 | 0.9545 | 0.9545 | Y/Y/Y |
| Shared14 | fraction_csp3=>=0.5 | error | 5627 | 0.9952 | 0.9545 | Y/Y/N |
| Shared14 | functional_group=carbonyl | correct | 65219 | 0.9466 | 0.9465 | Y/Y/Y |
| Shared14 | functional_group=carbonyl | error | 1360 | 0.9933 | 0.9465 | Y/Y/N |
| Shared14 | functional_group=nitrile | correct | 2418 | 0.9690 | 0.9546 | Y/Y/Y |
| Shared14 | functional_group=nitrile | error | 4 | 0.9973 | 0.9546 | N/N/N |
| Shared14 | functional_group=alkyne | correct | 51782 | 0.9492 | 0.9490 | Y/Y/Y |
| Shared14 | functional_group=alkyne | error | 126789 | 0.9837 | 0.9490 | Y/Y/N |
| Shared14 | functional_group=amine | correct | 122526 | 0.9460 | 0.9460 | Y/Y/Y |
| Shared14 | functional_group=amine | error | 26342 | 0.9854 | 0.9460 | Y/Y/N |
| Shared14 | functional_group=hydroxyl | correct | 83727 | 0.9555 | 0.9555 | Y/Y/Y |
| Shared14 | functional_group=hydroxyl | error | 118709 | 0.9931 | 0.9555 | Y/Y/N |
| UVSPEC15 | ring_count=0 | correct | 8489 | 0.9898 | 0.9898 | Y/Y/Y |
| UVSPEC15 | ring_count=0 | error | 129434 | 0.9994 | 0.9898 | Y/Y/N |
| UVSPEC15 | ring_count=1 | correct | 31468 | 0.9534 | 0.9534 | Y/Y/Y |
| UVSPEC15 | ring_count=1 | error | 93681 | 0.9990 | 0.9534 | Y/Y/N |
| UVSPEC15 | ring_count=2 | correct | 79416 | 0.9941 | 0.9941 | Y/Y/Y |
| UVSPEC15 | ring_count=2 | error | 88705 | 0.9992 | 0.9941 | Y/Y/N |
| UVSPEC15 | ring_count=>=3 | correct | 77165 | 0.9913 | 0.9913 | Y/Y/Y |
| UVSPEC15 | ring_count=>=3 | error | 65226 | 0.9979 | 0.9913 | Y/Y/N |
| UVSPEC15 | aromatic_ring_count=0 | correct | 13654 | 0.9938 | 0.9938 | Y/Y/Y |
| UVSPEC15 | aromatic_ring_count=0 | error | 129434 | 0.9994 | 0.9938 | Y/Y/N |
| UVSPEC15 | aromatic_ring_count=1 | correct | 122119 | 0.9422 | 0.9421 | Y/Y/Y |
| UVSPEC15 | aromatic_ring_count=1 | error | 3791 | 0.9978 | 0.9421 | Y/Y/N |
| UVSPEC15 | aromatic_ring_count=>=2 | correct | 125242 | 0.9417 | 0.9422 | Y/Y/Y |
| UVSPEC15 | aromatic_ring_count=>=2 | error | 123764 | 0.9774 | 0.9422 | Y/Y/N |
| UVSPEC15 | n_o_f=0 | correct | 87857 | 0.9977 | 0.9977 | Y/Y/Y |
| UVSPEC15 | n_o_f=0 | error | 119578 | 0.9885 | 0.9977 | Y/N/Y |
| UVSPEC15 | n_o_f=1 | correct | 57578 | 0.9973 | 0.9973 | Y/Y/Y |
| UVSPEC15 | n_o_f=1 | error | 37287 | 0.9944 | 0.9973 | Y/Y/N |
| UVSPEC15 | n_o_f=2 | correct | 85180 | 0.9937 | 0.9937 | Y/Y/Y |
| UVSPEC15 | n_o_f=2 | error | 88705 | 0.9992 | 0.9937 | Y/Y/N |
| UVSPEC15 | n_o_f=>=3 | correct | 52972 | 0.9719 | 0.9719 | Y/Y/Y |
| UVSPEC15 | n_o_f=>=3 | error | 129434 | 0.9994 | 0.9719 | Y/Y/N |
| UVSPEC15 | heavy_atom_count=Q1 | correct | 79333 | 0.9897 | 0.9897 | Y/Y/Y |
| UVSPEC15 | heavy_atom_count=Q1 | error | 129434 | 0.9994 | 0.9897 | Y/Y/N |
| UVSPEC15 | rotatable_bond_count=0 | correct | 85609 | 0.9873 | 0.9873 | Y/Y/Y |
| UVSPEC15 | rotatable_bond_count=0 | error | 88705 | 0.9992 | 0.9873 | Y/Y/N |
| UVSPEC15 | rotatable_bond_count=1 | correct | 56978 | 0.9912 | 0.9912 | Y/Y/Y |
| UVSPEC15 | rotatable_bond_count=1 | error | 129434 | 0.9994 | 0.9912 | Y/Y/N |
| UVSPEC15 | rotatable_bond_count=>=2 | correct | 58778 | 0.9940 | 0.9940 | Y/Y/Y |
| UVSPEC15 | rotatable_bond_count=>=2 | error | 113991 | 0.9989 | 0.9940 | Y/Y/N |
| UVSPEC15 | fraction_csp3=<0.5 | correct | 21270 | 0.9400 | 0.9399 | Y/Y/Y |
| UVSPEC15 | fraction_csp3=<0.5 | error | 3791 | 0.9978 | 0.9399 | Y/Y/N |
| UVSPEC15 | fraction_csp3=>=0.5 | correct | 79443 | 0.9941 | 0.9941 | Y/Y/Y |
| UVSPEC15 | fraction_csp3=>=0.5 | error | 129434 | 0.9994 | 0.9941 | Y/Y/N |
| UVSPEC15 | functional_group=carbonyl | correct | 63995 | 0.9844 | 0.9844 | Y/Y/Y |
| UVSPEC15 | functional_group=carbonyl | error | 99124 | 0.9984 | 0.9844 | Y/Y/N |
| UVSPEC15 | functional_group=nitrile | correct | 35249 | 0.9920 | 0.9920 | Y/Y/Y |
| UVSPEC15 | functional_group=nitrile | error | 113991 | 0.9989 | 0.9920 | Y/Y/N |
| UVSPEC15 | functional_group=alkyne | correct | 76917 | 0.9826 | 0.9826 | Y/Y/Y |
| UVSPEC15 | functional_group=alkyne | error | 99321 | 0.9937 | 0.9826 | Y/Y/N |
| UVSPEC15 | functional_group=amine | correct | 32460 | 0.9733 | 0.9733 | Y/Y/Y |
| UVSPEC15 | functional_group=amine | error | 46293 | 0.9989 | 0.9733 | Y/Y/N |
| UVSPEC15 | functional_group=hydroxyl | correct | 108517 | 0.9951 | 0.9951 | Y/Y/Y |
| UVSPEC15 | functional_group=hydroxyl | error | 113991 | 0.9989 | 0.9951 | Y/Y/N |

## Status and caveats

- Shared14 `heavy_atom_count=Q2`: n=0, correct=empty_subgroup, error=empty_subgroup.
- Shared14 `heavy_atom_count=Q3`: n=0, correct=empty_subgroup, error=empty_subgroup.
- Shared14 `heavy_atom_count=Q4`: n=0, correct=empty_subgroup, error=empty_subgroup.
- Shared14 `functional_group=aromatic`: n=0, correct=empty_subgroup, error=empty_subgroup.
- UVSPEC15 `heavy_atom_count=Q2`: n=0, correct=empty_subgroup, error=empty_subgroup.
- UVSPEC15 `heavy_atom_count=Q3`: n=0, correct=empty_subgroup, error=empty_subgroup.
- UVSPEC15 `heavy_atom_count=Q4`: n=0, correct=empty_subgroup, error=empty_subgroup.
- UVSPEC15 `functional_group=aromatic`: n=0, correct=empty_subgroup, error=empty_subgroup.
- Spectra use the prepared axes (IR/Raman 500–4000; UV-Vis 1–15); the prepared metadata records axis source but not units, so panels label the axis value rather than asserting units.
- Structure images are RDKit renders of the structure-library canonical SMILES; no Pillow schematic fallback is used.
- Overlapping subgroups may select the same molecule; no molecule is used as both correct and error case within one subgroup.
