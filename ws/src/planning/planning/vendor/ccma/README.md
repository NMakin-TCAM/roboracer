# ccma (vendored)

`ccma.py` is the Curvature Corrected Moving Average of T. Steinecker and
H.-J. Wuensche (IEEE IV 2023), copied **byte for byte** from ForzaETH's fork:

- Repository: https://github.com/ForzaETH/ccma (a fork of UniBwTAS/ccma)
- Commit: `6b8a80e20e0dfda40b9fe60c7b839e086291b3d5`, the commit ForzaETH's
  race_stack pins as its `f110_utils/libs/ccma` submodule (race_stack
  `e7d07f7f4ffe0d4c30b90340b0f49a76af2877c0`)
- File: `src/ccma/ccma.py`, sha256
  `ea60ae29ae95b756adc0548f2d06e30e140accd69db26bb937e40ad3397e8b03`
- License: BSD 3-Clause, copyright UniBwTAS, kept in `LICENSE` beside it

`planning.dense_gp_opp_traj` (the port of Predictive Spliner's
`gaussian_process_opp_traj.py`) is its only user. Upstream imports it as
`from ccma import CCMA` after `pip install race_stack/f110_utils/libs/ccma`;
here it is `from planning.vendor.ccma import CCMA`.

## Why vendored rather than installed in the image

- It is not on PyPI. Installing it in the image means cloning a fork at a
  pinned commit during `docker compose build`, and every developer rebuilding.
- It is one pure-Python module that needs only numpy and `scipy.stats`, both
  already ROS dependencies of `planning`. Nothing has to be compiled.
- Its packaging metadata is a template ("A small example package", MIT
  classifier while the license file is BSD), and its `requirements.txt` asks
  for `scipy>=1.9.3` and matplotlib, neither of which this module needs. The
  image's ROS scipy is 1.8.0. Vendoring avoids anything resolving those.
- A test (`test_vendored_ccma_is_the_pinned_upstream_file`) fails if the file
  stops matching the sha256 above, so it cannot drift silently. To update it,
  copy the new upstream file, update the commit and hash here and in the test,
  and rerun the differential test.

Do not edit `ccma.py`. NumPy 2 deprecates its `np.row_stack`; the image has
NumPy 1.21, where it is fine. If that ever needs changing, change it here with
a note, not by patching upstream behaviour in place.
