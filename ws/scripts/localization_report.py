#!/usr/bin/env python3
"""HTML report of localization benchmark results (scripts/localization_benchmark.py), one page, no dependencies.

    python3 scripts/localization_report.py log/localization_benchmark/<a>/ log/localization_benchmark/<b>/ ...
        [-o log/localization_benchmark/report.html]

Groups the runs by speed_scale (from each folder's meta.json) and localizer, and draws one small chart per
metric (x = speed scale, one line per localizer through the mean of its runs, a dot per run), plus a table of
every run. Open the file in a browser.
"""
import argparse
import html
import json
from pathlib import Path

METRICS = [   # key, title, unit, scale, what it means
    ('err_mean', 'Mean position error', 'cm', 100, 'average distance between the estimate and the true pose'),
    ('err_max', 'Worst position error', 'cm', 100, 'the single largest error in the run'),
    ('head_p95', 'Heading error, 95th percentile', 'deg', 1, '95 % of the time the heading was off by less'),
    ('aligned_mean', 'Error with map placement removed', 'cm', 100,
     'mean error after the best rigid fit: the localizer alone, not the map'),
    ('jumps', 'Pose jumps per run', '', 1, 'steps of the estimate faster than 15 m/s (snaps)'),
    ('collisions', 'Crashes per run', '', 1, 'the sim\'s collision count over the 60 s'),
]
TABLE = ['err_mean', 'err_median', 'err_p95', 'err_max', 'far_share', 'head_mean', 'head_p95', 'aligned_mean',
         'jumps', 'max_jump', 'gap_s', 'stalls', 'laps', 'collisions', 'scan_hz']


def load(dirs):
    runs = []
    for d in dirs:
        s = json.loads((Path(d) / 'summary.json').read_text())
        meta = s['meta']
        for name, r in s['runs'].items():
            runs.append({'name': name, 'localizer': name.rsplit('_', 1)[0], 'speed': meta['speed_scale'],
                         'track': meta['track'], 'duration': meta['duration'], 'folder': Path(d).name, **r})
    return runs


PAGE = (Path(__file__).parent / 'localization_report.html').read_text()


def plural(n, one, many=None):
    return f'{n} {one if n == 1 else many or one + "s"}'


def findings(runs):
    """Headline tiles: (label, value, note)."""
    out = []

    def pick(loc, key, speeds=None):
        v = [r[key] for r in runs if r['localizer'] == loc and key in r and (speeds is None or r['speed'] in speeds)]
        return v
    for loc, label in (('cartographer', 'Cartographer'), ('mcl', 'MCL (ours)')):
        e = pick(loc, 'err_mean')
        if e:
            out.append((f'{label}: mean error, all runs', f'{100 * sum(e) / len(e):.1f} cm',
                        f'worst single error {100 * max(pick(loc, "err_max")):.0f} cm; '
                        f'{plural(sum(pick(loc, "jumps")), "jump")}, '
                        f'{plural(sum(pick(loc, "collisions")), "crash", "crashes")} in {len(e)} runs'))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('-o', '--output', default='log/localization_benchmark/report.html')
    args = ap.parse_args(argv)
    runs = load(args.dirs)
    tracks = sorted({r['track'] for r in runs})
    speeds = sorted({r['speed'] for r in runs})
    reps = max(sum(1 for r in runs if r['localizer'] == loc and r['speed'] == s)
               for loc in {r['localizer'] for r in runs} for s in speeds)
    sub = (f'Track {", ".join(tracks)} in the AutoDRIVE simulator, pure pursuit on the raceline, '
           f'{runs[0]["duration"]:.0f} s per run, up to {reps} runs per localizer at speed scale '
           f'{", ".join(str(s) for s in speeds)} (fraction of the planned raceline speed). '
           'Ground truth from the simulator is used only for scoring.')
    page = (PAGE.replace('__SUB__', html.escape(sub)).replace('__RUNS__', json.dumps(runs))
            .replace('__METRICS__', json.dumps(METRICS)).replace('__TABLE__', json.dumps(TABLE))
            .replace('__FINDINGS__', json.dumps(findings(runs))))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(page)
    print(f'Wrote {args.output} ({len(runs)} runs)')


if __name__ == '__main__':
    main()
