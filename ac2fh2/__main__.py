"""Command line:  python -m ac2fh2 <AC car folder> <FH2 media\\cars folder or zips...> --db gamedbRC.slt --out <mods folder>"""
import argparse
import glob
import os

from .convert import Settings, convert
from .validate import validate


def main():
    ap = argparse.ArgumentParser(prog='ac2fh2', description='Convert an Assetto Corsa car into a Forza Horizon 2 add-on car')
    ap.add_argument('ac_folder')
    ap.add_argument('fh2_cars', nargs='+', help='FH2 car zips, or a folder containing them (your game\'s media\\cars)')
    ap.add_argument('--db', required=True, help='your game\'s media\\stripped\\gamedbRC.slt')
    ap.add_argument('--out', default='.', help='output folder (e.g. Azimuth\\mods\\Forza Horizon 2)')
    ap.add_argument('--id', type=int, default=3100, help='new car id (3000-8999)')
    ap.add_argument('--name'); ap.add_argument('--short'); ap.add_argument('--year', type=int)
    ap.add_argument('--skin'); ap.add_argument('--model-donor'); ap.add_argument('--physics-donor', type=int)
    ap.add_argument('--max-texture', type=int, default=2048)
    ap.add_argument('--ingame', help='what My Cars shows once you own the car, to correct the Autoshow stats/price: '
                                     'PI,speed,handling,acceleration,launch,braking  e.g. 779,6.2,7.2,8.1,7.2,8.7')
    ap.add_argument('--speedo-scale', type=float, default=1.0,
                    help='multiply the speedometer needle sweep (some AC mods do not match their own dial face)')
    a = ap.parse_args()
    measured = None
    if a.ingame:
        v = [float(x) for x in a.ingame.split(',')]
        measured = dict(zip(['PI', 'Speed', 'Handling', 'Acceleration', 'Launch', 'Braking'], v))
    zips = []
    for p in a.fh2_cars:
        zips += sorted(glob.glob(os.path.join(p, '*.zip'))) if os.path.isdir(p) else [p]
    root = convert(Settings(a.ac_folder, zips, a.db, a.out, a.id, a.name, a.short, a.year, a.skin,
                            a.model_donor, a.physics_donor, a.max_texture, measured_stats=measured,
                            gauge_speed_scale=a.speedo_scale))
    problems, _ = validate(root, a.db)
    raise SystemExit(1 if problems else 0)


if __name__ == '__main__':
    main()
