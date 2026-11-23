# Handoff: how to pick up this project

CLAUDE.md = technical memory + version-by-version status (what was tried, what the owner confirmed in game).
NOTES.md = early format research. This file = how the work is done day to day.

## Where things stand (latest: v0.13, waiting for in-game test)
- Working in game: loads/buyable/drivable, correct driver side, engine sound, paint/glass/interior render,
  steering wheel, 6 factory paints, special paints, upgrades, gold rims visible at rest AND at speed, no wobble,
  stock blur fades in at speed only, brakes sit behind the spokes, Autoshow big tile OK, small tile has no shadow.
- v0.12 pushed the wheels out to the AC rim face (they sat 6-8 cm too deep). v0.13 gives a thick tyre
  (235/37 R17, chosen by `fit_rim_size(target_aspect=37, tyre_scale=0.975)` in ac2fh2/convert.py) because the
  owner said the thin tyre looked "paper thin". Ask the owner for the v0.13 result first.
- Backlog, roughly in order: tune tyre/rim/stance from feedback; real rim morph targets (rim-size upgrades);
  glowing lights; gauge needles; the slight driving jitter; engine power 586 vs 674 hp (turbo model);
  AC FMOD engine sounds; scan all of the owner's FH2 car zips for donors; desktop UI (PySide6) packaged as a
  Windows .exe (PyInstaller, bundle sklearn).

## Working loop with the owner
1. Change code, rebuild the Subaru test car (command below), run validation.
2. Zip the output folder as out/AC2FH2_test_build_vX.YY_Subaru.zip (out/ is gitignored) and give it to the owner.
3. Owner installs it: SELL THE CAR IN GAME FIRST (changing part/upgrade ids crashes saves that own the car),
   delete the old folder in `mods\Forza Horizon 2`, unzip the new one, buy the car again.
4. Owner reports with screenshots and `FH2_ExtraOptions.log` / azimuth.log (next to azimuth.exe).
5. Record the result in CLAUDE.md's status list ("vX tested: ..."), commit, push.
Give the owner a short checklist of what to look at with every build. Keep explanations plain (they are not a
programmer). Change one or two things per build so results are attributable.

## Rules
- The repo is PUBLIC: never commit game files (FH2 zips, gamedb, swatchbins from the game) or AC car files,
  and never the owner's email or other personal details.
- Never decrypt protected/encrypted AC cars.
- Keep stock part ids stable (car N owns N and N*1000..N*1000+999; ContentId 0) so saves keep loading.
- No model names/identifiers in commit messages.

## Setup on a new machine
```
git clone https://github.com/Bigrigster10/AC2FH2
cd AC2FH2
pip install numpy pillow scipy scikit-learn etcpak texture2ddecoder
```
Local data the agent needs (not in git; get it from the owner):
- FH2 stock car zips (from the game's `media\cars`), at least SUB_Impreza22b_98.zip (the RHD donor used now),
  plus BMW_M4_14, FIA_AbarthPuntoSS_13, HON_CivicTypeR_97, KIA_Ceed_11.
- gamedbRC.slt from `media\stripped` (vanilla, sha256 starts 4d50013d).
- The AC test car folder subaru_impreza_wrx_2004 (from the owner's AC `content\cars`).
- Optional: thirdparty_eo_tools/ (Extra Options' carbin_json.py etc.; no licence, keep local only).

## Build + validate the test car
```
python -m ac2fh2 <AC>/subaru_impreza_wrx_2004 <folder with FH2 car zips> --db gamedbRC.slt --out <out dir> --id 3100
```
(Claude's runs used name 'Impreza WRX STI', short name "Subaru Impreza '04"; see ac2fh2/convert.py `Settings`.)
Validation runs automatically (ac2fh2/validate.py) and must print only `ok:` lines. The build log lines worth
checking each time: rim size, "wheels moved sideways", "brakes moved inward", class/PI/price.

## Code map (ac2fh2/)
convert.py (pipeline: donors, body fit, wheels/rims, brakes, carbin, database, thumbnails), writer.py (modelbin
writer: MeshIn, LODs, morph data, special-paint UVs), modelbin.py / bundle.py / carbin.py / material.py /
swatchbin.py / xbtile.py (game formats), kn5.py / acd.py / accar.py (AC input), physics.py (AC physics -> DB rows,
stat/price prediction), database.py, colors.py (factory paints), thumbnail.py (Autoshow tiles), validate.py.
