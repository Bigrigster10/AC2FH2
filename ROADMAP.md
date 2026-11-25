# AC2FH2 roadmap

Owner's goals for the converter (kept up to date by Claude; tell Claude to change it).
Done items are listed at the bottom.

## Short term: make one car work properly (the Subaru)
1. **Better Autoshow pictures**: new renderer in v0.22, waiting for the in-game check.
2. **Working gauge needles**: speedo + tacho in v0.22, waiting for the in-game check (speedo vs HUD reading).
3. **Accurate first stats estimate**: the prediction runs ~70 PI high (S1 850 predicted vs A 779 in game).
4. **Body shading**: paint should look like FH2 paint, not flat or plasticky (smooth normals, baked AO).
5. **Fuel needle** (and other small gauges) moving.

## Mid term: make cars feel like real FH2 cars, then build the app
1. **Correct parts**: proper upgrade options (rims, tyres, engine parts); paint and vinyls apply properly.
2. **Better donor matching**: scan every car in the FH2 install (body shape, driver side, sound), not just 5.
3. **Test 3 to 5 different AC cars** (old car, supercar, FWD hatchback, AWD car) to catch what the Subaru doesn't show.
   Includes making the light mapping work for any donor (light presets differ per car).
4. **Automatic speedo calibration**: detect when an AC mod's needle settings don't match its own dial (no hand tuning).
5. **The desktop app** (`.exe`):
   - Pick the AC car folder(s) and the FH2 folder.
   - Preview stats, class and price, and edit them.
   - Review how each surface is converted (paint, glass, chrome...).
   - Optional previews: car picture, wheels, dashboard needles.
   - One click to convert; the mod goes straight into the Extra Options mods folder.
   - **Bulk mode with "skip all menus"**: every choice has an automatic default, auto car ids, one mod per car,
     a warnings summary at the end.

## Long term: polish and extras
1. **Lower-detail models (LODs)** so many converted cars on screen run smoothly.
2. **Damage and dirt** on converted cars.
3. **Update and remove** converted cars safely without breaking saves (ids stay stable; warn to sell first).
4. **Custom AC engine sounds** (stretch goal: depends on cracking FH2's FMOD audio setup).
5. **Share-ready mods** other people can install.
6. **More AC cars**, including ones built for CSP (Custom Shaders Patch) where possible. Never encrypted cars.

## Done
- Car loads, drives, buyable, right driver side, paints and special paints, upgrades menu works (v0.1–v0.6).
- Wheels: stock-style morphing rims, correct size/stance, visible at speed, brakes behind spokes (v0.17).
- Wiper / cowl area no longer see-through (v0.18).
- Working lights: headlight beams, tail, brake, both reverse lamps (v0.21).
- Thumbnail colours, Autoshow stats can be set from the in-game values (v0.15–v0.16).
