# AC2FH2 — project memory (read this first)

Converter: Assetto Corsa car folder -> NEW add-on car for Forza Horizon 2 (Xbox One version run on PC via
Azimuth + Extra Options). Owner tests in-game and reports with screenshots + `FH2_ExtraOptions.log`
(next to azimuth.exe). Repo is PUBLIC: never commit game files or AC car files.
Sister project: github.com/Bigrigster10/FH2-AnimatedCamos (animated camos; copied fh2 format modules from here).

## Status (v0.21: lights done)
- v0.1 tested in game: car loads, in Autoshow, buyable, drivable. Problems seen: body rendered pure black,
  small thumbnail scrambled, driver on wrong side (LHD donor, RHD AC car), stats/price copied from donor.
- v0.2: AO texture added (suspected black cause), small-texture tiling fix, AC physics + predicted stats.
- v0.3: 22B RHD donor (driver side), wheels moved sideways to AC track (body no longer stretched in x),
  AC rims converted, bodykit upgrades reduced to stock only. NOT yet confirmed in game.
- v0.3 tested: loads (after selling v0.1 car), driver side CORRECT, engine sound works. Still black (every model we
  write: body, glass, rims, steering wheel; donor brakes render fine), steering wheel flew out when turning, rims
  oversized/sticking out and vanish at speed, car jitters a bit while driving, user finds it too strong/expensive
  (AC car really is a 674 bhp RWD tune: ui_car.json).
- v0.4: VLay flags fixed (were 0; TEXCOORDn=1<<n, TANGENTn=1<<(n+5) -> 111) = prime suspect for black;
  steering meshes stored in bone space (vertices local to boneSteeringWheelSpindle, world = local @ D);
  rim fitted by its own extents (AC rim mesh drawn 21" vs 18" in tyres.ini; RIM_BLUR excluded).
- Open: rims vanish at speed (donor wheel has blurRim/blurLip meshes, lod_flags 2, bucket 28, 'wheel_blur'
  material; ours has none); jitter; our hp 586 vs ui 674 (turbo model).
- v0.4 tested: BLACK FIXED (paint, glass, interior, lights render), steering OK. Left: rims invisible while
  driving (we saw the donor brake disc through them), thumbnail = dark AC photo, paint list 1 colour + blanks.
- v0.5: every mesh valid at all 6 LODs (mesh lod_flags 0x7F, Modl lod_min 5 / flags 127; game uses LOD1+ for wheels
  while driving); ManufacturerColors.bin generated from AC skins -> nearest _library paint (colors.py, format verified
  byte-exact) + matching Combo_Colors; thumbnails rendered on white by thumbnail.py (software renderer).
- v0.5 tested: manufacturer paints WORK (6 colours), upgrades work. Left: rims wrong even parked, special paints
  (camo/wood) stretched, thumbnail almost a black silhouette.
- v0.6: ROOT CAUSE rims = 3 of 4 wheel carbin records kept the donor's material_indices (dict keyed by shared model
  path) -> fixed + validate checks every record's material names vs its model. Special paints use TEXCOORD1
  (stock: even unwrap ~2.2 repeats/m, spread 1.3) -> writer.pattern_uv box projection for paint meshes.
  Thumbnail: studio reflections (fresnel env) + brighter exposure.
- v0.6 tested: SPECIAL PAINTS WORK. Rims still invisible (only donor rotor/caliper + gold sliver behind them);
  tile: user wants it like stock = transparent background, car only.
- v0.7: ROOT CAUSE rims = stock wheel meshes are a unit rim (r 0.14, x 0..1) + MORPH TARGETS (Mrph blob, MBuf,
  morph_targets 2) that the game blends to the real rim size/width; carbin does NOT scale. Our morph-less rim was
  drawn at r 0.14 m and 1 m deep -> behind the 0.143 m rotor. Now built at real size (radius = DB rim radius,
  depth = tyre width); separate _rear model when rear tyres differ. Rim-size upgrades won't resize it (no morphs).
  Thumbnail: RGBA transparent, faint contact shadow, camera front-RIGHT (stock tiles have the nose pointing left).
- v0.7 tested: RIMS VISIBLE, right place, 5-spoke; big tile OK. Left: rims dark (not gold) except lip, look small
  in the thick FH2 tyre, invisible at speed; small tile shadow looks bad.
- v0.8: diffuse textures exported with alpha 255 (AC diffuse alpha = spec mask; wheel_lod2 alpha was 0 everywhere
  -> FH2 drew it black); fit_rim_size: biggest rim <= AC drawn size keeping overall tyre diameter with aspect >= 30
  (Subaru: 235/35R18 -> 235/30R19); small tile without shadow. At-speed invisibility may have been black-on-black.
- v0.9: user spotted rim 'behind the brake': AC rim is deep-dish (hub plate ~11.5 cm in), donor rotor face at 9.4 cm
  -> rotor/hat poked through the rim centre. build_wheel returns a depth-vs-radius profile; brake records of each
  corner are pushed inward until clear (+5 mm) (_brake_clearance); verified 0 overlap at all 8 brake parts.
- v0.9 tested: GOLD RIMS, BRAKES BEHIND SPOKES OK. Left: rim looks small (thick 30% FH2 sidewall), no rim at speed.
- v0.10: at speed the game shows only blur meshes -> AC RIM_BLUR meshes become stock-style 'blurLip_LOD0' (opaque, 17)
  and 'blurLip_LOD0|blurRim_LOD0' (transparent, bucket 28), lod_flags 0x7E (no LODS0 bit); spokes named
  'wheel_LODS0'. MeshIn has lod_flags/bucket overrides. fit_rim_size min aspect 18 (EXPERIMENTAL, stock min 30),
  one size for both axles when AC rims are the same -> 235/24 R20. Stats/price predicted on the AC tyre spec.
- v0.10 tested: rims gold; AC blur disc VISIBLE AT REST (semi-transparent patches over spokes); still NOTHING at
  speed (even blur meshes vanish) -> not a mesh/LOD problem; rims wobble (AC 5.1 deg camber baked in); still small.
- v0.11: wheel meshes get stock-style MORPH data with zero deltas (writer morphs=True: Mrph 'weight[0]','weight[1]';
  MBuf Id 0, stride 32 = 2 targets x (pos,nrm) half4, pos.w 1 rim / 0 blur; morph_targets 2, morph_buf 0;
  ModelStats buffers 4). Stock morphs: spokes t0 collapses barrel x; blur t0 radial x1.318. Blur meshes now the
  DONOR's LOD0 blurLip/blurRim (stock wheel_blur/black MatI, uv from TEXCOORD2) scaled to real size; AC RIM_BLUR
  dropped. Rim straightened by WHEEL node axis (camber removed). Rim size rounds (21"). Wheel TRef filled.
- v0.11 tested: RIMS VISIBLE AT SPEED, no blur at rest, no wobble (all fixed). Left: rim too small / tyre too big
  (owner: don't grow the rim, shrink both a bit), wheel too deep in the car.
- v0.12: wheel outer face = AC rim's straightened outer face (AC WHEEL nodes sit ~6-8 cm inboard of the tyre centre;
  node +- half width was too deep), capped at body arch + 1 cm since camber is removed. fit_rim_size rim_drop=1",
  tyre_scale 0.975 overall -> 235/21 R20 (physics rolling radius 2.5% smaller; stats/price still on the AC spec).
- v0.12 feedback (before testing): owner realised the thin tyre is what looks wrong -> wants a MUCH thicker tyre.
- v0.13: fit_rim_size picks the rim whose profile is closest to target_aspect 37 (overall tyre 97.5% of AC) ->
  235/37 R17 (sidewall ~8.7 cm vs 4.9). Stance fix from v0.12 kept.
- v0.13 tested (4K): tyre thickness MUCH better. Left: rim looks a little small in the arch (owner: make the TYRE
  thicker rather than the rim bigger), wheels stick out the side a little.
- v0.14: target_aspect 44, tyre_scale 1.03 -> 235/44 R17 (sidewall ~10.3 cm, overall +3% vs AC; stats still on AC
  spec); wheel face capped 1 cm INSIDE the body arch extent (was 1 cm outside): front 2 cm in, rear 1.3 cm in.
- v0.13 Autoshow tile report: our thumbnails near-black with ORANGE rims in game, fine as PNG. ROOT CAUSE: TXCH format
  tagged sRGB (2e 01 12 b0) -> game gamma-decodes again. Working RX-7 port thumbnails use 2e 01 00 b8 -> now used.
  Also seen: Autoshow shows S1 850 (our predicted DB stats) but the owned car in My Cars shows A 776 = the game's own
  PI for the actual physics -> our stat prediction runs high; calibrate against in-game values later.
- v0.15 = v0.14 + thumbnail format fix (v0.14 was never installed).
- v0.15 tested: THUMBNAIL COLOURS FIXED (gold rims, normal brightness). Thick tyre / stance look good in My Cars.
  Stock owned car in game: A 779, speed 6.2 handling 7.2 accel 8.1 launch 7.2 braking 8.7 (our prediction S1 850,
  7.3/7.1/8.8/7.8/8.9). Autoshow can't render live (stock cars also use pictures) -> improve thumbnail renderer later
  (smooth normals, studio lighting, AA).
- v0.16: Settings.measured_stats / CLI --ingame PI,speed,handling,accel,launch,braking: in-game values replace the
  prediction; other stat columns + price re-predicted with them as features (Subaru: A 779, 105,000 CR, 0-60 4.5 s).
  PI display <-> value: piecewise linear on class cut-offs (physics.pi_from_display).
- v0.16 tested: owner liked the older (thinner) tyre better, "make the rim bigger instead"; asked for the ROOT CAUSE of
  the endless wheel problems + a bulletproof fix.
- v0.17 = WHEEL SYSTEM REWRITE (see "Wheels" below). Root cause: FH2 sizes rims AND tyres at runtime with morph
  weights from the database; we built fixed real-size rims, so any mismatch with the game's tyre (likely a lip/flange
  offset -> black ring = "rim too small"), every DB size tweak, rim-size upgrades and the at-speed blur texture (drawn
  from UNIT geometry) were all guesswork. Now the AC rim is converted into the stock unit convention with real morph
  targets -> the game sizes it exactly like a stock rim. DB tyre = AC spec (235/35 R18), fit_rim_size removed
  (stock_tyres only clamps to stock ranges). Validator checks the convention (all 5 stock wheels pass it too).
  Pre-existing bug fixed: rear-model selection used the side letter (L/R) instead of the axle letter.
- v0.17 tested: WHEELS FIXED ("perfect"). Left: see-through where the wipers attach (cowl).
- v0.18: classify: materials named glass/window but OPAQUE in AC (blend 0, not ksWindscreen, e.g. 'black_glass' =
  frit borders + wiper cowl) are no longer FH2 glass -> black_plastic (or textured).
- v0.18 tested: cowl/wiper area FIXED.
- v0.19: WORKING LIGHTS (lights.py). FH2 glow is material driven: light materials (lbk_custom_001 reflector etc.)
  carry LightFunctionCarLightParam (0x493073a0) = preset index in the car's LightPresets.bin (kept from the donor),
  glow colour 0xc308c931, strength 0x6f45c5f1, mask texture 0x35c82561. 22B functions: headlight 17, DRL 13, tail 32,
  brake 69, reverse L 1 / R 18, indicators L 21 / R 20. AC lights.ini [BRAKE_n]/[LIGHT_n] NAME meshes -> copy of the
  donor's material for that role (brake / head / amber = head fn + amber colour / tail), mask -> white; reverse lamps
  by name, split per side. Interior LIGHT entries (gauges) skipped. Lights.bin still the donor's (beam projection).
- v0.19 tested: headlamps bright but cast NO beam; amber, tail, brake, reverse dead. ROOT CAUSE: Lights.bin (light SOURCES:
  beams, brake/tail glow points) pins every source to one of the donor's lamp model files by path index; we had dropped
  those carbin records (lamps lived in the body) -> no sources. Format: deadbeef, 1, n paths (len, str, u16 1), count,
  76-byte records (u32 type, u32 fn [low 16 bits], u32 model idx, 2 x (pos xyz0, dir xyz0)), u32 0; same in all 5 donors.
- v0.20: lamp meshes go INTO the donor's Lights.bin model files (records kept: part SecondaryLights/PrimaryLights), by slot
  front_L/R, rear_L/R, centre (narrow brake = third brake light); meshes spanning both sides split. Bulbs file gets the
  lamps, other files in the slot an empty model. Sources moved by slot: x/y bbox centre, z keeps distance to the outer face.
- v0.20 tested: HEADLIGHT BEAMS + tail/brake WORK. Left: only ONE reverse lamp lights (22B right 'reverse' material
  = preset 18, same as the fog lamps -> it's a rear fog, the 22B has one reverse lamp). Brake light "only part of
  braking" = likely FH2 brake-to-reverse at standstill (brake turns into reverse), asked owner to confirm.
  NOTE: LightFunctionCarLightParam values are per-car PRESET indices (BMW head 37, Civic head 21...), not global
  functions; role mapping from material names works for the 22B, other donors need checking.
- v0.21: when a donor's two reverse materials use different presets, both sides use the one fewer other lamps share.
- v0.21: owner happy with the lights, moving on.
- Next: second AC car (generalise; light presets per donor); better
  thumbnails; why the stat prediction runs ~70 PI high;
  lights glow (lbk_* materials + LightFunctionCarLightParam, AC lights.ini, donor Lights.bin positions);
  gauge needles (boneSpeed/boneTach); baked AO; then the desktop UI + .exe (PySide6 + PyInstaller, sklearn bundled).
- Donor search in the final tool: scan ALL zips in the user's FH2 `media\cars` (side, wheelbase).

## Local data (not in repo; re-upload if the session is new)
- /home/user/gamefiles/cars/*.zip — FH2 stock car zips: BMW_M4_14, FIA_AbarthPuntoSS_13, HON_CivicTypeR_97 (RHD),
  KIA_Ceed_11 (RHD, from a mod), SUB_Impreza22b_98 (RHD). /home/user/gamefiles/gamedbRC.slt (sha256 4d50013d…, vanilla).
- AC test car: subaru_impreza_wrx_2004 (RWD 674bhp mod, turbo 1.5 bar, RHD). Output: /home/user/gamefiles/out.
- thirdparty_eo_tools/ (gitignored, no licence): Extra Options carbin_json.py etc. — local verification only.

## Run
python -m ac2fh2 <ac_car_folder> <fh2 cars folder|zips> --db gamedbRC.slt --out <mods\Forza Horizon 2> --id 3100
(validation runs automatically; ac2fh2/validate.py)

## Mod package (Extra Options, see EO docs MODS.md)
mods\Forza Horizon 2\<id>\ mod.json, database.json (format 2: tables{key, insert}), strings.json
(table "AddOnTable", ref "_&" + str(5679<<16 | key)), media\cars\<MEDIA>.zip (no zipmanifest needed),
media\stripped\RC0\Thumbnail_<id>_Big/_Small.swatchbin (816x459 / 192x108 BC7 1 mip).
New car ids 3000-8999; car N owns ids N and N*1000..N*1000+999. ContentId must be 0.

## Formats (all verified byte-for-byte on game files)
- Grub bundle `burG` 1.0 (bundle.py): blob infos 24 B, metadata 8 B entries (tag, len<<4|ver, rel offset).
  Models pad header to 4; textures (TXCB) don't, and no pad after last blob.
- modelbin (modelbin.py): Mesh 1.7, VLay 1.1 (elem offset -1 = append), VerB/IndB/MBuf 1.0, Modl 1.1, Skel 0.0.
  Mesh: material=MatI Id, layout by LIST POSITION, vertex buffers by 'Id ' meta (Id -1 = shared positions).
  Index range starts at ib_draw_offset; element addr = stream offset + (index + base_vertex)*stride.
  UV stored unorm16 with per-mesh transform (offU, scaleU, offV, scaleV) x5 channels. Mesh.bone = rigid bone.
  Body meshes use bone 1 ('root'). LOD: single-LOD lod_flags 3 / Modl lod flags 0x3. bucket 17 opaque, 9 shadow copy.
  FH2 vertex: POSITION f32x3 slot0; NORMAL f16x4, TEXCOORD0-3 unorm16x2, TANGENT0-1 f16x4 slot1 (40 B).
- Materials (material.py): MatI = nested bundle MATI (7-bit-len library path) + MTPR params v2.0.
  Param names CRC32 (DiffuseATexture 0x6dd98cd9, AlphaTexture 0x57d9d49e, GlossTexture 0x7e4a41e1).
  Texture path hash = CRC32(lowercase path). Modl 'TRef' = all texture hashes the model needs.
- carbin v5 (carbin.py): parts (2 body, 4 brakes, 43 wheels) + upgrade groups (9 wing, 34/35 bumpers, 36 hood,
  37 skirts, 41 cage) with upgrade ids = DB rows. material_indices = (name, flag: 1 paint, 2 secondary, 4/8/16 panels,
  128 glass). Every model needs an AO map entry (path + 27-byte data copied from donor).
  Wheel models: carbin T = outer face (no scale), right side mirrored; ONE model for all 4 corners. See "Wheels".
- Skeleton: bone list identical in skeleton.modelbin and each model's Skel. Steering: boneSteeringWheelSpindle (X axis
  = column). Locators.xml carLocator_steeringWheel. Driver animations (scene/Drivers/*.gr2) are baked per car -> pick
  donor with same steering side.
- swatchbin (swatchbin.py, xbtile.py): TXCH 35 B = GUID + w,h,1,w,h,1 + mips,mips,0 + format (BC7 sRGB 2e 01 12 b0,
  linear 2e 01 00 b0). Xbox One tiling: top level ALWAYS 2D thin (padded to 64x32-block macro tiles), lower mips
  2D while >= 64x32 blocks else 1D thin (8x8 micro tiles, >= 1 KiB). Mipped textures kept >= 256x128 (smaller unverified).
- AC: kn5 v5 (kn5.py); data.acd key from folder name (acd.py). AC +X = left; FH2 -X = left -> mirror X
  (accar.py: negate x of pos/nrm/tan, tan.w *= -1, reverse winding; nodes S@M@S).

## Wheels (verified on all 5 stock wheels + the game's own morphing shaders; convert.build_wheel, validate.check_wheel)
- Stock wheel model = UNIT rim: lip radius exactly 0.140, face/spokes at REAL depth (x 0 = outer face .. ~0.09,
  +x inward, not morphed), barrel back ring at x = 1.0. Mrph blob 'weight[0]','weight[1]'. MBuf (fmt 10 half4)
  per vertex 2n rows: n position deltas then n normal deltas; row .w = INDEX into g_MorphWeights.
  Radius target: weight 0, delta 1.318*(0,y,z), normal delta normalize(nx,ny/2.318,nz/2.318)-n.
  Width target: weight 1, back ring dx = -0.9 (others 0), normal delta 0. Blur meshes (blurLip|blurRim, mt1):
  radius target only (weight 0). Low LODs (LOD2-5) are decimated, deltas off by <1.6 cm.
- Morph VS (Shaders/*/*Morphing.pc.vso in every car zip; tools/dxbc.py disassembles): row = 2n*(SV_VertexID +
  g_AnimationInfo.w) + i; pos += g_MorphWeights[row.w] * row.xyz (x tyre-deformation factor for weight index 2-4,
  used by tyres); normals likewise then normalised. Our morph models use base_vertex 0 / stream offsets 0 / absolute
  indices so row = vertex number under any engine offset convention.
- WheelBlurScenario VS does NOT apply morphs: the at-speed blur texture is rendered from the UNIT rim; blur discs map
  it with TEXCOORD2 -> rims MUST be in unit space for a correct blur.
- Weights are computed by the exe from Data_Car Front/RearWheelDiameterIN + tyre width (formula not in the files; the
  tyre shares weight 0/1, so a unit rim always meets the tyre). rim_scale() bounds the scale (lip at nominal radius
  .. nominal + ~2 cm flange) for brake-clearance checks only; clearance is checked at stock size and +1/+2".
- Ours: AC rim straightened (camber), lip -> 0.14 (ku = 0.14 / AC lip radius), depth fitted into the tyre width
  (only compressed), face kept, barrel behind the face (split) interpolates to x 1.0 with dx -0.9*t (no folding).
  Donor blur meshes copied unchanged with their own morph rows. One model for all corners unless the AC rear rim
  design differs (_rim_signature). DB tables: List_Wheels (StockWheelID, BrakeDrawOffset), WheelNormOffsets,
  List_UpgradeRimSize*/CarBodyTireWidth* (ours copied from the physics donor).

## Database rules (database.py, physics.py)
- Rows from 3 stock cars: model donor (zip: body, bumpers, wing, cameras, part positions), identity donor (same brand,
  closest year: make, Autoshow bucket, paint colours, body aero), physics donor (drive/placement/power/mass: suspension,
  brakes, etc.). Then AC physics replace engine (new EngineID + all List_UpgradeEngine* + List_TorqueCurve),
  drivetrain (new id, gears, diff), mass, tyres. Verified by inserting into a copy of gamedb.
- Calibrated: SimPeakPower = 10 x kW; boost power = base x (1 + 0.43 x bar); SimPeakTorque = TorqueScale/100.
  Torque curve normalised (peak 1.0 x TorqueScale), 0..TorqueCurveMaxRPM, last value negative.
- Stats/price predicted with sklearn GradientBoosting trained on the user's gamedb (PI err ~0.018, price ~x1.4).
  Class from PI cut-offs [0.436, 0.5265, 0.619, 0.7485, 0.8555] (D C B A S1 S2).
- Saves: changing upgrade row ids crashes saves with the car -> tell user to sell/back up before updating.
  CONFIRMED: v0.3 crashed at load (game+0x105beec, null row while reading ClutchMaxTorque, R15=3100) because the
  save owned the v0.1 car (v0.1 used SUB_BRZ_13's engine/drivetrain part ids; v0.3 has its own EngineID/DrivetrainID
  3100). Fix path: install v0.1 rebuild (commit 306c38f, cars without 22B), sell car, then v0.3. Keep stock part
  ids stable (N*1000) from now on.
- Audio: FH2 uses FMOD Studio banks (Media\Audio\FmodAnthem\DURANGO\*.bank; EO redirects bank loads) -> AC
  sfx .bank files (also FMOD) may be usable; investigate per-car engine event names.
