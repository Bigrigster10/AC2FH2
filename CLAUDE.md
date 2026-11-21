# AC2FH2 — project memory (read this first)

Converter: Assetto Corsa car folder -> NEW add-on car for Forza Horizon 2 (Xbox One version run on PC via
Azimuth + Extra Options). Owner tests in-game and reports with screenshots + `FH2_ExtraOptions.log`
(next to azimuth.exe). Repo is PUBLIC: never commit game files or AC car files.

## Status (v0.6 sent for testing)
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
- Next: confirm rims/patterns/tile in game; rim offset (AC RIM_OFFSET: rim sticks out ~8 cm, radius 0.166 vs 0.14);
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
  Wheel models: unit rim, x 0 (outer face) .. 1 (inner), radius 0.14; carbin T = outer face, right side mirrored.
- Skeleton: bone list identical in skeleton.modelbin and each model's Skel. Steering: boneSteeringWheelSpindle (X axis
  = column). Locators.xml carLocator_steeringWheel. Driver animations (scene/Drivers/*.gr2) are baked per car -> pick
  donor with same steering side.
- swatchbin (swatchbin.py, xbtile.py): TXCH 35 B = GUID + w,h,1,w,h,1 + mips,mips,0 + format (BC7 sRGB 2e 01 12 b0,
  linear 2e 01 00 b0). Xbox One tiling: top level ALWAYS 2D thin (padded to 64x32-block macro tiles), lower mips
  2D while >= 64x32 blocks else 1D thin (8x8 micro tiles, >= 1 KiB). Mipped textures kept >= 256x128 (smaller unverified).
- AC: kn5 v5 (kn5.py); data.acd key from folder name (acd.py). AC +X = left; FH2 -X = left -> mirror X
  (accar.py: negate x of pos/nrm/tan, tan.w *= -1, reverse winding; nodes S@M@S).

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
