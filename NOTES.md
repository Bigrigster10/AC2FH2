# AC → FH2 (Xbox One, via Azimuth) converter — research notes

## Azimuth mod package layout (from ZS Williams FW24 pack)
```
mod.json            id/name/version/author, points at database.json + media/
strings.json        localized strings; keys are numeric hashes referenced as "_&<n>" in DB rows
database.json       JSON patch applied to media/stripped/gamedbRC.slt (SQLite)
                    format_version 2, target.base_sha256 pins the exact game DB
                    tables -> {key:[...], insert:[rows...]}
media/cars/<MEDIANAME>.zip            the car archive itself
media/stripped/RC0/*.swatchbin        manufacturer logos + thumbnails (Thumbnail_<carId>_Big/Small)
media/audio/fmodanthem/durango/*.bank engine audio (FMOD Studio bank)
```

### DB tables a new car touches (~45)
Data_Car (core row, MediaName = zip name), Data_CarBody, Data_Engine, Data_Drivetrain,
List_TorqueCurve (normalized v0..vN * TorqueScale), List_Upgrade* (stock part per slot,
IsStock=1), List_SpringDamperPhysics (front=xxx003, rear=xxx103), List_AntiSwayPhysics,
List_AeroPhysics, Combo_Colors, CameraOverrides, List_Wheels, List_CarMake, ...
ID conventions: CarId N, part ids N*1000+level, front/rear physics N*1000+3 / +103.

## Car zip
The FW24 is a *donor* build: the game's Lotus E21 archive (BuildNumber 2013, LOT_* files)
with new meshes/textures swapped in. Our converter should do the same:
copy a donor car zip, replace modelbins + textures, rename.

Contents: scene/**.modelbin (per part, LODs inside), Textures/**.swatchbin, Animations/*.gr2
(Granny), Locators.xml, LiveryMasks/, physicsdefinition.bin, Lights.bin, *.carbin, UI/*.xaml

## Grub container (`burG`, little endian) — modelbin & swatchbin
header: magic 'burG', u8 vmaj, u8 vmin, u16 count, u32 hdrSize, u32 dataSize
entry (24B): tag[4] (reversed ascii), u8 maj, u8 min, u16 metaCount, u32 metaOff,
             u32 dataOff, u32 size, u32 size2
modelbin tags: Skel, Mrph, MatI, Mesh (one per LOD/material), IndB, VLay, VerB, MBuf, Modl
  VLay semantics seen: POSITION NORMAL TEXCOORD TANGENT
  MatI = nested Grub pointing to library materials, e.g.
       Game:\Media\cars\_library\materials\basematerials\carpaint_livery.materialbin
       + texture overrides (paths to swatchbins, library or car-local)
swatchbin: one TXCB blob; meta[24..] = u16 w, u16 h, u16 depth, (repeat), u8 mips, u8 mips,
           u8 0, format bytes:
   2e 01 12 b0  ~1 B/px  BC7 (diffuse, sRGB?)
   6e 00 00 a0  0.5 B/px BC4 (opacity / masks)
   6e 02 00 90  1 B/px   R8 (AO)
   0e 00 00 a0  0.5 B/px BC1?
   ae 01 xx 90  4 B/px   RGBA8
**Pixel data is Xbox One (GCN) tiled** — linear decode gives correct colours but scrambled
blocks. Need GCN 2D-thin tiling (AMD addrlib / Mesa) to write textures.

## Assetto Corsa side — done
kn5.py  : reads sc6969 v5: textures (DDS), materials (shader+props+tex slots), node tree, meshes
acd.py  : decrypts data.acd (key from folder name) → ini/lut physics files
Subaru: 168 meshes, 221k tris, 63 mats, 51 textures. Node names like sub_wrxsti_04_*.

## Next steps
1. Fully decode Mesh/VerB/IndB/VLay (vertex formats, LOD table, per-mesh material index)
2. Implement GCN texture tiling (decode first to verify on FW24 textures, then encode)
3. Writer: kn5 meshes → modelbin; DDS → swatchbin
4. Physics mapper: AC ini/lut → database.json rows
5. Packager: mod.json + strings.json + database.json + media/

## Update: Kia cee'd mod + docs
- Kia = FM5 car zip dropped into FH2 → FM5-format car zips work. Car 1364 already existed in the
  DB as a traffic car; the mod just deletes its TrafficCars row (+ wheel/upgrade tweaks).
  Full road-car part set (doors, lights, windows, interior) → best donor template for the Subaru.
- fh2-mod-platform (Trinculo54) docs/car-packs.md + game-notes.md: car-pack format
  (pack.ini + .sql with {car:X} ids 3000-8999, part-ids.txt for carbin wing ids),
  zip rules (upper-case name order, no local extra field, 0x1123 central extra = data offset,
  zipmanifest.xml line), rows the game trusts (PowertrainID must exist in Powertrains 0-7).
- TEXTURE TILING SOLVED for BC7 mip0 (xbtile.py, formula from game-notes.md#thumbnails).
  Verified on FW24 rim / steering textures. Mips + 8-byte (BC1/BC4) + R8 layouts still TODO.
- fh2mods.com and azimuthforza.com did not resolve from this cloud session.

## Update 2: tools & references received
- AzimuthModStudio 0.2.3 (Python src): donor-clone workflow. swatch.py reads textures via xg.dll
  (Windows) - DXGI map: 98 BC7, 71 BC1, 77 BC3, 80 BC4, 83 BC5, 95 BC6H, 28 RGBA8, 61 R8.
  model.py documents Mesh field offsets (mat u16@0, istart@0x19, base@0x1D, icount@0x21,
  vcount@0x2D, layout id/nstreams@0x31, streams 16B each @0x39), VLay/VerB/IndB headers,
  bundle metadata (8B entries: tag, u16 len<<4, u16 rel offset).
  pack.py: DB rows cloned in SQL from a donor car (INSERT..SELECT, ids car*1000+n).
- RX-7 (FH6 port, id 3700): full standalone database.json (29 tables) = template for new cars;
  strings via custom "AddOnTable" (_& ref = table_hash<<16 | key).
- Extra Options MODS.md = full database.json spec (format 2: insert/update/delete, policies,
  id_conflicts nearest). Loose zips in media/ need no zipmanifest. strings.json spec.
- carbin_json.py (extracted from EO mod_diff.py → thirdparty_eo_tools/): lossless FH2 carbin
  <-> JSON. Verified on Kia + RX-7. NOTE: EO repo has no licence file — personal use only.
- ForzaTechStudio (Nenkai et al, MIT, C#/WinUI, Windows-only): Modelbin Creator (OBJ/FBX →
  modelbin, FH2 target supported), conversion tool across games, swatchbin PC<->Xbox
  tiling (2D_THIN/1D_THIN), carbin editor. FH2 vertex: NORMAL/TANGENT R16G16B16A16_FLOAT,
  TEXCOORD R16G16_UNORM. Key files: Services/ModelBuilderService.cs, BundleBuilderService.cs,
  MaterialInstanceBuilderService.cs, ModelbinConversionService.cs, DurangoTypes/.
- Doliman100 ForzaTech-extraction-tools (GPLv3): modelbin.hexpat / carbin.hexpat specs.
