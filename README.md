# AC2FH2

Convert Assetto Corsa cars into **new** Forza Horizon 2 cars (Xbox One version running through
[Azimuth](https://almost-there-games.itch.io/azimuth) with Extra Options). Converted cars appear in the
Autoshow next to the stock cars; nothing in the game is replaced.

> Work in progress. No game files are included: the tool builds on files from **your own** FH2 install.

## What it does
1. Reads the Assetto Corsa car (`.kn5` model, textures, skins, `data.acd` / `data` physics).
2. Picks the closest stock FH2 car as a *template* for the parts AC doesn't have (animations, skeleton,
   cameras, collision), and the closest real car for baseline physics.
3. Writes FH2 models (`.modelbin`) and Xbox One-tiled BC7 textures (`.swatchbin`), maps AC materials to
   FH2 library materials (paint stays paintable in game).
4. Outputs an Extra Options mod folder: `mod.json`, `database.json`, `strings.json`, `media/cars/<car>.zip`,
   thumbnails. Everything is re-read and checked against your game database before you play.

## Usage (command line, for now)
```
pip install numpy pillow scipy scikit-learn etcpak texture2ddecoder
python -m ac2fh2 "path\to\assettocorsa\content\cars\some_car" "path\to\FH2\Mount\media\cars" --db "path\to\FH2\Mount\media\stripped\gamedbRC.slt" --out "path\to\Azimuth\mods\Forza Horizon 2"
```
Encrypted (protected) AC cars are not supported.

## Credits / references
Format knowledge from fh2-mod-platform (Trinculo54), ForzaHorizon2 Extra Options (RokkuDayo),
ForzaTech Studio (Nenkai et al., MIT), ForzaTech-extraction-tools (Doliman100), AzimuthModStudio.
<img width="1080" height="1440" alt="temp_image_068099A7-2E53-480D-8982-E4E5F6742380" src="https://github.com/user-attachments/assets/b9c49281-ae49-47d3-b0a2-f6f560c9c66c" />
