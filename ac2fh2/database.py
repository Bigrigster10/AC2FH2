"""Generate an Extra Options database.json (format 2) for a new car.

Rows are copied from two stock cars in the user's own gamedbRC.slt and re-keyed to the new car id:
  * the *model donor* (whose car zip we build on): body, bumpers, wing, cameras, part positions,
    because the carbin's upgrade ids and the skeleton must agree with them;
  * the *physics donor* (the closest real car): Data_Car, engine / drivetrain / suspension / brakes /
    tyres / rims / weight, colours and the Autoshow group.
Ids follow the game's convention: car N owns N and N*1000 .. N*1000+999.
"""
import sqlite3
from dataclasses import dataclass, field

ADDON_STRING_TABLE = 'AddOnTable'
ADDON_STRING_HASH = 5679          # as used by working add-on cars (RX-7 port): '_&' + (hash << 16 | key)

# table -> (source, owner column, owner kind)  kind: 'car' = value N, 'body' = value N*1000
MODEL_TABLES = {
    'Data_CarBody': ('Id', 'body'),
    'List_UpgradeCarBody': ('Ordinal', 'car'),
    'List_UpgradeCarBodyFrontBumper': ('CarBodyID', 'body'),
    'List_UpgradeCarBodyRearBumper': ('CarBodyID', 'body'),
    'List_UpgradeCarBodySideSkirt': ('CarBodyID', 'body'),
    'List_UpgradeCarBodyHood': ('CarBodyID', 'body'),
    'List_UpgradeRearWing': ('Ordinal', 'car'),
    'CarPartPositions': ('Ordinal', 'car'),
    'CameraOverrides': ('CarId', 'car'),
    'CarExceptions': ('CarID', 'car'),
}
PHYSICS_TABLES = {
    'Data_Car_Buckets': ('CarId', 'car'),
    'Combo_Colors': ('Ordinal', 'car'),
    'List_AntiSwayPhysics': ('Ordinal', 'car'),
    'List_SpringDamperPhysics': ('Ordinal', 'car'),
    'List_UpgradeAntiSwayFront': ('Ordinal', 'car'),
    'List_UpgradeAntiSwayRear': ('Ordinal', 'car'),
    'List_UpgradeBrakes': ('Ordinal', 'car'),
    'List_UpgradeDrivetrain': ('Ordinal', 'car'),
    'List_UpgradeEngine': ('Ordinal', 'car'),
    'List_UpgradeRimSizeFront': ('Ordinal', 'car'),
    'List_UpgradeRimSizeRear': ('Ordinal', 'car'),
    'List_UpgradeSpringDamper': ('Ordinal', 'car'),
    'List_UpgradeTireCompound': ('Ordinal', 'car'),
    # body-keyed but physical: re-parented onto the new body
    'List_UpgradeCarBodyChassisStiffness': ('CarbodyId', 'body'),
    'List_UpgradeCarBodyTireWidthFront': ('CarBodyId', 'body'),
    'List_UpgradeCarBodyTireWidthRear': ('CarBodyId', 'body'),
    'List_UpgradeCarBodyWeight': ('CarBodyId', 'body'),
}
# columns that hold ids owned by a car (remapped when they fall in the source car's range)
ID_COLUMNS = {'id', 'ordinal', 'carid', 'carbodyid', 'antiswayphysicsid', 'springdamperphysicsid',
              'frontspringdamperphysicsid', 'rearspringdamperphysicsid', 'modelid'}
# Data_Car columns that describe the model (taken from the model donor, not the physics donor)
MODEL_CAR_COLUMNS = ('StockWheelID', 'CaliperAngleFront', 'CaliperAngleRear', 'IsOpenCockpit', 'IsOpenWheel',
                     'HasLivery', 'WingMask', 'GaugeID')


@dataclass
class NewCar:
    car_id: int
    media_name: str
    name: str                    # Autoshow / top bar name, e.g. "Impreza WRX STI"
    short: str                   # e.g. "Subaru Impreza '04"
    year: int
    model_donor: int             # car id whose zip we build on
    physics_donor: int           # car id whose physics rows we copy
    thumbnail_rc: str = 'RC0'
    overrides: dict = field(default_factory=dict)   # Data_Car column -> value


def _cols(db, t):
    return [(r[1], r[5]) for r in db.execute(f'pragma table_info("{t}")')]


def _remap(v, src, dst, col):
    if not isinstance(v, int) or col.lower() not in ID_COLUMNS:
        return v
    if v == src:
        return dst
    if src * 1000 <= v <= src * 1000 + 999:
        return dst * 1000 + (v - src * 1000)
    return v


def _rows(db, table, col, kind, src):
    if kind == 'car':
        q, a = f'select * from "{table}" where "{col}"=?', (src,)
    else:
        q, a = f'select * from "{table}" where "{col}" between ? and ?', (src * 1000, src * 1000 + 999)
    names = [c for c, _ in _cols(db, table)]
    return [dict(zip(names, r)) for r in db.execute(q, a)]


def string_ref(key):
    return f'_&{(ADDON_STRING_HASH << 16) | key}'


def build(db_path, car: NewCar):
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    if db.execute('select 1 from Data_Car where Id=?', (car.car_id,)).fetchone():
        raise ValueError(f'car id {car.car_id} is already used by the game')
    if db.execute('select 1 from Data_Car where MediaName=? collate nocase', (car.media_name,)).fetchone():
        raise ValueError(f'media name {car.media_name} is already used by the game')
    tables = {}

    def add(table, rows, src):
        if not rows:
            return
        pk = [c for c, p in sorted(_cols(db, table), key=lambda x: x[1]) if p]
        out = [{k: _remap(v, src, car.car_id, k) for k, v in r.items()} for r in rows]
        tables.setdefault(table, {'key': pk, 'insert': []})['insert'] += out

    # Data_Car: physics donor row, model columns from the model donor, identity from the new car
    phys = _rows(db, 'Data_Car', 'Id', 'car', car.physics_donor)[0]
    model = _rows(db, 'Data_Car', 'Id', 'car', car.model_donor)[0]
    row = dict(phys)
    for c in MODEL_CAR_COLUMNS:
        if c in model: row[c] = model[c]
    k1, k2 = (car.car_id * 10) % 65000, (car.car_id * 10 + 1) % 65000
    row.update(Id=car.car_id, ModelId=car.car_id, MediaName=car.media_name, Year=car.year,
               DisplayName=string_ref(k1), ModelShort=string_ref(k2), ContentId=0,
               Thumbnail=f'game:\\Media\\Stripped\\{car.thumbnail_rc}\\Thumbnail_{car.car_id}.png')
    row.update(car.overrides)
    tables['Data_Car'] = {'key': ['Id'], 'insert': [row]}

    for t, (col, kind) in MODEL_TABLES.items():
        add(t, _rows(db, t, col, kind, car.model_donor), car.model_donor)
    for t, (col, kind) in PHYSICS_TABLES.items():
        rows = _rows(db, t, col, kind, car.physics_donor)
        if kind == 'body':   # physical body rows: re-parent onto the model donor's body numbering first
            for r in rows:
                for c in list(r):
                    if c.lower() == 'carbodyid' and r[c] == car.physics_donor * 1000:
                        r[c] = car.model_donor * 1000
            add(t, [{k: (_remap(v, car.physics_donor, car.model_donor, k) if k.lower() == 'id' else v)
                     for k, v in r.items()} for r in rows], car.model_donor)
        else:
            add(t, rows, car.physics_donor)
    # Combo_Colors ids are not car-owned in every car; make sure they are free
    used = {r[0] for r in db.execute('select ID from Combo_Colors')}
    for r in tables.get('Combo_Colors', {}).get('insert', []):
        while r['ID'] in used:
            r['ID'] += 1
        used.add(r['ID'])
    db.close()
    doc = {'format_version': 2,
           'target': {'database': 'media/stripped/gamedbRC.slt'},
           'policy': {'id_conflicts': 'error', 'update_conflicts': 'error', 'foreign_keys': 'error'},
           'requires': {'records': {}},
           'tables': tables}
    strings = {'format_version': 1, 'fallback_language': 'EN',
               'languages': {'EN': {ADDON_STRING_TABLE: {str(k1): car.name, str(k2): car.short}}}}
    return doc, strings


def verify(db_path, doc):
    """apply the inserts to an in-memory copy of the game database: catches key clashes and bad columns"""
    src = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    mem = sqlite3.connect(':memory:')
    src.backup(mem); src.close()
    mem.execute('pragma foreign_keys=ON')
    n = 0
    for t, spec in doc['tables'].items():
        for r in spec.get('insert', []):
            cols = ', '.join(f'"{c}"' for c in r)
            mem.execute(f'insert into "{t}" ({cols}) values ({", ".join("?" * len(r))})', list(r.values()))
            n += 1
    bad = mem.execute('pragma foreign_key_check').fetchall()
    mem.close()
    return n, bad
