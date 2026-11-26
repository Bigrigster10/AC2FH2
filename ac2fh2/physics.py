"""Assetto Corsa physics -> FH2 database rows (engine, torque curves, gearbox, drivetrain, weight, tyres),
and stock stats (PI, class, ratings, times, price) predicted the way the game's own cars are rated.

Calibrated on the stock game database:
  * SimPeakPower = 10 x peak kW;  SimPeakTorque = TorqueScale / 100
  * forced induction: SimPeakPower ~ curve peak x (1 + 0.43 x StockBoost[bar]) (stat only; in game the curve is
    multiplied by the stock aspiration part's scales, which we make neutral because AC curves include boost)
"""
import sqlite3
from dataclasses import dataclass

import numpy as np

BOOST_K = 0.43          # DB stat relation only (SimPeakPower vs curve on stock cars); NOT how the game applies boost
ASPIRATION_SCALES = {'List_UpgradeEngineDSC': ('ZeroRPMScale', 'RedlineRPMScale', 'RobScale'),
                     'List_UpgradeEngineCSC': ('ZeroRPMScale', 'RedlineRPMScale', 'RobScale'),
                     'List_UpgradeEngineTurboSingle': ('MaxScale', 'MinScale', 'RobScale'),
                     'List_UpgradeEngineTurboTwin': ('MaxScale', 'MinScale', 'RobScale'),
                     'List_UpgradeEngineTurboQuad': ('MaxScale', 'MinScale', 'RobScale')}


def ac_ini(text):
    out, sec = {}, None
    for line in (text or '').splitlines():
        line = line.split(';')[0].strip()
        if line.startswith('['): sec = line.strip('[]'); out[sec] = {}
        elif '=' in line and sec: k, v = line.split('=', 1); out[sec][k.strip()] = v.strip()
    return out


def lut(text):
    pts = []
    for line in (text or '').splitlines():
        line = line.split(';')[0].strip()
        if '|' in line:
            a, b = line.split('|')[:2]
            try: pts.append((float(a), float(b)))
            except ValueError: pass
    return np.array(sorted(pts)) if pts else np.zeros((0, 2))


def f(d, sec, key, default=0.0):
    try: return float(d.get(sec, {}).get(key, default))
    except ValueError: return default


@dataclass
class ACPhysics:
    limiter: float
    idle: float
    inertia: float
    rpm: np.ndarray            # effective (boosted) torque curve
    torque: np.ndarray
    boost: float               # bar, 0 = NA
    drive: int                 # 1 FWD, 2 RWD, 3 AWD
    gears: list                # forward ratios
    reverse: float
    final: float
    mass: float                # kg without driver
    front_weight: float        # fraction on the front axle
    tyres: dict                # 'F'/'R': (width mm, aspect %, rim inches)
    diff_power: float
    diff_coast: float

    @property
    def peak_kw(self):
        return float((self.torque * self.rpm * 2 * np.pi / 60 / 1000).max())

    @property
    def peak_torque(self):
        return float(self.torque.max())


def read(car) -> ACPhysics:
    eng, drv = ac_ini(car.data.get('engine.ini')), ac_ini(car.data.get('drivetrain.ini'))
    carini, sus, ty = ac_ini(car.data.get('car.ini')), ac_ini(car.data.get('suspensions.ini')), ac_ini(car.data.get('tyres.ini'))
    pc = eng.get('HEADER', {}).get('POWER_CURVE', 'power.lut')
    base = lut(car.data.get(pc) or car.data.get('power.lut'))
    limiter = f(eng, 'ENGINE_DATA', 'LIMITER', 7000)
    rpm = np.linspace(0, limiter * 1.05, 120)
    tq = np.interp(rpm, base[:, 0], base[:, 1]) if len(base) else np.full_like(rpm, 200)
    boost = 0.0
    for sec in [s for s in eng if s.startswith('TURBO_')]:
        mb = min(f(eng, sec, 'MAX_BOOST'), f(eng, sec, 'WASTEGATE', 99) or 99)
        ref, gam = f(eng, sec, 'REFERENCE_RPM', 4000) or 4000, f(eng, sec, 'GAMMA', 1) or 1
        b = np.minimum(mb, mb * (rpm / ref) ** gam)
        tq = tq * (1 + b); boost += mb
    n = int(f(drv, 'GEARS', 'COUNT', 5))
    gears = [f(drv, 'GEARS', f'GEAR_{i}', 1.0) for i in range(1, n + 1)]

    def tyre(side):
        w, r, rim = (f(ty, side, k) for k in ('WIDTH', 'RADIUS', 'RIM_RADIUS'))
        if not w: return None
        return (round(w * 1000 / 5) * 5, max(20, min(80, round((r - rim) / w * 100 / 5) * 5)), max(13, round(rim * 2 / 0.0254)))
    return ACPhysics(limiter, f(eng, 'ENGINE_DATA', 'MINIMUM', 900) or 900, f(eng, 'ENGINE_DATA', 'INERTIA', 0.1),
                     rpm, tq, boost, {'FWD': 1, 'RWD': 2}.get(drv.get('TRACTION', {}).get('TYPE', 'RWD').upper(), 3),
                     gears, f(drv, 'GEARS', 'GEAR_R', -3.5), f(drv, 'GEARS', 'FINAL', 4.0),
                     f(carini, 'BASIC', 'TOTALMASS', 1300) - 75, f(sus, 'BASIC', 'CG_LOCATION', 0.55),
                     {'F': tyre('FRONT'), 'R': tyre('REAR') or tyre('FRONT')},
                     f(drv, 'DIFFERENTIAL', 'POWER', 0.3), f(drv, 'DIFFERENTIAL', 'COAST', 0.1))


# ------------------------------------------------------------------ rows
def _cols(db, t):
    return [r[1] for r in db.execute(f'pragma table_info("{t}")')]


def _rows(db, t, where, args):
    c = _cols(db, t)
    return [dict(zip(c, r)) for r in db.execute(f'select * from "{t}" where {where}', args)]


def _pk(db, t):
    return [r[1] for r in sorted(db.execute(f'pragma table_info("{t}")'), key=lambda r: r[5]) if r[5]]


def _put(doc, db, table, rows):
    spec = doc['tables'].setdefault(table, {'key': _pk(db, table), 'insert': []})
    spec['insert'] += rows


def _tables(doc, name):
    return doc['tables'].get(name, {}).get('insert', [])


def _curve(db, cid):
    r = _rows(db, 'List_TorqueCurve', 'TorqueCurveID=?', (cid,))[0]
    n = r['NumTorqueValues']
    return r, np.array([r[f'v{i}'] for i in range(n)])


def apply(doc, db_path, car_id, ph: ACPhysics, log=print):
    """replace the physics donor's engine / drivetrain / mass / tyres in `doc` with the AC car's"""
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    dc = _tables(doc, 'Data_Car')[0]

    # ------------------------------------------------ engine
    stock_eng = next(r for r in _tables(doc, 'List_UpgradeEngine') if r['IsStock'])
    src_eng = stock_eng['EngineID']
    eid = car_id
    while db.execute('select 1 from Data_Engine where EngineID=?', (eid,)).fetchone(): eid += 1
    stock_eng['EngineID'] = eid
    de = _rows(db, 'Data_Engine', 'EngineID=?', (src_eng,))[0]
    # the game multiplies the curve by the STOCK forced-induction part (DSC/CSC Zero/RedlineRPMScale, turbo Min/MaxScale,
    # all x RobScale), not by StockBoost: measured in game on v0.22 (Corvette DSC part, x1.29 -> 460 hp instead of 586).
    # The AC curve already includes its boost, so that stock part is made neutral below and upgrades become relative.
    base = ph.torque
    peak = float(base.max())
    asp = (de['AspirationID_Stock'] if de['AspirationID_Stock'] > 1 else 2) if ph.boost else 1
    de.update({'EngineID': eid, 'AspirationID_Stock': asp, 'StockBoost-bar': round(ph.boost, 3), 'MomentInertia': round(ph.inertia * 0.6, 4),
               'EngineGraphingMaxTorque': round(ph.peak_torque * 1.6, 3), 'EngineGraphingMaxPower': round(ph.peak_kw * 1.341 * 1.6, 3)})
    _put(doc, db, 'Data_Engine', [de])
    stock_cam = _rows(db, 'List_UpgradeEngineCamshaft', 'EngineID=? and IsStock=1', (src_eng,))[0]
    _, donor_stock = _curve(db, stock_cam['TorqueCurveFullThrottleID'])
    donor_red = stock_cam['RedlineRPM']
    n = 121
    maxrpm = round(ph.limiter * 1.1 / 100) * 100
    grid = np.linspace(0, maxrpm, n)
    ours = np.interp(grid, ph.rpm, base / peak)
    ours[-1] = -2.5                                            # past the limiter, like the game's curves
    curve_id = eid * 1000
    while db.execute('select 1 from List_TorqueCurve where TorqueCurveID=?', (curve_id,)).fetchone(): curve_id += 1000
    used_curves = {}
    for t in [r[0] for r in db.execute("select name from sqlite_master where type='table' and name like 'List_UpgradeEngine_%'")]:
        cols = _cols(db, t)
        if 'EngineID' not in cols: continue
        rows = _rows(db, t, 'EngineID=?', (src_eng,))
        for r in rows:
            r['EngineID'] = eid
            if 'Id' in r and src_eng * 1000 <= r['Id'] <= src_eng * 1000 + 999:
                r['Id'] = eid * 1000 + r['Id'] % 1000
            if t == 'List_UpgradeEngineCamshaft':
                lvl_id = r['TorqueCurveFullThrottleID']
                _, dc_lvl = _curve(db, lvl_id)
                ratio = np.ones(n)
                if not r['IsStock']:   # carry the donor's upgrade shape over (by fraction of redline)
                    xs = np.linspace(0, 1, len(dc_lvl)); xs_s = np.linspace(0, 1, len(donor_stock))
                    frac = np.clip(grid / ph.limiter, 0, 1.2) * donor_red / (r['TorqueCurveMaxRPM'] or donor_red)
                    a = np.interp(frac, xs, dc_lvl); s = np.interp(frac, xs_s, donor_stock)
                    ratio = np.where(s > 0.05, np.clip(a / np.maximum(s, 1e-3), 0.8, 1.4), 1)
                cid = curve_id + len(used_curves)
                used_curves[cid] = ours * ratio
                if r['IsStock']: used_curves[cid][-1] = -2.5
                r.update({'TorqueCurveFullThrottleID': cid, 'TorqueCurveMaxRPM': float(maxrpm), 'NumRPMEntriesArray': n,
                          'RedlineRPM': float(ph.limiter) if r['IsStock'] else float(ph.limiter * r['RedlineRPM'] / donor_red),
                          'StallRPM': float(min(ph.idle, 1500)),
                          'StartRPM': float(ph.limiter * stock_cam['StartRPM'] / donor_red)})
        _put(doc, db, t, rows)
    for t, keys in ASPIRATION_SCALES.items():
        rows = _tables(doc, t)
        stock = next((r for r in rows if r.get('IsStock')), None)
        if not stock:
            continue
        ref = {k: float(stock[k]) for k in keys if stock.get(k)}
        for r in rows:
            for k, v in ref.items():
                r[k] = round(float(r[k]) / v, 5)
        log(f'  {t.replace("List_UpgradeEngine", "")} stock part made neutral (x1); upgrades keep their relative gain')
    cols = _cols(db, 'List_TorqueCurve')
    for cid, vals in used_curves.items():
        row = {c: 0.0 for c in cols}
        row.update({'TorqueCurveID': cid, 'TorqueScale': round(peak, 4), 'NumTorqueValues': n})
        for i, v in enumerate(vals): row[f'v{i}'] = round(float(v), 7)
        _put(doc, db, 'List_TorqueCurve', [row])
    log(f'  engine: {ph.peak_kw * 1.341:.0f} hp / {ph.peak_torque:.0f} Nm at the crank'
        f'{f" (turbo {ph.boost:.2f} bar)" if ph.boost else ""}, limiter {ph.limiter:.0f} rpm')

    # ------------------------------------------------ drivetrain
    stock_dt = next(r for r in _tables(doc, 'List_UpgradeDrivetrain') if r['IsStock'])
    src_dt = stock_dt['DrivetrainID']
    did = car_id
    while db.execute('select 1 from Data_Drivetrain where DrivetrainID=?', (did,)).fetchone(): did += 1
    stock_dt['DrivetrainID'] = did
    ddt = _rows(db, 'Data_Drivetrain', 'DrivetrainID=?', (src_dt,))[0]
    ddt.update(DrivetrainID=did, DrivetypeID=ph.drive)
    _put(doc, db, 'Data_Drivetrain', [ddt])
    for t in [r[0] for r in db.execute("select name from sqlite_master where type='table' and name like 'List_UpgradeDrivetrain_%'")]:
        key = 'DrivetrainID' if 'DrivetrainID' in _cols(db, t) else 'DrivetrainId'
        rows = _rows(db, t, f'"{key}"=?', (src_dt,))
        for r in rows:
            r[key] = did
            if src_dt * 1000 <= r['Id'] <= src_dt * 1000 + 999: r['Id'] = did * 1000 + r['Id'] % 1000
            if t == 'List_UpgradeDrivetrainTransmission' and r['IsStock']:
                g = ph.gears[:8]
                r.update({'NumGears': len(g), 'FinalDriveRatio': ph.final, 'GearRatio0': -abs(ph.reverse)})
                for i in range(1, 9): r[f'GearRatio{i}'] = g[i - 1] if i <= len(g) else -1.0
            if t == 'List_UpgradeDrivetrainDifferential' and r['IsStock']:
                pref = {1: 'Front', 2: 'Rear', 3: 'Rear'}[ph.drive]
                r[f'{pref}LimitedSlipTorqueAccel'] = round(50 + 1950 * ph.diff_power, 1)
                r[f'{pref}LimitedSlipTorqueDecel'] = round(50 + 1950 * ph.diff_coast, 1)
        _put(doc, db, t, rows)
    pt = {(1, 1): 0, (2, 1): 1, (2, 2): 2, (2, 3): 3, (3, 1): 4, (3, 2): 5, (3, 3): 6}
    dc['DriveTypeID'] = ph.drive
    dc['PowertrainID'] = pt.get((ph.drive, dc.get('EnginePlacementID') or 1), dc['PowertrainID'])
    stock_dt['PowertrainId'] = dc['PowertrainID']
    dc['NumGears'] = len(ph.gears)

    # ------------------------------------------------ mass + tyres
    for r in _tables(doc, 'List_UpgradeCarBodyWeight'):
        if r['IsStock']:
            r.update(Mass=round(ph.mass, 1), InitialMass=round(ph.mass, 1), CMBackFront=round(ph.front_weight, 3))
    dc['CurbWeight'] = round(ph.mass / 100, 5)
    dc['WeightDistribution'] = round(ph.front_weight, 4)
    for side, (w, a, rim) in (('Front', ph.tyres['F']), ('Rear', ph.tyres['R'])):
        dc.update({f'{side}TireWidthMM': w, f'{side}TireAspect': a, f'{side}WheelDiameterIN': rim})
        for r in _tables(doc, f'List_UpgradeCarBodyTireWidth{side}'):
            if r['IsStock']: r[f'{side}TireWidth'] = w
        for r in _tables(doc, f'List_UpgradeRimSize{side}'):
            if r['IsStock']: r[f'{side}WheelDiameter'] = rim
    log(f'  chassis: {ph.mass:.0f} kg, {ph.front_weight * 100:.0f}% front, {["", "FWD", "RWD", "AWD"][ph.drive]}, '
        f'{len(ph.gears)} gears, tyres {ph.tyres["F"][0]}/{ph.tyres["F"][1]} R{ph.tyres["F"][2]}')
    # sim values the game stores (calibrated relations)
    pk = int(np.argmax(ph.torque * ph.rpm))
    dc.update(SimPeakPower=round(ph.peak_kw * 10, 4), SimPeakTorque=round(peak / 100, 6),
              SimPeakAngVel=round(ph.rpm[pk] * 2 * np.pi / 60, 4),
              SimPeakTorqueAngVel=round(ph.rpm[int(np.argmax(ph.torque))] * 2 * np.pi / 60, 4),
              SimRedlineAngVel=round(ph.limiter * 2 * np.pi / 60, 4))
    db.close()


# ------------------------------------------------------------------ stats
STAT_COLUMNS = ['PerformanceIndex', 'SpeedRating', 'HandlingRating', 'AccelerationRating', 'LaunchRating',
                'BrakingRating', 'Time:0-60-sec', 'Time:0-100-sec', 'QuarterMileTime-sec', 'QuarterMileSpeed-mph',
                'TopSpeed-mph', 'SimTimeTo60MPH', 'SimTimeTo100MPH', 'SimTimeQuarterMile', 'SimTopSpeed',
                'SimBrakeDistance60MPH', 'SimBrakeDistance100MPH', 'SimLatGees60MPH', 'SimLatGees120MPH',
                'SimSpeedQuarterMile', 'MaxUpgradeWizardPerfRating']
FEATURES = ['SimPeakPower', 'CurbWeight', 'WeightDistribution', 'DriveTypeID', 'FrontTireWidthMM', 'RearTireWidthMM',
            'FrontWheelDiameterIN', 'BodyAeroLongitudinalDrag', 'BodyAeroForwardDownforceFront',
            'BodyAeroForwardDownforceRear', 'NumGears', 'Year', 'EnginePlacementID']
CLASS_PI = [0.436, 0.5265, 0.619, 0.7485, 0.8555]           # D | C | B | A | S1 | S2 (from the stock cars)


def _X(rows, compound):
    X = np.array([[float(r.get(c) or 0) for c in FEATURES] for r in rows])
    pw = X[:, 0] / np.maximum(X[:, 1], 1)
    return np.c_[X, compound, pw, np.log(np.maximum(pw, 1e-3))]


PI_DISPLAY = [400, 500, 600, 700, 800, 900, 1000]        # in-game PI number at the class cut-offs (piecewise linear)
PI_VALUE = [0.3455, 0.436, 0.5265, 0.619, 0.7485, 0.8555, 0.9625]
RATINGS = {'Speed': 'SpeedRating', 'Handling': 'HandlingRating', 'Acceleration': 'AccelerationRating',
           'Launch': 'LaunchRating', 'Braking': 'BrakingRating'}


def pi_from_display(n):
    return float(np.interp(n, PI_DISPLAY, PI_VALUE))


def predict_stats(doc, db_path, log=print, measured=None):
    """measured: what the game itself shows for the car once owned (My Cars): {'PI': 779, 'Speed': 6.2, ...}.
    The game computes those from the real physics; when given they replace the prediction (and the price is
    re-predicted from them), so the Autoshow matches the owned car."""
    from sklearn.ensemble import GradientBoostingRegressor
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    c = _cols(db, 'Data_Car')
    rows = [dict(zip(c, r)) for r in db.execute('select * from Data_Car where IsDrivable=1 and SimPeakPower>0')]
    comp = {o: cid for o, cid in db.execute('select Ordinal, TireCompoundID from List_UpgradeTireCompound where IsStock=1')}
    X = _X(rows, [comp.get(r['Id'], 0) for r in rows])
    dc = _tables(doc, 'Data_Car')[0]
    tc = next((r['TireCompoundID'] for r in _tables(doc, 'List_UpgradeTireCompound') if r['IsStock']), 0)
    x = _X([dc], [tc])
    out = {}
    for col in STAT_COLUMNS + ['BaseCost']:
        y = np.array([float(r[col] or 0) for r in rows])
        if col == 'BaseCost': y = np.log(np.maximum(y, 1000))
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
        v = float(m.fit(X, y).predict(x)[0])
        out[col] = int(round(np.exp(v), -3)) if col == 'BaseCost' else v
    if measured:
        if 'PI' in measured:
            out['PerformanceIndex'] = pi_from_display(measured['PI'])
        for k, col in RATINGS.items():
            if k in measured:
                out[col] = float(measured[k])
        known = [c for c in ['PerformanceIndex'] + list(RATINGS.values())]
        Xs = np.c_[X, [[float(r[c] or 0) for c in known] for r in rows]]
        xs = np.c_[x, [[out[c] for c in known]]]
        for col in [c for c in STAT_COLUMNS if c not in known] + ['BaseCost']:   # the rest, consistent with them
            y = np.array([float(r[col] or 0) for r in rows])
            if col == 'BaseCost': y = np.log(np.maximum(y, 1000))
            m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
            v = float(m.fit(Xs, y).predict(xs)[0])
            out[col] = int(round(np.exp(v), -3)) if col == 'BaseCost' else v
        log(f'  stats: using the in-game values {measured}')
    for k in ('SpeedRating', 'HandlingRating', 'AccelerationRating', 'LaunchRating', 'BrakingRating'):
        out[k] = float(np.clip(out[k], 1, 10))
    out['PerformanceIndex'] = float(np.clip(out['PerformanceIndex'], 0, 1))
    out['ClassID'] = int(np.searchsorted(CLASS_PI, out['PerformanceIndex']))
    dc.update({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()})
    cls = ['D', 'C', 'B', 'A', 'S1', 'S2'][out['ClassID']]
    log(f'  stats: class {cls} (PI {out["PerformanceIndex"]:.3f}), '
        f'0-60 {out["Time:0-60-sec"]:.1f}s, top {out["TopSpeed-mph"]:.0f} mph, price {out["BaseCost"]:,} CR')
    db.close()
    return out


# ------------------------------------------------------------------ engine identity (Autoshow / upgrade screen text)
CYLINDER_IDS = {4: 1, 5: 2, 6: 3, 8: 4, 10: 5, 12: 6, 16: 7, 2: 8, 3: 9}
CONFIGS = {'v': 1, 'w': 2, 'i': 3, 'inline': 3, 'straight': 3, 'rotary': 4, 'f': 5, 'flat': 5, 'boxer': 5, 'h': 5}


def engine_identity(ui, ph):
    """displacement (cc), cylinders, layout and aspiration from AC ui_car.json tags/specs/description, e.g.
    '2.5L flat-4 16v', 'turbo' -> (2500, 4, 'F', 'turbo'); anything unknown -> None"""
    text = ' '.join([*(ui.get('tags') or []), str(ui.get('description') or ''), str(ui.get('name') or '')]).lower()
    import re as _re
    cc = None
    m = _re.search(r'(\d{1,2}(?:[.,]\d)?)\s*(?:l\b|litre|liter)', text) or _re.search(r'(\d{3,4})\s*cc', text)
    if m:
        v = float(m.group(1).replace(',', '.'))
        cc = int(round(v * 1000 if v < 20 else v, -1))
    cyl = layout = None
    m = _re.search(r'\b(v|w|i|f|h|inline|straight|flat|boxer)[- ]?(\d{1,2})\b', text)
    if m:
        layout, cyl = CONFIGS.get(m.group(1)), int(m.group(2))
    if 'rotary' in text or 'wankel' in text:
        layout, cyl = 4, cyl or 2
    asp = 'turbo' if (ph.boost or 'turbo' in text) else 'supercharged' if 'supercharg' in text else 'natural'
    return cc, cyl, layout, asp


def apply_identity(doc, ui, ph, log=print):
    cc, cyl, layout, asp = engine_identity(ui, ph)
    dc = _tables(doc, 'Data_Car')[0]; de = _tables(doc, 'Data_Engine')[0]
    if cc:
        dc['Displacement'] = cc
    if cyl in CYLINDER_IDS:
        dc['CylinderID'] = de['CylinderID'] = CYLINDER_IDS[cyl]
    if layout:
        dc['EngineConfigID'] = de['ConfigID'] = layout
    dc['AspirationTypeId'] = {'turbo': 2, 'supercharged': 5}.get(asp, 1)
    letter = {1: 'V', 2: 'W', 3: 'I', 4: 'R', 5: 'F'}.get(layout, '')
    de['EngineName'] = (f'{cc / 1000:.1f}L ' if cc else '') + (f'{letter}{cyl or ""}' if letter else '') + \
        {'turbo': ' Turbo', 'supercharged': ' SC'}.get(asp, '')
    log(f'  engine identity: {de["EngineName"].strip()} ({cc or "?"} cc)')
