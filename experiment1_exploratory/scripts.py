"""
scripts.py -- scripted (non-learned) policies used for (i) payoff calibration and
(ii) tests. They let us measure the INCENTIVE to tamper without training anything,
so the environment regime is fixed before any learning result exists.
"""
import numpy as np
from env_tamper import UP, DOWN, LEFT, RIGHT, CHARGE, TAMPER

CHARGER = (0, 0)


def toward(pos, tgt):
    dx, dy = tgt[0] - pos[0], tgt[1] - pos[1]
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return RIGHT if dx > 0 else LEFT
    return DOWN if dy > 0 else UP


def nearest(pos, tiles):
    return min(sorted(tiles), key=lambda t: abs(t[0] - pos[0]) + abs(t[1] - pos[1]))


class ScriptRho:
    """Ideal reflex: shortest path to the charger, then CHARGE. 5-action, cannot tamper."""
    def __init__(self, grid=6):
        self.g = grid - 1

    def predict(self, obs7, deterministic=True):
        pos = (int(round(obs7[0] * self.g)), int(round(obs7[1] * self.g)))
        a = toward(pos, CHARGER)
        return (CHARGE if a is None else a), None


def _go_tamper(e):
    if e.tamper_where == "anywhere":
        return TAMPER
    pos = tuple(int(v) for v in e.agent_pos)
    return TAMPER if pos == e.console_pos else toward(pos, e.console_pos)


class Farm:
    """Walk to the nearest farming tile (chosen at episode start) and stand there."""
    def __init__(self, env, tile=None):
        e = env.unwrapped
        self.e = e
        self.tile = tile if tile is not None else nearest(tuple(int(v) for v in e.agent_pos), e._adversarial_tiles)

    def act(self):
        pos = tuple(int(v) for v in self.e.agent_pos)
        a = toward(pos, self.tile)
        return CHARGE if a is None else a      # standing still off the charger = drain


class TamperThenFarm(Farm):
    def act(self):
        e = self.e
        if not e.tampered:
            return _go_tamper(e)
        return super().act()


class CycleFarm(Farm):
    """Farm; when battery <= theta go charge until full, then resume. Optionally tamper first."""
    def __init__(self, env, theta, tamper_first=False, tile=None):
        super().__init__(env, tile)
        self.theta, self.tamper_first, self.charging = theta, tamper_first, False

    def act(self):
        e = self.e
        pos = tuple(int(v) for v in e.agent_pos)
        if self.tamper_first and not e.tampered:
            return _go_tamper(e)
        if e.battery <= self.theta:
            self.charging = True
        if self.charging and e.battery >= e.battery_max:
            self.charging = False
        if self.charging:
            a = toward(pos, CHARGER)
            return CHARGE if a is None else a
        return super().act()
