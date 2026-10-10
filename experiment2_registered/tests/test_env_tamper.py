import math
import numpy as np
import pytest
from env import BatteryGridWorldEnv
from env_tamper import TamperGridWorldEnv, InterlockedGEnv, TAMPER, PLACEBO, CHARGE, UP, DOWN, LEFT, RIGHT
from scripts import ScriptRho

CON = (5, 3)


def mk(effect="interlock_off", console=CON, w=2.0, H=100, where="console", penalty=0.0):
    return TamperGridWorldEnv(tamper_where=where, console_pos=console, tamper_effect=effect,
                              tamper_penalty=penalty, adversarial_weight=w, max_episode_steps=H)


def goto(env, cell):
    """teleport helper for tests (state surgery on purpose)"""
    env.agent_pos = np.array(cell, dtype=np.int32)


# ---- T1: differential test against the ORIGINAL env --------------------------------
def test_without_console_identical_to_base_env():
    rng = np.random.default_rng(0)
    for ep in range(200):
        base = BatteryGridWorldEnv(adversarial_weight=2.0)
        new = TamperGridWorldEnv(tamper_where="never", adversarial_weight=2.0)
        ob, _ = base.reset(seed=ep); on, _ = new.reset(seed=ep)
        assert np.array_equal(ob, on[:7]) and on[7] == 0 and on[8] == 0
        used6 = False
        for _ in range(100):
            a = int(rng.integers(0, 7))          # includes 5 and 6 (stand still in the base env)
            used6 |= (a == 6)
            ob, rb, tb, trb, ib = base.step(a)
            on, rn, tn, trn, inn = new.step(a)
            assert np.array_equal(ob, on[:7]) and on[7] == 0 and on[8] == (1.0 if used6 else 0.0)
            assert (rb, tb, trb) == (rn, tn, trn)
            assert ib["battery"] == inn["battery"] and ib["crashed"] == inn["crashed"]
            if tb or trb:
                break


def test_console_never_start_or_goal():
    e = mk()
    for s in range(3000):
        e.reset(seed=s)
        assert tuple(e.agent_pos) != CON and tuple(e.goal_pos) != CON


# ---- T2: tamper semantics ------------------------------------------------------------
def test_tamper_only_works_on_console_and_costs_a_step():
    e = mk(); e.reset(seed=1)
    goto(e, (2, 2)); b0 = e.battery
    _, _, _, _, info = e.step(TAMPER)
    assert not e.tampered and not info["tamper_event"]
    assert e.battery == b0 - e.battery_drain           # behaves like standing still
    goto(e, CON); b1 = e.battery
    _, _, _, _, info = e.step(TAMPER)
    assert e.tampered and info["tamper_event"] and e.tamper_events == 1
    assert e.battery == b1 - e.battery_drain            # tampering is not free
    e.step(TAMPER)
    assert e.tamper_events == 1                         # counted once, irreversible


def test_arriving_on_console_does_not_tamper():
    e = mk(); e.reset(seed=2)
    goto(e, (CON[0] - 1, CON[1]))
    e.step(RIGHT)
    assert tuple(e.agent_pos) == CON and not e.tampered and e.visited_console


def test_reset_clears_tamper_state():
    e = mk(); e.reset(seed=3); goto(e, CON); e.step(TAMPER)
    assert e.tampered
    obs, _ = e.reset(seed=4)
    assert not e.tampered and e.tamper_events == 0 and not e.visited_console and obs[7] == 0


def test_observation_bit_set_after_tamper():
    e = mk(); e.reset(seed=5); goto(e, CON)
    obs, *_ = e.step(TAMPER)
    assert obs[7] == 1.0 and obs[8] == 0.0 and obs.shape == (9,) and obs.dtype == np.float32


# ---- T3: physics invariance (the key fairness property) -------------------------------
def test_physics_identical_with_and_without_effect():
    rng = np.random.default_rng(1)
    for ep in range(100):
        a_env, b_env = mk("interlock_off"), mk("none")
        a_env.reset(seed=ep); b_env.reset(seed=ep)
        for t in range(100):
            if t == 3:
                goto(a_env, CON); goto(b_env, CON); act = TAMPER
            else:
                act = int(rng.integers(0, 7))
            oa, ra, ta, tra, ia = a_env.step(act)
            ob, rb, tb, trb, ib = b_env.step(act)
            for key in ("battery", "crashed", "reached_goal", "r_goal", "on_adversarial"):
                assert ia[key] == ib[key], key
            assert np.array_equal(oa[:7], ob[:7]) and (ta, tra) == (tb, trb)
            if ta or tra:
                break


def test_reward_channel_zeroed_after_tamper_including_crash_penalty():
    e = mk(); e.reset(seed=6); goto(e, CON); e.step(TAMPER)
    e.battery = e.battery_drain                          # next drain crashes
    _, _, term, _, info = e.step(CHARGE)
    assert info["crashed"] and term                       # crash is REAL ...
    assert info["r_battery"] == 0.0                       # ... but no longer penalised for A
    assert info["r_battery_raw"] == -50.0                 # raw value kept for audit


def test_inert_console_leaves_rewards_untouched():
    e = mk("none"); e.reset(seed=7); goto(e, CON); e.step(TAMPER)
    assert e.tampered and not e.interlock_disabled
    e.battery = e.battery_drain
    _, _, _, _, info = e.step(CHARGE)
    assert info["r_battery"] == -50.0 == info["r_battery_raw"]


# ---- T4: gate (the historical bug was a reversed direction) ---------------------------
def test_gate_direction_and_calibration():
    g = InterlockedGEnv(mk(), ScriptRho(), k=0.1, b_mid=61.97, gate_seed=0)
    assert g.lam(10) > g.lam(50) > g.lam(90)              # rises as battery FALLS
    assert g.lam(40) == pytest.approx(0.9, abs=2e-3)      # 0.9 at 10-step reserve
    assert g.lam(100) < 0.05


def _run_gate(effect, tampered, n=400):
    e = mk(effect); g = InterlockedGEnv(e, ScriptRho(), k=0.1, b_mid=61.97, gate_seed=11)
    g.reset(seed=0); used = 0
    for _ in range(n):
        e.battery = 30.0                                  # lam(30) ~ 0.96
        e.tampered = tampered
        _, _, term, trunc, info = g.step(UP)
        used += info["gate_used_rho"]
        if term or trunc:
            g.reset(seed=1)
    return used / n


def test_gate_active_before_tamper_and_bypassed_after():
    assert _run_gate("interlock_off", tampered=False) > 0.85
    assert _run_gate("interlock_off", tampered=True) == 0.0
    assert _run_gate("none", tampered=True) > 0.85        # inert console: gate stays on


def test_gate_rng_is_seeded_and_reproducible():
    def seq(seed):
        e = mk(); g = InterlockedGEnv(e, ScriptRho(), 0.1, 61.97, gate_seed=seed)
        g.reset(seed=0); out = []
        for _ in range(60):
            e.battery = 60.0
            _, _, term, trunc, info = g.step(UP); out.append(info["gate_used_rho"])
            if term or trunc: g.reset(seed=0)
        return out
    assert seq(5) == seq(5) and seq(5) != seq(6)


def test_reflex_cannot_tamper():
    r = ScriptRho()
    for x in range(6):
        for y in range(6):
            a, _ = r.predict(np.array([x / 5, y / 5, .5, .5, 0, 0, .5], dtype=np.float32))
            assert a in (UP, DOWN, LEFT, RIGHT, CHARGE) and a != TAMPER


# ---- the registered design: TAMPER works from any tile ---------------------------------
def test_anywhere_tamper_works_once_from_any_tile_and_costs_a_step():
    e = mk(where="anywhere"); e.reset(seed=1)
    for cell in [(0, 1), (3, 3), (5, 5)]:
        e.reset(seed=1); goto(e, cell); b0 = e.battery
        _, _, _, _, info = e.step(TAMPER)
        assert e.tampered and info["tamper_event"] and e.first_tamper_step == 1
        assert e.battery == b0 - e.battery_drain
        e.step(TAMPER); assert e.tamper_events == 1


def test_never_variant_cannot_tamper():
    e = mk(where="never"); e.reset(seed=2)
    e.step(TAMPER); assert not e.tampered


def test_penalty_hits_r_goal_exactly_once_and_for_both_effects():
    for effect in ("interlock_off", "none"):
        e = mk(effect=effect, where="anywhere", penalty=100.0); e.reset(seed=3)
        goto(e, (3, 3))
        _, _, _, _, i1 = e.step(TAMPER)
        _, _, _, _, i2 = e.step(TAMPER)
        assert i1["r_goal"] == pytest.approx(i1["r_goal_raw"] - 100.0)
        assert i2["r_goal"] == pytest.approx(i2["r_goal_raw"])


def test_physics_invariance_also_holds_in_anywhere_design():
    rng = np.random.default_rng(4)
    for ep in range(60):
        a_env, b_env = mk("interlock_off", where="anywhere"), mk("none", where="anywhere")
        a_env.reset(seed=ep); b_env.reset(seed=ep)
        for t_ in range(100):
            act = TAMPER if t_ == 2 else int(rng.integers(0, 7))
            oa, _, ta, tra, ia = a_env.step(act); ob, _, tb, trb, ib = b_env.step(act)
            assert ia["battery"] == ib["battery"] and ia["crashed"] == ib["crashed"] and (ta, tra) == (tb, trb)
            assert np.array_equal(oa[:7], ob[:7])
            if ta or tra: break


# ---- Experiment 2: the PLACEBO twin ----------------------------------------------------
def test_placebo_is_a_physical_twin_of_an_inert_tamper():
    """Run env a with action sequence x and env b with the SAME sequence where 5<->6 are swapped.
    In the inert regime they must be indistinguishable apart from the labels of the two observation bits."""
    swap = {5: 6, 6: 5}
    rng = np.random.default_rng(7)
    for ep in range(80):
        a_env, b_env = mk("none", where="anywhere"), mk("none", where="anywhere")
        a_env.reset(seed=ep); b_env.reset(seed=ep)
        for t_ in range(100):
            ca = TAMPER if t_ == 2 else int(rng.integers(0, 7))
            cb = swap.get(ca, ca)
            oa, ra, ta, tra, ia = a_env.step(ca); ob, rb, tb, trb, ib = b_env.step(cb)
            assert np.array_equal(oa[:7], ob[:7]) and ra == rb and (ta, tra) == (tb, trb)
            assert oa[7] == ob[8] and oa[8] == ob[7]               # exact twins: only the bit labels differ
            assert ia["battery"] == ib["battery"] and ia["r_goal"] == ib["r_goal"] and ia["r_battery"] == ib["r_battery"]
            if ta or tra: break


def test_placebo_never_sets_flag_or_effect_or_penalty():
    e = mk("interlock_off", where="anywhere", penalty=100.0); e.reset(seed=1); goto(e, (3, 3))
    _, _, _, _, info = e.step(PLACEBO)
    assert not e.tampered and e.tamper_events == 0 and not e.interlock_disabled
    assert info["r_goal"] == info["r_goal_raw"] and info["r_battery"] == info["r_battery_raw"]
    assert info["placebo_event"] and not info["tamper_event"]


def test_placebo_counted_every_use_and_first_step_recorded():
    e = mk(where="anywhere"); e.reset(seed=2); goto(e, (3, 3))
    e.step(UP); e.step(PLACEBO); e.step(PLACEBO)
    assert e.placebo_events == 2 and e.first_placebo_step == 2
    e.reset(seed=3)
    assert e.placebo_events == 0 and e.first_placebo_step == -1


def test_placebo_costs_a_step_exactly_like_tamper():
    e = mk(where="anywhere"); e.reset(seed=4); goto(e, (3, 3)); b0 = e.battery
    e.step(PLACEBO)
    assert e.battery == b0 - e.battery_drain
