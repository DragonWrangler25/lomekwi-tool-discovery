"""Multi-type construction world and state primitives, shared by Lomekwi (lomekwi.py)
and replay_lomekwi.py.

Compared to the single-byproduct v1 world (run_lomekwi_llm.py), this module hardens
CONSTRUCTION: with only one generic byproduct, "combine(bp,bp) -> machine" is
trivially discoverable ("I hold several identical junk items; combine is a verb;
try it"), so models build the tool ~always, hint-independently, and built_tool
carries no signal. Here:
  - T distinct byproduct TYPES, each its own nonsense label.
  - examine(door) drops ONE byproduct of a UNIFORMLY RANDOM type + the usual
    random door-key (keys are coupon-collector exactly as v1 -> BRUTE UNCHANGED).
  - the machine needs a SPECIFIC unordered pair of DISTINCT types {A,B}
    (randomized per episode). combine(x,y) builds iff {type(x),type(y)}=={A,B};
    every other combine -> "nothing happens". Now "which two components
    combine?" is a genuine search (C(T,2)+T candidates, one correct).

Brute baseline, the machine interface (use(machine,door)->key; use(key,door)
->open), the index correspondence, and "the agent is told nothing" are all
shared with v1.

lomekwi.py's make_world() delegates to this module's make_world() verbatim
(for n_types >= 2), so Lomekwi worlds stay byte-identical to this module's for
the same seeds; replay_lomekwi.py reuses this module's State for the same reason.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from scripts.shared.obfuscation import assign


def recipe_indices(relabel_seed: int, n_types: int) -> tuple[int, int]:
    """The two DISTINCT byproduct-type indices that build the machine this
    episode. Deterministic in relabel_seed so the world replays exactly."""
    rng = random.Random(relabel_seed * 2654435761 + 7)
    a, b = rng.sample(range(n_types), 2)
    return tuple(sorted((a, b)))


def make_world(relabel_seed: int, n: int, n_types: int = 3) -> dict:
    elements = ["door", "key", "machine"] + [f"bp{i}" for i in range(n_types)]
    labels = assign(elements, seed=relabel_seed)
    types = [labels[f"bp{i}"] for i in range(n_types)]
    ri, rj = recipe_indices(relabel_seed, n_types)
    return {
        "n": n,
        "n_types": n_types,
        "door_base": labels["door"],
        "key_base": labels["key"],
        "machine": labels["machine"],
        "types": types,                      # all byproduct-type labels
        "recipe": sorted([types[ri], types[rj]]),  # the two that combine
        "doors": [f"{labels['door']}_{i}" for i in range(1, n + 1)],
    }


def world_from_labels(labels: dict, n: int) -> dict:
    """Reconstruct a world dict from the labels recorded in a result/JSONL row,
    bypassing assign(). make_world derives its label strings from the GLOBAL,
    mutable scripts.shared.obfuscation.DEFAULT_SCHEME, so a replay that re-derives them
    only matches if that global happens to hold the same scheme the run used
    (see sweep_config.OBFUSCATION_SCHEME). Reconstructing from the recorded
    labels removes that coupling: the State reads only these strings, so replay
    is byte-exact regardless of the ambient scheme. Mirrors make_world's output
    for every field State touches."""
    types = list(labels["types"])
    return {
        "n": n,
        "n_types": len(types),
        "door_base": labels["door_base"],
        "key_base": labels["key_base"],
        "machine": labels["machine"],
        "types": types,
        "recipe": sorted(labels["recipe"]),
        "doors": [f"{labels['door_base']}_{i}" for i in range(1, n + 1)],
    }


@dataclass
class State:
    world: dict
    drop_rng: random.Random
    hint: bool = True
    byproducts: dict = field(default_factory=dict)  # type_label -> count held
    held_keys: set = field(default_factory=set)
    has_machine: bool = False
    opened: dict = field(default_factory=dict)       # door_idx -> (method, turn)
    machine_used_on: set = field(default_factory=set)
    turn: int = 0

    # --- helpers ---
    def door_idx(self, tok: str):
        b = self.world["door_base"]
        m = re.fullmatch(rf"{re.escape(b)}_(\d+)", tok)
        return int(m.group(1)) if m else None

    def key_idx(self, tok: str):
        b = self.world["key_base"]
        m = re.fullmatch(rf"{re.escape(b)}_(\d+)", tok)
        return int(m.group(1)) if m else None

    def key_tok(self, i: int) -> str:
        return f"{self.world['key_base']}_{i}"

    # --- actions ---
    def examine(self, tok: str) -> str:
        di = self.door_idx(tok)
        if di is not None and 1 <= di <= self.world["n"]:
            ty = self.world["types"][self.drop_rng.randrange(self.world["n_types"])]
            j = self.drop_rng.randint(1, self.world["n"])
            self.byproducts[ty] = self.byproducts.get(ty, 0) + 1
            self.held_keys.add(j)
            return (f"You search {tok}. Out falls {self.key_tok(j)} and a {ty}. "
                    f"You now hold {self.byproducts[ty]} {ty}(s).")
        if tok in self.world["types"]:
            if self.hint:
                return (f"A small dull {tok}. It feels faintly active, and the "
                        f"sensation shifts when held alongside a different kind.")
            return f"A small dull {tok}. Nothing remarkable."
        if tok == self.world["machine"] and self.has_machine:
            return (f"A {self.world['machine']}. It has a recess; you are unsure "
                    f"what it does.")
        if self.key_idx(tok) is not None:
            return f"{tok}: a small fitted object."
        return f"You examine {tok}: nothing of note."

    def combine(self, a: str, b: str) -> str:
        types = self.world["types"]
        if a in types and b in types:
            # need to actually hold the components
            have_a = self.byproducts.get(a, 0)
            have_b = self.byproducts.get(b, 0)
            enough = (have_a >= 1 and have_b >= 1) if a != b else have_a >= 2
            if not enough:
                return f"You combine {a} and {b}: nothing happens."
            if a != b and sorted([a, b]) == self.world["recipe"]:
                if not self.has_machine:
                    self.has_machine = True
                    return (f"You combine a {a} and a {b}. They fuse into a "
                            f"{self.world['machine']} (new). It persists.")
                return f"You already hold a {self.world['machine']}."
            return f"You combine {a} and {b}: nothing happens."
        return f"You combine {a} and {b}: nothing happens."

    def use(self, item: str, target: str) -> str:
        di = self.door_idx(target)
        if di is None or not (1 <= di <= self.world["n"]):
            return f"There is no {target}."
        if di in self.opened:
            return f"{target} is already open."
        if item == self.world["machine"]:
            if not self.has_machine:
                return f"You don't have a {self.world['machine']}."
            self.held_keys.add(di)
            self.machine_used_on.add(di)
            return (f"You operate the {self.world['machine']} on {target}. It "
                    f"yields {self.key_tok(di)}. (the {self.world['machine']} "
                    f"remains with you.)")
        ki = self.key_idx(item)
        if ki is not None:
            if ki not in self.held_keys:
                return f"You don't have {item}."
            if ki == di:
                method = "machine" if di in self.machine_used_on else "drop"
                self.opened[di] = (method, self.turn)
                return f"You use {item} on {target}: it fits. {target} opens."
            return f"You use {item} on {target}: it does not fit."
        if item in self.world["types"]:
            return f"You press the {item} to {target}: nothing."
        return f"You use {item} on {target}: nothing happens."

    def solved(self) -> bool:
        return len(self.opened) == self.world["n"]

    def initial_obs(self) -> str:
        return (
            f"You are in a sealed vault. There are {self.world['n']} locked doors: "
            f"{', '.join(self.world['doors'])}.\n\n"
            f"GOAL: open ALL the doors, using as few actions as possible.\n\n"
            f"You may act on objects with these verbs (one action per turn, on its "
            f"own line):\n"
            f"  examine <object>      - inspect or search an object\n"
            f"  combine <a> <b>       - combine two objects you hold\n"
            f"  use <object> <door>   - use an object on a door\n"
            f"You hold nothing yet. Things you obtain persist unless stated otherwise."
        )
