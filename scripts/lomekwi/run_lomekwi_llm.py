"""Byproduct->machine tool world (obfuscated, leak-free) -- v1 world/parser library.

Reused as a library by lomekwi.py (_clean) and replay_lomekwi.py (State/make_world,
for rows without 'n_types').

Design:
  - n locked doors. key_i opens door_i; the index correspondence is the only
    transparent structure (the brute baseline).
  - examine(door) drops a key for a UNIFORMLY RANDOM door (over all n, WITH
    replacement) + one generic byproduct. So brute is a coupon-collector grind.
  - combine(byproduct, byproduct) -> a MACHINE (persists).
  - use(machine, door_i) -> yields key_i (the machine persists).
  - use(key_i, door_i) -> opens door_i.
  The machine converts the random-draw grind into deterministic key-generation;
  it is the rational (optional) tool at n >= 3.

Obfuscation (scripts.shared.obfuscation): per episode we draw a fresh uniformly-random
relabeling of the four latent roles {door, key, byproduct, machine}. The two
tool roles (byproduct, machine) MUST be obfuscated -- naming them "shard"/
"machine" would leak that the junk combines into a functional tool. door/key are
the brute baseline; their words are harmless, but we relabel all four for a clean
Sym(V) expectation. Results are reported as a mean over relabelings.

We never tell the agent there are keys, byproducts, or a machine; it discovers
the affordances by acting. examine(<byproduct>) gives a SUBTLE interaction hint,
toggleable via State's `hint` flag.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from scripts.shared.obfuscation import assign


def make_world(relabel_seed: int, n: int) -> dict:
    labels = assign(["door", "key", "byproduct", "machine"], seed=relabel_seed)
    return {
        "n": n,
        "door_base": labels["door"],
        "key_base": labels["key"],
        "byproduct": labels["byproduct"],
        "machine": labels["machine"],
        "doors": [f"{labels['door']}_{i}" for i in range(1, n + 1)],
    }


@dataclass
class State:
    world: dict
    drop_rng: random.Random
    hint: bool = True
    byproducts: int = 0
    held_keys: set = field(default_factory=set)   # door indices held as keys
    has_machine: bool = False
    opened: dict = field(default_factory=dict)     # door_idx -> (method, turn)
    machine_used_on: set = field(default_factory=set)  # door indices
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
            j = self.drop_rng.randint(1, self.world["n"])  # random door's key
            self.held_keys.add(j)
            self.byproducts += 1
            return (f"You search {tok}. Out falls {self.key_tok(j)} and a "
                    f"{self.world['byproduct']}. You now hold "
                    f"{self.byproducts} {self.world['byproduct']}(s).")
        if tok == self.world["byproduct"]:
            if self.hint:
                return (f"A small dull {self.world['byproduct']}. It is faintly "
                        f"warm, and the warmth grows when several are held together.")
            return f"A small dull {self.world['byproduct']}. Nothing remarkable."
        if tok == self.world["machine"] and self.has_machine:
            return (f"A {self.world['machine']}. It has a recess; you are unsure "
                    f"what it does.")
        if self.key_idx(tok) is not None:
            return f"{tok}: a small fitted object."
        return f"You examine {tok}: nothing of note."

    def combine(self, a: str, b: str) -> str:
        if a == self.world["byproduct"] and b == self.world["byproduct"]:
            if self.byproducts >= 2:
                if not self.has_machine:
                    self.has_machine = True
                    return (f"You combine two {self.world['byproduct']}s. They fuse "
                            f"into a {self.world['machine']} (new). It persists.")
                return f"You already hold a {self.world['machine']}."
            return f"You need at least two {self.world['byproduct']}s to combine."
        return f"You combine {a} and {b}: nothing happens."

    def use(self, item: str, target: str) -> str:
        di = self.door_idx(target)
        if di is None or not (1 <= di <= self.world["n"]):
            return f"There is no {target}."
        if di in self.opened:
            return f"{target} is already open."
        # operate the machine -> yields that door's key
        if item == self.world["machine"]:
            if not self.has_machine:
                return f"You don't have a {self.world['machine']}."
            self.held_keys.add(di)
            self.machine_used_on.add(di)
            return (f"You operate the {self.world['machine']} on {target}. It "
                    f"yields {self.key_tok(di)}. (the {self.world['machine']} "
                    f"remains with you.)")
        # use a key
        ki = self.key_idx(item)
        if ki is not None:
            if ki not in self.held_keys:
                return f"You don't have {item}."
            if ki == di:
                method = "machine" if di in self.machine_used_on else "drop"
                self.opened[di] = (method, self.turn)
                return f"You use {item} on {target}: it fits. {target} opens."
            return f"You use {item} on {target}: it does not fit."
        if item == self.world["byproduct"]:
            return f"You press the {self.world['byproduct']} to {target}: nothing."
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


ACT = re.compile(r"^\s*(?:>?\s*)?(examine|combine|use)\b[\s:]*(.*)\s*$", re.IGNORECASE)


def _clean(t: str) -> str:
    return re.sub(r"^(a|an|the)\s+", "", t.strip().lower())


def parse(line: str):
    m = ACT.match(line.strip())
    if not m:
        return None
    v, rest = m.group(1).lower(), m.group(2).strip()
    toks = [_clean(x) for x in re.split(r"[+,\s]+|\bwith\b|\bto\b|\bon\b|\band\b", rest)
            if _clean(x)]
    if v == "examine" and toks:
        return ("examine", toks[0])
    if v == "combine" and len(toks) >= 2:
        return ("combine", toks[0], toks[1])
    if v == "use" and len(toks) >= 2:
        return ("use", toks[0], toks[1])
    return None


def extract(text: str):
    for ln in text.splitlines():
        a = parse(ln)
        if a:
            return a
    return None


SYS = ("You are an agent in a vault and must open every locked door using as few "
       "actions as possible. End each turn with exactly one action on its own "
       "line: examine <object>, combine <a> <b>, or use <object> <door>. Think "
       "briefly, then act. Minimize total actions.")
