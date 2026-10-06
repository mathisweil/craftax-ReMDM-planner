#!/usr/bin/env python3
"""Render the README animation: one Craftax Classic world, before and after fine-tuning.

The DAgger checkpoint and its return-weighted ELBO fine-tune (``baseline_rl``)
play the same world under the key chain of ``build_eval_fn``: the raw env, one
environment, a fresh 50-step plan every 8 steps, up to EVAL_STEPS. The world
is the lowest seed in 0..31 where both episodes end, last at least 16 steps
and land within one unlock of their agent's median; the sweep is printed.
The cache keeps each agent's actions with its unlock and health traces. Every
frame comes from replaying those actions through the env on the same key
chain, checked against both traces. With the local rollout cache, a re-render
needs JAX on CPU but neither the model nor a GPU; without it, both models run
once to rebuild the cache.

Usage:
  env JAX_PLATFORMS=cpu uv run python scripts/render_rollout_gif.py \
      --out ../mathisweil/assets/craftax-before-after.gif
"""

from __future__ import annotations

import argparse
import functools
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Chrome block shared with sibling GIF renderers; keep identical.
W, H, M, FPS = 840, 480, 24, 10
BG, INK, MUTED, FAINT = "#0d1117", "#e6edf3", "#8b949e", "#6e7681"
RULE, CELL, ACCENT, HARM = "#30363d", "#21262d", "#3987e5", "#e66767"
GAIN = ACCENT
FONTS = {
    "serif": ("/usr/share/fonts/stix-fonts/STIX2Text-Regular.otf", "STIXGeneral.ttf"),
    "italic": (
        "/usr/share/fonts/stix-fonts/STIX2Text-Italic.otf",
        "STIXGeneralItalic.ttf",
    ),
    "label": (
        "/usr/share/fonts/adobe-source-code-pro/SourceCodePro-Medium.otf",
        "DejaVuSansMono.ttf",
    ),
    "value": (
        "/usr/share/fonts/adobe-source-code-pro/SourceCodePro-Regular.otf",
        "DejaVuSansMono.ttf",
    ),
}


@functools.cache
def font(role: str, size: int) -> ImageFont.FreeTypeFont:
    path, fallback = FONTS[role]
    if not Path(path).exists():
        import matplotlib  # type: ignore[import-not-found, unused-ignore]

        path = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / fallback)
    raqm = features.check("raqm")
    layout = ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC
    return ImageFont.truetype(path, size, layout_engine=layout)


def caps(
    d: ImageDraw.ImageDraw,
    x: float,
    y: int,
    text: str,
    *,
    align: str = "left",
    fill: str = MUTED,
    size: int = 13,
    tracking: int = 1,
) -> float:
    f = font("label", size)
    text = text.upper()
    width = sum(f.getlength(c) for c in text) + tracking * (len(text) - 1)
    x -= {"left": 0, "centre": width / 2, "right": width}[align]
    for c in text:
        d.text((x, y), c, font=f, fill=fill, anchor="ls")
        x += f.getlength(c) + tracking
    return width


def value(
    d: ImageDraw.ImageDraw,
    x: float,
    y: int,
    text: str,
    *,
    align: str = "left",
    size: int = 20,
    fill: str = INK,
) -> None:
    anchor = {"left": "ls", "centre": "ms", "right": "rs"}[align]
    d.text((x, y), text, font=font("value", size), fill=fill, anchor=anchor)


def headline(
    d: ImageDraw.ImageDraw, runs: list[tuple[str, bool]], max_width: float
) -> None:
    fonts = [font("italic" if italic else "serif", 28) for _, italic in runs]
    widths = [f.getlength(text) for (text, _), f in zip(runs, fonts, strict=True)]
    if sum(widths) > max_width:
        raise ValueError(f"headline is {sum(widths):.0f} px, over {max_width:.0f}")
    x: float = M
    for (text, _), f, w in zip(runs, fonts, widths, strict=True):
        d.text((x, 48), text, font=f, fill=INK, anchor="ls")
        x += w


def hairline(
    d: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, fill: str = RULE
) -> None:
    d.line([(x0, y0), (x1, y1)], fill=fill, width=1)


def corner_ticks(
    d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], length: int = 6
) -> None:
    x0, y0, x1, y1 = box
    for x, y, sx, sy in (
        (x0, y0, 1, 1),
        (x1, y0, -1, 1),
        (x0, y1, 1, -1),
        (x1, y1, -1, -1),
    ):
        hairline(d, x, y, x + sx * (length - 1), y)
        hairline(d, x, y, x, y + sy * (length - 1))


def chrome(
    runs: list[tuple[str, bool]], kicker: str, foot_left: str, foot_right: str
) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    kicker_width = caps(d, W - M, 48, kicker, align="right")
    headline(d, runs, W - 3 * M - kicker_width)
    hairline(d, M, 64, W - M, 64)
    hairline(d, M, 432, W - M, 432)
    footer = caps(d, M, 456, foot_left) + caps(d, W - M, 456, foot_right, align="right")
    if footer > W - 3 * M:
        raise ValueError(f"footer is {footer:.0f} px, over {W - 3 * M}")
    return img


def encode_gif(
    frames: Iterable[Image.Image], out: Path, colours: int, max_bytes: int
) -> int:
    graph = (
        f"[0:v]split[a][b];[a]palettegen=max_colors={colours}:stats_mode=full[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
        "-filter_complex", graph, "-loop", "0", "-final_delay", "250", str(out),
    ]  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(frame.convert("RGB").tobytes() for frame in frames)
    subprocess.run(cmd, input=raw, check=True)
    size = out.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{out} is {size:,} B, over the {max_bytes:,} B cap")
    return size


HEADLINE = [("Fine-tuning made the same planner ", False), ("worse", True)]
KICKER = "NEURIPS 2026 · BENTO"
FOOT_RIGHT = "SCORE 11.81 → 8.22 · 3 SEEDS"
LABELS = ("BEFORE · DAGGER CHECKPOINT", "AFTER · RETURN-WEIGHTED ELBO")
LEGEND = "RED = UNLOCKED BEFORE, MISSED HERE"
COLUMNS, COLUMN_W = (M, 432), 384
MAP_X, MAP_Y, MAP_W, MAP_H, RAIL, SCALE = 4, 104, 288, 224, 312, 2
COLOURS, MAX_BYTES = 128, 4_000_000
SWEEP, TOL, MIN_STEPS, REPLAN = 32, 1, 16, 8
POSTER, START, SLOW, FAST, HOLD = 5, 6, 24, 4, 10
TICKER_Y, TICKER_FRAMES, MIN_TICKER = 347, 6, 4
GRID_Y, PER_ROW, PITCH, ICON = (352, 392), 11, 34, 32
VITALS = ("health", "food", "drink", "energy")
VITAL_Y = (208, 236, 264, 292)
GRID = (
    ("COLLECT_WOOD", "wood"),
    ("COLLECT_SAPLING", "sapling"),
    ("PLACE_PLANT", "plant"),
    ("COLLECT_DRINK", "drink"),
    ("EAT_COW", "cow"),
    ("WAKE_UP", "player-sleep"),
    ("PLACE_TABLE", "table"),
    ("MAKE_WOOD_PICKAXE", "wood_pickaxe"),
    ("MAKE_WOOD_SWORD", "wood_sword"),
    ("DEFEAT_ZOMBIE", "zombie"),
    ("EAT_PLANT", "plant-ripe"),
    ("COLLECT_STONE", "stone"),
    ("PLACE_STONE", "stone"),
    ("PLACE_FURNACE", "furnace"),
    ("MAKE_STONE_PICKAXE", "stone_pickaxe"),
    ("MAKE_STONE_SWORD", "stone_sword"),
    ("COLLECT_COAL", "coal"),
    ("DEFEAT_SKELETON", "skeleton"),
    ("COLLECT_IRON", "iron"),
    ("MAKE_IRON_PICKAXE", "iron_pickaxe"),
    ("MAKE_IRON_SWORD", "iron_sword"),
    ("COLLECT_DIAMOND", "diamond"),
)

BEFORE = "checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M"
AFTER = "experiments/rl_finetuning/outputs/gif_baseline_rl/checkpoint_baseline_rl"
CONFIGS = (
    "configs/defaults.yaml",
    "experiments/rl_finetuning/configs/ablations_default.yaml",
    "experiments/rl_finetuning/configs/ablations_final_craftax_classic_gpu_24gb.yaml",
)


class Rollout(NamedTuple):
    actions: np.ndarray  # [L]
    unlocks: np.ndarray  # [L + 1] after each env step; 0 is reset
    health: np.ndarray  # [L + 1]
    ended: np.ndarray  # scalar bool: the episode ended before the step cap


class Trace(NamedTuple):
    pixels: np.ndarray  # [L + 1, 112, 144, 3] map after each env step
    achievements: np.ndarray  # [L + 1, 22] bool, in GRID order
    vitals: np.ndarray  # [L + 1, 4] health, food, drink, energy
    final: Any  # env state after the last step


class Sim(NamedTuple):
    reset: Callable
    cycle: Callable
    cycles: int


class Icons(NamedTuple):
    grid: list[Image.Image]
    masks: list[Image.Image]
    vitals: list[Image.Image]


class Panel(NamedTuple):
    x: int
    trace: Trace
    title: str
    cause: str
    reference: np.ndarray | None  # [L' + 1, 22] the before agent's achievements


class Beat(NamedTuple):
    step: int
    fast: bool


class Event(NamedTuple):
    frame: int
    cell: int
    cells: frozenset[int]


def load_config() -> dict:
    import yaml

    merged: dict = {}
    for path in CONFIGS:
        merged.update(yaml.safe_load((ROOT / path).read_text()) or {})
    return {k.upper(): v for k, v in merged.items()}


def make_env(cfg: dict) -> tuple[Any, Any]:
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name(cfg["ENV_NAME"], auto_reset=False)
    return env, env.default_params


def env_fns(env, env_params) -> tuple[Callable, Callable]:
    import jax

    def reset(seed):
        rng, env_rng = jax.random.split(jax.random.PRNGKey(seed))
        obs, state = env.reset(env_rng, env_params)
        return state, obs, rng

    def step(carry, action):
        state, obs, r = carry
        r, s_rng = jax.random.split(r)
        obs, state, _, done, _ = env.step(s_rng, state, action, env_params)
        return (state, obs, r), (state, done)

    def env_step(state, obs, rng, actions):
        return jax.lax.scan(step, (state, obs, rng), actions)

    return reset, env_step


def make_sim(cfg: dict, env, env_params, apply_eval) -> Sim:
    import jax

    from src.diffusion.sampling import sample_plan
    from src.diffusion.schedules import SCHEDULE_MAP

    if cfg["EVAL_REPLAN"] != REPLAN:
        raise ValueError(f"the replay steps {REPLAN} actions per plan")
    schedule_fn, _ = SCHEDULE_MAP[cfg["DIFFUSION_SCHEDULE"]]
    num_actions = env.action_space(env_params).n
    reset, env_step = env_fns(env, env_params)

    def cycle(params, state, obs, rng):
        rng, p_rng = jax.random.split(rng)
        plan = sample_plan(
            apply_eval,
            params,
            p_rng,
            obs[None],
            num_actions,
            cfg["PLAN_HORIZON"],
            num_steps=cfg["VAL_DIFFUSION_STEPS"],
            schedule_fn=schedule_fn,
            remask_strategy=cfg["REMASK_STRATEGY"],
            eta=cfg["ETA"],
            use_loop=cfg["USE_LOOP"],
            t_on=cfg["T_ON"],
            t_off=cfg["T_OFF"],
            temperature=cfg["TEMPERATURE"],
            top_p=cfg["TOP_P"],
        )
        actions = plan[0, : cfg["EVAL_REPLAN"]]
        carry, (states, dones) = env_step(state, obs, rng, actions)
        return carry, (actions, states, dones)

    cycles = cfg["EVAL_STEPS"] // cfg["EVAL_REPLAN"]
    return Sim(jax.jit(reset), jax.jit(cycle), cycles)


def snapshot(states, n: int) -> tuple[np.ndarray, np.ndarray]:
    unlocks = np.asarray(states.achievements.sum(-1))[:n]
    return unlocks, np.asarray(states.player_health)[:n]


def rollout(sim: Sim, params, seed: int) -> Rollout:
    import jax

    state, obs, rng = sim.reset(seed)
    chunks = [snapshot(jax.tree.map(lambda x: x[None], state), 1)]
    actions, ended = [], False
    for _ in range(sim.cycles):
        (state, obs, rng), (plan, states, dones) = sim.cycle(params, state, obs, rng)
        dones = np.asarray(dones)
        ended = bool(dones.any())
        n = int(dones.argmax()) + 1 if ended else len(dones)
        chunks.append(snapshot(states, n))
        actions.append(np.asarray(plan)[:n])
        if ended:
            break
    unlocks, health = (np.concatenate(c) for c in zip(*chunks, strict=True))
    return Rollout(
        np.concatenate(actions).astype(np.int8),
        unlocks.astype(np.int16),
        health.astype(np.int16),
        np.bool_(ended),
    )


def sweep(sim: Sim, agents: Sequence[Any], seeds: Iterable[int]) -> np.ndarray:
    rows = []
    for seed in seeds:
        row = [seed]
        for params in agents:
            r = rollout(sim, params, seed)
            row += [len(r.actions), int(r.unlocks[-1]), int(r.ended)]
        rows.append(row)
    return np.array(rows)


def medians(table: np.ndarray) -> list[float]:
    unlocks, ended = table[:, 2::3], table[:, 3::3].astype(bool)
    return [float(np.median(unlocks[ended[:, i], i])) for i in range(2)]


def format_table(table: np.ndarray) -> str:
    lines = ["       before          after", "seed   steps unlocks   steps unlocks"]
    for seed, lb, ub, eb, la, ua, ea in table:
        before = f"{lb:>4}{' +'[not eb]} {ub:>7}"
        lines.append(f"{seed:>4}   {before}   {la:>4}{' +'[not ea]} {ua:>7}")
    mb, ma = medians(table)
    lines.append(f"median unlocks over ended episodes: before {mb:g}, after {ma:g}")
    lines.append("+ = still alive at the step cap")
    return "\n".join(lines)


def pick_seed(table: np.ndarray) -> int:
    lengths, unlocks = table[:, 1::3], table[:, 2::3]
    ended = table[:, 3::3].astype(bool)
    typical = np.abs(unlocks - np.array(medians(table))) <= TOL
    ok = (ended & (lengths >= MIN_STEPS) & typical).all(1)
    if not ok.any():
        raise RuntimeError("no seed qualifies:\n" + format_table(table))
    return int(table[ok.argmax(), 0])


def make_rollouts(args) -> tuple[int, tuple[Rollout, ...]]:
    import jax

    from src.planners.model import (
        abstract_params,
        build_model,
        load_checkpoint,
        make_apply_fns,
        restore_latest_params,
    )

    cfg = load_config()
    env, env_params = make_env(cfg)
    obs_dim = env.observation_space(env_params).shape[0]
    net = build_model(cfg, env.action_space(env_params).n)
    key = jax.random.PRNGKey(0)
    before = load_checkpoint(net, key, obs_dim, cfg["PLAN_HORIZON"], str(args.before))
    inner = abstract_params(net, key, obs_dim, cfg["PLAN_HORIZON"])["params"]
    after = {"params": restore_latest_params(str(args.after), inner)[0]}
    sim = make_sim(cfg, env, env_params, make_apply_fns(net)[0])
    print(f"JAX backend: {jax.default_backend()}")

    seed = args.seed
    if seed is None:
        table = sweep(sim, (before, after), range(SWEEP))
        seed = pick_seed(table)
        print(format_table(table))
        print(f"chosen seed: {seed}")
    rollouts = tuple(rollout(sim, p, seed) for p in (before, after))
    lengths = [len(r.actions) for r in rollouts]
    if min(lengths) < MIN_STEPS:
        raise ValueError(f"seed {seed}: episodes of {lengths} steps, under {MIN_STEPS}")
    return seed, rollouts


def sources(args) -> list[str]:
    return [str(args.before.resolve()), str(args.after.resolve())]


def save_rollouts(path: Path, seed: int, rollouts, agents) -> None:
    arrays = {
        f"{agent}_{field}": value
        for agent, r in zip(("before", "after"), rollouts, strict=True)
        for field, value in r._asdict().items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, seed=seed, sources=np.array(agents), **arrays)


def load_rollouts(path: Path, args) -> tuple[int, tuple[Rollout, ...]]:
    with np.load(path) as data:
        seed, cached = int(data["seed"]), data["sources"].tolist()
        if args.seed not in (None, seed) or cached != sources(args):
            raise SystemExit(
                f"{path} holds seed {seed} for {cached}; delete the cache to recompute"
            )
        rollouts = tuple(
            Rollout(*(data[f"{agent}_{field}"] for field in Rollout._fields))
            for agent in ("before", "after")
        )
        return seed, rollouts


@functools.cache
def replay_fns() -> tuple[Callable, Callable, Callable]:
    import jax
    from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_IMG, OBS_DIM
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer

    env, env_params = make_env(load_config())
    reset, env_step = (jax.jit(f) for f in env_fns(env, env_params))
    draw = make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_IMG)
    rows = OBS_DIM[0] * BLOCK_PIXEL_SIZE_IMG

    @jax.jit
    def render(states):
        frozen = jax.vmap(lambda s: s.replace(state_rng=jax.random.PRNGKey(0)))(states)
        return jax.vmap(draw)(frozen)[:, :rows]

    return reset, env_step, render


def replay(seed: int, r: Rollout) -> Trace:
    import jax
    import jax.numpy as jnp
    from craftax.craftax_classic.constants import Achievement

    reset, env_step, render = replay_fns()
    state, obs, rng = reset(seed)
    chunks = [jax.tree.map(lambda x: x[None], state)]
    pixels, dones = [render(chunks[0])], []
    padded = np.pad(r.actions, (0, -len(r.actions) % REPLAN)).reshape(-1, REPLAN)
    for actions in padded.astype(np.int32):
        rng, _ = jax.random.split(rng)
        (state, obs, rng), (states, done) = env_step(state, obs, rng, actions)
        chunks.append(states)
        pixels.append(render(states))
        dones.append(np.asarray(done))
    done = np.concatenate(dones)
    n = int(done.argmax()) + 1 if done.any() else len(done)
    states = jax.tree.map(lambda *xs: jnp.concatenate(xs)[: n + 1], *chunks)
    unlocks = np.asarray(states.achievements.sum(-1)).astype(np.int16)
    health = np.asarray(states.player_health).astype(np.int16)
    same = np.array_equal(unlocks, r.unlocks) and np.array_equal(health, r.health)
    if not same or n != len(r.actions) or bool(done.any()) != bool(r.ended):
        raise RuntimeError(f"the seed {seed} replay diverges from the cached rollout")
    order = [Achievement[name].value for name, _ in GRID]
    vitals = [getattr(states, f"player_{name}") for name in VITALS]
    return Trace(
        np.concatenate(pixels)[: n + 1].clip(0, 255).astype(np.uint8),
        np.asarray(states.achievements)[:, order],
        np.stack(vitals, -1).astype(np.int16),
        jax.tree.map(lambda x: x[n], states),
    )


def attackers(final) -> int:
    zombies = final.zombies
    offset = np.asarray(zombies.position) - np.asarray(final.player_position)
    adjacent = np.abs(offset).sum(-1) == 1
    struck = np.asarray(zombies.attack_cooldown) == 5
    return int((np.asarray(zombies.mask) & struck & adjacent).sum())


def cause(final) -> str:
    night = " AT NIGHT" if float(final.light_level) < 0.5 else ""
    n = attackers(final)
    if n == 0:
        return "HEALTH RAN OUT" + night
    return ("KILLED BY A ZOMBIE" if n == 1 else "KILLED BY ZOMBIES") + night


def card(r: Rollout, t: Trace) -> tuple[str, str]:
    if not r.ended:
        return f"Alive at step {len(r.actions)}", "THE STEP CAP ENDED THE EPISODE"
    return f"Died at step {len(r.actions)}", cause(t.final)


def upscale(rgb: np.ndarray, k: int) -> Image.Image:
    black = (rgb == 0).all(-1, keepdims=True)
    bg = np.array(ImageColor.getrgb(BG), dtype=np.uint8)
    img = Image.fromarray(np.where(black, bg, rgb).astype(np.uint8))
    return img.resize((img.width * k, img.height * k), Image.Resampling.NEAREST)


def load_icons() -> Icons:
    import craftax

    assets = Path(craftax.__file__).parent / "craftax_classic" / "assets"
    grid = [
        Image.open(assets / f"{name}.png")
        .convert("RGBA")
        .resize((ICON, ICON), Image.Resampling.NEAREST)
        for _, name in GRID
    ]
    masks = [icon.getchannel("A").point(lambda a: 255 * (a > 127)) for icon in grid]
    vitals = [Image.open(assets / f"{name}.png").convert("RGBA") for name in VITALS]
    return Icons(grid, masks, vitals)


def timeline(ends: Sequence[int]) -> list[Beat]:
    steps = [0] * START + list(range(1, min(SLOW, *ends) + 1))
    beats = [Beat(s, False) for s in steps]
    for end in sorted(set(ends)):
        start = beats[-1].step
        beats += [Beat(s, True) for s in range(start + FAST, end, FAST)]
        if end > start:
            beats.append(Beat(end, end - beats[-1].step == FAST))
        beats += [Beat(end, False)] * HOLD
    return beats


def newest(achievements: np.ndarray, step: int, new: np.ndarray) -> int:
    first = achievements[: step + 1, new].argmax(0)
    return int(max(zip(first, new, strict=True))[1])


def unlock_events(achievements: np.ndarray, steps: Sequence[int]) -> list[Event]:
    event = Event(-TICKER_FRAMES, 0, frozenset())
    events = []
    for f, (prev, step) in enumerate(zip([0, *steps], steps, strict=False)):
        new = np.flatnonzero(achievements[step] & ~achievements[prev])
        if new.size and f - event.frame < MIN_TICKER:
            event = event._replace(cells=event.cells | set(new.tolist()))
        elif new.size:
            event = Event(f, newest(achievements, step, new), frozenset(new.tolist()))
        events.append(event)
    return events


def unlock_text(event: Event) -> str:
    more = f" · +{len(event.cells) - 1} MORE" if len(event.cells) > 1 else ""
    return f"+ {GRID[event.cell][0].replace('_', ' ')}{more}"


def base_image(seed: int, icons: Icons) -> Image.Image:
    foot_left = f"CRAFTAX CLASSIC · SAME WORLD · SEED {seed}"
    img = chrome(HEADLINE, KICKER, foot_left, FOOT_RIGHT)
    d = ImageDraw.Draw(img)
    hairline(d, 420, 80, 420, GRID_Y[-1] + ICON - 1)
    for x, label in zip(COLUMNS, LABELS, strict=True):
        caps(d, x, 92, label)
        corner_ticks(d, (x, MAP_Y - 4, x + MAP_W + 7, MAP_Y + MAP_H + 3))
        caps(d, x + RAIL, 116, "STEP")
        caps(d, x + RAIL, 168, "UNLOCKED")
        for y, icon in zip(VITAL_Y, icons.vitals, strict=True):
            img.paste(icon, (x + RAIL, y), icon)
    return img


def end_card(
    img: Image.Image, x: int, pixels: np.ndarray, title: str, why: str
) -> None:
    serif = font("serif", 28)
    if serif.getlength(title) > MAP_W:
        raise ValueError(f"{title!r} is wider than the {MAP_W} px map")
    blank = Image.new("RGB", (MAP_W, MAP_H), BG)
    img.paste(Image.blend(upscale(pixels, SCALE), blank, 0.7), (x + MAP_X, MAP_Y))
    d = ImageDraw.Draw(img)
    cx, cy = x + MAP_X + MAP_W // 2, MAP_Y + MAP_H * 3 // 4
    d.text((cx, cy + 4), title, font=serif, fill=INK, anchor="ms")
    caps(d, cx, cy + 28, why, align="centre")


def draw_grid(
    img: Image.Image,
    icons: Icons,
    x: int,
    unlocked: np.ndarray,
    marked: frozenset[int],
    red: frozenset[int],
) -> None:
    d = ImageDraw.Draw(img)
    for i, (icon, mask) in enumerate(zip(icons.grid, icons.masks, strict=True)):
        row, col = divmod(i, PER_ROW)
        x0, y0 = x + col * PITCH, GRID_Y[row]
        if unlocked[i]:
            img.paste(icon, (x0, y0), icon)
        else:
            fill = HARM if i in red else RULE
            img.paste(fill, (x0, y0, x0 + ICON, y0 + ICON), mask)
        if i in marked:
            d.rectangle((x0, y0 + ICON, x0 + ICON - 1, y0 + ICON + 1), fill=INK)


def missed(panel: Panel, clock: int) -> frozenset[int]:
    last = len(panel.trace.pixels) - 1
    if panel.reference is None or clock < last:
        return frozenset()
    seen = panel.reference[min(clock, len(panel.reference) - 1)]
    return frozenset(np.flatnonzero(seen & ~panel.trace.achievements[last]).tolist())


def ticker(beat: Beat, f: int, event: Event, red: frozenset[int]) -> tuple[str, str]:
    if beat.step == 0:
        return "SAME WORLD · SAME START", MUTED
    if red:
        return LEGEND, MUTED
    if f - event.frame < TICKER_FRAMES:
        return unlock_text(event), INK
    return "", MUTED


def draw_panel(
    img: Image.Image, icons: Icons, panel: Panel, beat: Beat, f: int, event: Event
) -> None:
    d = ImageDraw.Draw(img)
    x, t = panel.x, panel.trace
    last = len(t.pixels) - 1
    step = min(beat.step, last)
    if step == last:
        end_card(img, x, t.pixels[last], panel.title, panel.cause)
    else:
        img.paste(upscale(t.pixels[step], SCALE), (x + MAP_X, MAP_Y))
    if beat.fast and step < last:
        caps(d, x + COLUMN_W, 116, f"×{FAST}", align="right")
    value(d, x + RAIL, 140, f"{step:04d}")
    value(d, x + RAIL, 192, f"{int(t.achievements[step].sum()):02d}")
    for y, v in zip(VITAL_Y, t.vitals[step], strict=True):
        value(d, x + RAIL + 24, y + 15, str(max(int(v), 0)))
    red = missed(panel, beat.step)
    text, fill = ticker(beat, f, event, red)
    if text:
        caps(d, x, TICKER_Y, text, fill=fill)
    marked = event.cells if f - event.frame < TICKER_FRAMES else frozenset()
    draw_grid(img, icons, x, t.achievements[step], marked, red)


def frames(seed: int, panels: Sequence[Panel]) -> Iterator[Image.Image]:
    icons = load_icons()
    base = base_image(seed, icons)
    beats = timeline([len(p.trace.pixels) - 1 for p in panels])
    events = [
        unlock_events(
            p.trace.achievements,
            [min(b.step, len(p.trace.pixels) - 1) for b in beats],
        )
        for p in panels
    ]

    def draw(f: int) -> Image.Image:
        img = base.copy()
        for panel, panel_events in zip(panels, events, strict=True):
            draw_panel(img, icons, panel, beats[f], f, panel_events[f])
        return img

    poster = draw(len(beats) - 1)
    yield from [poster] * POSTER
    yield from (draw(f) for f in range(len(beats)))
    yield poster


def names(cells: Iterable[int]) -> list[str]:
    return [GRID[i][0] for i in sorted(cells)]


def make_panels(seed: int, rollouts: Sequence[Rollout]) -> list[Panel]:
    traces = [replay(seed, r) for r in rollouts]
    print(f"replay check passed: seed {seed}, unlock and health traces match")
    before, after = (set(np.flatnonzero(t.achievements[-1]).tolist()) for t in traces)
    print(f"after missed: {names(before - after)}")
    print(f"after only: {names(after - before)}")
    panels = []
    agents = ("before", "after")
    references = (None, traces[0].achievements)
    for x, agent, r, t, reference in zip(
        COLUMNS, agents, rollouts, traces, references, strict=True
    ):
        title, why = card(r, t)
        health = f"health {int(r.health[-2])} -> {int(r.health[-1])}"
        print(f"{agent}: {title}, {why} (attackers: {attackers(t.final)}, {health})")
        panels.append(Panel(x, t, title, why, reference))
    return panels


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--before", type=Path, default=ROOT / BEFORE)
    p.add_argument("--after", type=Path, default=ROOT / AFTER)
    p.add_argument(
        "--seed",
        type=int,
        help=f"world seed to render instead of sweeping 0..{SWEEP - 1} (new cache)",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "results/gif/craftax-rollout.npz",
        help="rollout cache: read if present (delete it to recompute), else written",
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "results/gif/craftax-before-after.gif"
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cache.exists():
        seed, rollouts = load_rollouts(args.cache, args)
        print(f"Loaded the seed {seed} rollouts from {args.cache}")
    else:
        seed, rollouts = make_rollouts(args)
        save_rollouts(args.cache, seed, rollouts, sources(args))
        print(f"Cached the rollouts in {args.cache}")
    for agent, r in zip(("before", "after"), rollouts, strict=True):
        print(f"{agent}: {r.unlocks[-1]} unlocks in {len(r.actions)} steps")
    panels = make_panels(seed, rollouts)
    size = encode_gif(frames(seed, panels), args.out, COLOURS, MAX_BYTES)
    print(f"Wrote {args.out} ({size:,} B)")


if __name__ == "__main__":
    main()
