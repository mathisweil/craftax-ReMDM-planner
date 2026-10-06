#!/usr/bin/env python3
"""Render the README animation: one Craftax Classic world, before and after fine-tuning.

The DAgger checkpoint and its return-weighted ELBO fine-tune (``baseline_rl``)
play the same world under the key chain of ``build_eval_fn``: the raw env, one
environment, a fresh 50-step plan every 8 steps, up to EVAL_STEPS. The world
is the lowest seed in 0..31 where both episodes end, last at least 16 steps
and land within one unlock of their agent's median; the sweep is printed.
Rollouts are cached, so a layout change re-renders without importing JAX.

Usage:
  env XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/render_rollout_gif.py \
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
CONTENT = (24, 80, 816, 416)
TOKEN_FILLS = (RULE, MUTED, ACCENT, INK, FAINT)
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
    right: bool = False,
    fill: str = MUTED,
    size: int = 13,
    tracking: int = 1,
) -> float:
    f = font("label", size)
    text = text.upper()
    width = sum(f.getlength(c) for c in text) + tracking * (len(text) - 1)
    if right:
        x -= width
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
    right: bool = False,
    size: int = 20,
    fill: str = INK,
) -> None:
    anchor = "rs" if right else "ls"
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
    kicker_width = caps(d, W - M, 48, kicker, right=True)
    headline(d, runs, W - 3 * M - kicker_width)
    hairline(d, M, 64, W - M, 64)
    hairline(d, M, 432, W - M, 432)
    footer = caps(d, M, 456, foot_left) + caps(d, W - M, 456, foot_right, right=True)
    if footer > W - 3 * M:
        raise ValueError(f"footer is {footer:.0f} px, over {W - 3 * M}")
    return img


def upscale(rgb: np.ndarray, k: int) -> Image.Image:
    black = (rgb == 0).all(-1, keepdims=True)
    bg = np.array(ImageColor.getrgb(BG), dtype=np.uint8)
    img = Image.fromarray(np.where(black, bg, rgb).astype(np.uint8))
    return img.resize((img.width * k, img.height * k), Image.Resampling.NEAREST)


def token_strip(
    d: ImageDraw.ImageDraw,
    x: int,
    y: int,
    states: Sequence[int],
    pitch: int = 12,
    cell: tuple[int, int] = (10, 20),
) -> None:
    for i, s in enumerate(states):
        x0 = x + i * pitch
        d.rectangle((x0, y, x0 + cell[0] - 1, y + cell[1] - 1), fill=TOKEN_FILLS[s])


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
COLUMNS = (M, 432)
MAP_Y, RAIL, SCALE, STRIP_W = 104, 312, 2, 31 * 12 + 10
COLOURS, MAX_BYTES = 128, 4_000_000
SWEEP, TOL, MIN_STEPS, REPLAN, FF_FRAMES = 32, 1, 16, 8, 40

BEFORE = "checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M"
AFTER = "experiments/rl_finetuning/outputs/gif_baseline_rl/checkpoint_baseline_rl"
CONFIGS = (
    "configs/defaults.yaml",
    "experiments/rl_finetuning/configs/ablations_default.yaml",
    "experiments/rl_finetuning/configs/ablations_final_craftax_classic_gpu_24gb.yaml",
)


class Rollout(NamedTuple):
    pixels: np.ndarray  # [L + 1, 112, 144, 3] map after each env step; 0 is reset
    unlocks: np.ndarray  # [L + 1]
    health: np.ndarray  # [L + 1]
    actions: np.ndarray  # [L]
    plans: np.ndarray  # [C, 32] one plan per cycle
    paths: np.ndarray  # [C, T, 32] z after each denoising step
    ended: np.ndarray  # scalar bool: the episode ended before the step cap


class Sim(NamedTuple):
    reset: Callable
    cycle: Callable
    render: Callable
    cycles: int
    names: tuple[str, ...]


class Shot(NamedTuple):
    step: int
    tokens: np.ndarray
    status: str


def load_config() -> dict:
    import yaml

    merged: dict = {}
    for path in CONFIGS:
        merged.update(yaml.safe_load((ROOT / path).read_text()) or {})
    return {k.upper(): v for k, v in merged.items()}


def make_sim(cfg: dict, env, env_params, apply_eval) -> Sim:
    import jax
    import jax.numpy as jnp
    from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_IMG, OBS_DIM, Action
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer

    from src.diffusion.sampling import sample_plan
    from src.diffusion.schedules import SCHEDULE_MAP

    if cfg["EVAL_REPLAN"] != REPLAN:
        raise ValueError(f"the layout draws {REPLAN}-step plans")
    schedule_fn, _ = SCHEDULE_MAP[cfg["DIFFUSION_SCHEDULE"]]
    num_actions = env.action_space(env_params).n
    draw = make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_IMG)
    map_rows = OBS_DIM[0] * BLOCK_PIXEL_SIZE_IMG

    def reset(seed):
        rng, env_rng = jax.random.split(jax.random.PRNGKey(seed))
        obs, state = env.reset(env_rng, env_params)
        return state, obs, rng

    def cycle(params, state, obs, rng):
        rng, p_rng = jax.random.split(rng)
        plan, path = sample_plan(
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
            return_path=True,
        )

        def step(carry, i):
            st, ob, r = carry
            r, s_rng = jax.random.split(r)
            ob, st, _, done, _ = env.step(s_rng, st, plan[0, i], env_params)
            return (st, ob, r), (st, done)

        carry, (states, dones) = jax.lax.scan(
            step, (state, obs, rng), jnp.arange(cfg["EVAL_REPLAN"])
        )
        return carry, (plan[0], path[:, 0], states, dones)

    def render(states):
        frozen = jax.vmap(lambda s: s.replace(state_rng=jax.random.PRNGKey(0)))(states)
        return jax.vmap(draw)(frozen)[:, :map_rows]

    names = tuple(a.name.replace("_", " ") for a in Action)
    cycles = cfg["EVAL_STEPS"] // cfg["EVAL_REPLAN"]
    return Sim(jax.jit(reset), jax.jit(cycle), jax.jit(render), cycles, names)


def snapshot(sim: Sim, states, n: int, render: bool) -> tuple[np.ndarray, ...]:
    unlocks = np.asarray(states.achievements.sum(-1))[:n]
    health = np.asarray(states.player_health)[:n]
    if not render:
        return np.zeros((0, 0, 0, 3), np.uint8), unlocks, health
    pixels = np.asarray(sim.render(states)[:n]).clip(0, 255).astype(np.uint8)
    return pixels, unlocks, health


def rollout(sim: Sim, params, seed: int, render: bool) -> Rollout:
    import jax

    state, obs, rng = sim.reset(seed)
    chunks = [snapshot(sim, jax.tree.map(lambda x: x[None], state), 1, render)]
    plans, paths, actions, ended = [], [], [], False
    for _ in range(sim.cycles):
        (state, obs, rng), (plan, path, states, dones) = sim.cycle(
            params, state, obs, rng
        )
        dones = np.asarray(dones)
        ended = bool(dones.any())
        n = int(dones.argmax()) + 1 if ended else len(dones)
        chunks.append(snapshot(sim, states, n, render))
        plans.append(np.asarray(plan))
        paths.append(np.asarray(path))
        actions.append(np.asarray(plan)[:n])
        if ended:
            break
    pixels, unlocks, health = (np.concatenate(c) for c in zip(*chunks, strict=True))
    return Rollout(
        pixels,
        unlocks.astype(np.int16),
        health.astype(np.int16),
        np.concatenate(actions).astype(np.int8),
        np.stack(plans).astype(np.int8),
        np.stack(paths).astype(np.int8),
        np.bool_(ended),
    )


def sweep(sim: Sim, agents: Sequence[Any], seeds: Iterable[int]) -> np.ndarray:
    rows = []
    for seed in seeds:
        row = [seed]
        for params in agents:
            r = rollout(sim, params, seed, render=False)
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


def make_rollouts(args) -> tuple[int, tuple[Rollout, Rollout], tuple[str, ...]]:
    import jax
    from craftax.craftax_env import make_craftax_env_from_name

    from src.planners.model import (
        abstract_params,
        build_model,
        load_checkpoint,
        make_apply_fns,
        restore_latest_params,
    )

    cfg = load_config()
    env = make_craftax_env_from_name(cfg["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
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
    rollouts = tuple(rollout(sim, p, seed, render=True) for p in (before, after))
    lengths = [len(r.actions) for r in rollouts]
    if min(lengths) < MIN_STEPS:
        raise ValueError(f"seed {seed}: episodes of {lengths} steps, under {MIN_STEPS}")
    return seed, rollouts, sim.names


def sources(args) -> list[str]:
    return [str(args.before.resolve()), str(args.after.resolve())]


def save_rollouts(path: Path, seed: int, rollouts, names, agents) -> None:
    arrays = {
        f"{agent}_{field}": value
        for agent, r in zip(("before", "after"), rollouts, strict=True)
        for field, value in r._asdict().items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"seed": seed, "names": np.array(names), "sources": np.array(agents)}
    np.savez_compressed(path, **meta, **arrays)


def load_rollouts(path: Path, args) -> tuple[int, tuple[Rollout, ...], tuple[str, ...]]:
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
        return seed, rollouts, tuple(str(n) for n in data["names"])


def schedule(cycles: int, steps: int) -> list[tuple]:
    ks = [round(steps * i / 10) for i in range(1, 11)]
    events = [
        ("denoise", 0, k, prev) for prev, k in zip([0, *ks[:-1]], ks, strict=True)
    ]
    events += [("exec", 0, j) for j in range(REPLAN)]
    half = steps // 2
    events += [("denoise", 1, half, 0), ("denoise", 1, steps, half)]
    events += [("exec", 1, j) for j in range(REPLAN)]
    g = max(1, -(-(cycles - 2) // FF_FRAMES))
    events += [("ff", c, g) for c in range(cycles - 1 - g, 1, -g)][::-1]
    return [("end",)] * 5 + events + [("end",)]


def exec_tokens(done: int, now: int | None = None) -> np.ndarray:
    tokens = np.ones(32, dtype=np.int8)
    tokens[:done] = 4
    if now is not None:
        tokens[now] = 3
    return tokens


def denoise_tokens(r: Rollout, cycle: int, k: int, prev: int, mask: int) -> np.ndarray:
    steps = r.paths.shape[1]
    z = r.plans[cycle] if k == steps else r.paths[cycle, k - 1]
    shown = np.full_like(z, mask) if prev == 0 else r.paths[cycle, prev - 1]
    return np.where(z == mask, 0, np.where(z != shown, 2, 1))


def end_shot(r: Rollout) -> Shot:
    last = len(r.actions)
    status = "EPISODE OVER" if r.ended else "STEP CAP"
    return Shot(last, exec_tokens((last - 1) % REPLAN + 1), status)


def shot(r: Rollout, event: tuple, names: tuple[str, ...]) -> Shot:
    kind, *rest = event
    if kind == "denoise":
        c, k, prev = rest
        steps = r.paths.shape[1]
        tokens = denoise_tokens(r, c, k, prev, len(names))
        return Shot(c * REPLAN, tokens, f"DENOISE {k:02d}/{steps}")
    if kind == "exec":
        c, j = rest
        step = c * REPLAN + j + 1
        if step > len(r.actions):
            return end_shot(r)
        action = names[r.actions[step - 1]]
        return Shot(step, exec_tokens(j, j), f"NOW · {action}")
    if kind == "ff":
        c, g = rest
        step = (c + 1) * REPLAN
        if step >= len(r.actions):
            return end_shot(r)
        return Shot(step, exec_tokens(REPLAN), f"FAST-FORWARD ×{REPLAN * g}")
    return end_shot(r)


def base_image(seed: int) -> Image.Image:
    foot_left = f"CRAFTAX CLASSIC · SAME WORLD · SEED {seed}"
    img = chrome(HEADLINE, KICKER, foot_left, FOOT_RIGHT)
    d = ImageDraw.Draw(img)
    hairline(d, 420, 80, 420, 415)
    for x, label in zip(COLUMNS, LABELS, strict=True):
        caps(d, x, 92, label)
        corner_ticks(d, (x, MAP_Y - 4, x + 295, MAP_Y + 227))
        for y, name in zip((116, 172, 228), ("STEP", "UNLOCKS", "HEALTH"), strict=True):
            caps(d, x + RAIL, y, name)
        caps(d, x, 352, "PLAN · 32 TOKENS")
        span = REPLAN * 12 - 3
        hairline(d, x, 392, x + span, 392, FAINT)
        hairline(d, x, 388, x, 392, FAINT)
        hairline(d, x + span, 388, x + span, 392, FAINT)
        caps(d, x, 408, f"EXECUTE {REPLAN}")
    return img


def draw_panel(img: Image.Image, x: int, r: Rollout, s: Shot) -> None:
    d = ImageDraw.Draw(img)
    img.paste(upscale(r.pixels[s.step], SCALE), (x + 4, MAP_Y))
    value(d, x + RAIL, 140, f"{s.step:04d}")
    value(d, x + RAIL, 196, f"{r.unlocks[s.step]:02d}")
    value(d, x + RAIL, 252, str(max(int(r.health[s.step]), 0)))
    caps(d, x + STRIP_W, 352, s.status, right=True)
    token_strip(d, x, 364, s.tokens)


def frames(seed: int, rollouts, names) -> Iterator[Image.Image]:
    base = base_image(seed)
    cycles = max(len(r.plans) for r in rollouts)
    for event in schedule(cycles, rollouts[0].paths.shape[1]):
        img = base.copy()
        for x, r in zip(COLUMNS, rollouts, strict=True):
            draw_panel(img, x, r, shot(r, event, names))
        yield img


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
        seed, rollouts, names = load_rollouts(args.cache, args)
        print(f"Loaded the seed {seed} rollouts from {args.cache}")
    else:
        seed, rollouts, names = make_rollouts(args)
        save_rollouts(args.cache, seed, rollouts, names, sources(args))
        print(f"Cached the rollouts in {args.cache}")
    for agent, r in zip(("before", "after"), rollouts, strict=True):
        print(f"{agent}: {r.unlocks[-1]} unlocks in {len(r.actions)} steps")
    size = encode_gif(frames(seed, rollouts, names), args.out, COLOURS, MAX_BYTES)
    print(f"Wrote {args.out} ({size:,} B)")


if __name__ == "__main__":
    main()
