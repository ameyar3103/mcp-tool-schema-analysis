"""Accuracy and twin-confusion against the schema budget K, from one salted run."""

from __future__ import annotations

import sys

from plot_pareto import ASSETS, collect, metrics

# Budget in schemas per arm. index-only sends none, full-catalog sends the padded catalog.
BUDGET = {"index-only": 0, "static-16": 16, "static-32": 32, "static-64": 64, "full-catalog": 300}


def twin_rate(results) -> float:
    scored = [r for r in results if not r.error]
    return sum(r.twin for r in scored) / (len(scored) or 1)


def main(salt: str, out: str = "frontier.png") -> None:
    raw = collect(salt)
    if not raw:
        raise SystemExit(f"no results for salt {salt}")
    known = ((BUDGET[n], n, metrics(rs), twin_rate(rs)) for n, rs in raw.items() if n in BUDGET)
    curve = sorted(known)
    ks = [k for k, *_ in curve]
    # Categorical x: the budgets tested are five chosen points, not samples of a continuum.
    xs = list(range(len(curve)))
    at64 = ks.index(64)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)

    strict = [m["accuracy"] for *_, m, _ in curve]
    lo = [m["accuracy"] - m["lo"] for *_, m, _ in curve]
    hi = [m["hi"] - m["accuracy"] for *_, m, _ in curve]
    top.errorbar(xs, strict, yerr=[lo, hi], fmt="o-", capsize=3, lw=1.5, label="strict")
    top.plot(xs, [m["lenient"] for *_, m, _ in curve], "s--", alpha=0.6, label="lenient")
    # The adaptive arm shares static-64's budget, so it plots as a second point at K=64.
    if (adaptive := raw.get("adaptive-64")) is not None:
        top.plot([at64], [metrics(adaptive)["accuracy"]], "x", ms=11, mew=2, label="adaptive-64")
    top.axvline(xs[max(range(len(curve)), key=lambda i: curve[i][2]["accuracy"])], ls=":", alpha=0.4)
    top.set_ylabel("tool-selection accuracy")
    top.set_title("Sending 21% of the catalog beats sending all of it (bars: 95% Wilson CI)")
    top.legend(fontsize=9)
    top.grid(alpha=0.3)

    bot.plot(xs, [t for *_, t in curve], "o-", color="tab:red", lw=1.5)
    if adaptive is not None:
        bot.plot([at64], [twin_rate(adaptive)], "x", ms=11, mew=2, color="tab:red")
    bot.set_xlabel("schemas sent per turn (K)")
    bot.set_ylabel("twin confusion rate")
    bot.set_title("The gain is twin disambiguation, and it reverses past the peak")
    bot.grid(alpha=0.3)
    for ax in (top, bot):
        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in ks])

    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / out, dpi=150)
    print(f"wrote {ASSETS / out}")


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit("usage: plot_frontier.py <salt> [out.png]")
    main(*sys.argv[1:])
