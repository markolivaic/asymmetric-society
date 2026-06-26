"""Game factory: build the right :class:`BaseGame` from a :class:`GameConfig`."""

from __future__ import annotations

import random

from asymmetric_society.config import GameConfig
from asymmetric_society.games.base import BaseGame
from asymmetric_society.games.public_goods import PublicGoodsGame
from asymmetric_society.games.trust_game import TrustGame

_REGISTRY: dict[str, type[BaseGame]] = {
    "public_goods": PublicGoodsGame,
    "trust_game": TrustGame,
}


def create_game(game: GameConfig, *, rng: random.Random) -> BaseGame:
    """Instantiate the game named by ``game.name`` from its params.

    Args:
        game: Validated game configuration; ``params`` supplies the constructor
            keyword arguments (n, endowment, multiplier, rounds, communication).
        rng: Seeded RNG passed to the game for reproducible randomization.

    Returns:
        A ready-to-use :class:`BaseGame` subclass instance.

    Raises:
        ValueError: If ``game.name`` is not a known game.
    """
    try:
        cls = _REGISTRY[game.name]
    except KeyError as exc:
        raise ValueError(f"Unknown game: {game.name!r}") from exc
    return cls(rng=rng, **game.params)  # type: ignore[arg-type]
