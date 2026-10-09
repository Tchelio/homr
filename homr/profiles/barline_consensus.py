from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol


@dataclass
class _BarlineCandidate:
    voice: int
    symbol: EncodedSymbol
    x: float
    unit: float


def _is_barline(symbol: EncodedSymbol) -> bool:
    return "barline" in symbol.rhythm or "repeat" in symbol.rhythm


def _split_systems(voice: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
    systems: list[list[EncodedSymbol]] = [[]]
    for symbol in voice:
        if symbol.rhythm == "newline":
            systems.append([])
        else:
            systems[-1].append(symbol)
    if systems and not systems[-1]:
        systems.pop()
    return systems


def _candidate_x(symbol: EncodedSymbol) -> float | None:
    if symbol.barline_geometry_page_x is not None:
        return symbol.barline_geometry_page_x
    if symbol.page_coordinates is not None:
        return symbol.page_coordinates[0]
    return None


def _cluster(candidates: list[_BarlineCandidate], tolerance: float) -> list[list[_BarlineCandidate]]:
    clusters: list[list[_BarlineCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: item.x):
        best: list[_BarlineCandidate] | None = None
        best_distance = float("inf")
        for cluster in clusters:
            if candidate.voice in {item.voice for item in cluster}:
                continue
            center = statistics.median(item.x for item in cluster)
            distance = abs(candidate.x - center)
            if distance <= tolerance and distance < best_distance:
                best = cluster
                best_distance = distance
        if best is None:
            clusters.append([candidate])
        else:
            best.append(candidate)
    return clusters


def _canonical_rhythm(cluster: list[_BarlineCandidate]) -> str:
    return Counter(item.symbol.rhythm for item in cluster).most_common(1)[0][0]


def _insert_by_x(symbols: list[EncodedSymbol], symbol: EncodedSymbol, x: float) -> None:
    for index, current in enumerate(symbols):
        if current.page_coordinates is not None and current.page_coordinates[0] > x:
            symbols.insert(index, symbol)
            return
    symbols.append(symbol)


def reconcile_system_barlines(
    voices: list[list[EncodedSymbol]], profile_name: str
) -> tuple[int, int]:
    """Make multi-staff systems share one majority-supported barline topology.

    TrOMR barline tokens provide semantic candidates, while Stage-1 vertical boxes provide
    the image geometry stored on those tokens. With four independent players, a 3/4 majority
    is enough to confirm a system barline; isolated 1/4 tokens are discarded. Missing tokens
    at confirmed positions are synthesized so all voices split measures at the same places.
    """
    if len(voices) < 2:
        return 0, 0

    systems_by_voice = [_split_systems(voice) for voice in voices]
    system_counts = {len(systems) for systems in systems_by_voice}
    if len(system_counts) != 1:
        eprint(f"PROFILE {profile_name}: barline consensus skipped; system counts disagree")
        return 0, 0

    removed = 0
    inserted = 0
    number_of_voices = len(voices)
    minimum_support = max(2, math.ceil(number_of_voices * 0.75))

    for system_index in range(next(iter(system_counts))):
        candidates: list[_BarlineCandidate] = []
        units: list[float] = []
        for voice_index, systems in enumerate(systems_by_voice):
            for symbol in systems[system_index]:
                if not _is_barline(symbol):
                    continue
                x = _candidate_x(symbol)
                if x is None:
                    continue
                unit = symbol.barline_geometry_unit_size
                if unit > 0:
                    units.append(unit)
                candidates.append(_BarlineCandidate(voice_index, symbol, x, unit))

        if not candidates:
            continue
        median_unit = statistics.median(units) if units else 8.0
        clusters = _cluster(candidates, tolerance=9 * median_unit)
        confirmed = [
            cluster
            for cluster in clusters
            if len({candidate.voice for candidate in cluster}) >= minimum_support
        ]
        confirmed_ids = {id(candidate.symbol) for cluster in confirmed for candidate in cluster}

        # Preserve why a fragment became suspicious before deleting the false boundary.
        # A later recovery pass may safely reinterpret only the low-confidence symbols that
        # originally lived immediately before a suppressed isolated barline.
        for cluster in clusters:
            if cluster in confirmed:
                continue
            for candidate in cluster:
                system = systems_by_voice[candidate.voice][system_index]
                boundary_index = next(
                    (i for i, symbol in enumerate(system) if symbol is candidate.symbol), None
                )
                if boundary_index is None:
                    continue
                for symbol in reversed(system[:boundary_index]):
                    if _is_barline(symbol):
                        break
                    if symbol.structural_visual_mismatch:
                        symbol.suppressed_barline_after_page_x = candidate.x

        for voice_index, systems in enumerate(systems_by_voice):
            system = systems[system_index]
            before = len(system)
            system[:] = [
                symbol
                for symbol in system
                if not _is_barline(symbol) or id(symbol) in confirmed_ids
            ]
            removed += before - len(system)

            for cluster in confirmed:
                if any(candidate.voice == voice_index for candidate in cluster):
                    continue
                x = statistics.median(candidate.x for candidate in cluster)
                rhythm = _canonical_rhythm(cluster)
                new_barline = EncodedSymbol(rhythm)
                new_barline.page_coordinates = (float(x), 0.0)
                new_barline.barline_geometry_page_x = float(x)
                new_barline.barline_geometry_unit_size = float(median_unit)
                _insert_by_x(system, new_barline, x)
                inserted += 1

        for cluster in clusters:
            support = len({candidate.voice for candidate in cluster})
            x = statistics.median(candidate.x for candidate in cluster)
            status = "CONFIRMED" if cluster in confirmed else "SUPPRESSED"
            eprint(
                f"PROFILE {profile_name} system {system_index + 1} BARLINE_{status}: "
                f"x={x:.1f}, support={support}/{number_of_voices}"
            )

    for voice_index, voice in enumerate(voices):
        rebuilt: list[EncodedSymbol] = []
        for system in systems_by_voice[voice_index]:
            rebuilt.extend(system)
            rebuilt.append(EncodedSymbol("newline"))
        voice[:] = rebuilt

    eprint(
        f"PROFILE {profile_name}: system barline consensus removed={removed}, inserted={inserted}"
    )
    return removed, inserted
