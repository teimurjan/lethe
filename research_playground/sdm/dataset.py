"""Synthetic episodic dataset designed to stress near-duplicate discrimination.

Four event families (trip, meeting, purchase, preference). Each base event has
5 "sibling" variants that share most attributes but differ in 1-2 — the memory
system must distinguish them when given noisy/partial cues.

Query noise modes:
- partial: drop several attributes
- paraphrase: swap synonyms
- fragment: keep only 1-2 attributes
- noisy: add unrelated tokens
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

NoiseMode = Literal["partial", "paraphrase", "fragment", "noisy"]


@dataclass
class Event:
    event_id: int
    family: str
    text: str
    attrs: dict[str, str]
    # Siblings share this key (same base template, different attribute values).
    family_key: str = ""


@dataclass
class Query:
    text: str
    target_id: int
    sibling_ids: list[int]  # includes target_id
    noise_mode: NoiseMode
    family: str = ""


@dataclass
class Dataset:
    events: list[Event]
    queries: list[Query]
    # event_id → list of sibling event_ids (including self)
    siblings_map: dict[int, list[int]] = field(default_factory=dict)


# ---------- templates ----------

CITIES = ["New York", "Paris", "Tokyo", "Berlin", "London", "Rome", "Sydney",
          "Toronto", "Madrid", "Lisbon", "Amsterdam", "Seoul", "Bangkok"]
PURPOSES = ["work", "vacation", "a conference", "a wedding", "family",
            "a medical appointment", "a product launch"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

MEETING_TYPES = ["stand-up", "planning", "retrospective", "kickoff",
                 "1-on-1", "design review", "roadmap", "quarterly"]
TOPICS = ["the Q3 roadmap", "onboarding", "the API redesign",
          "pricing", "hiring", "the outage postmortem", "the migration plan",
          "the annual budget"]
PEOPLE = ["Alex", "Sam", "Jordan", "Priya", "Wei", "Marcus", "Elena",
          "Tomás", "Aiko", "Oliver", "Zara", "Kenji"]

ITEMS = ["laptop", "espresso machine", "bicycle", "bookshelf", "winter jacket",
         "headphones", "backpack", "digital camera", "running shoes", "tent",
         "air purifier", "standing desk"]
STORES = ["Amazon", "a local shop", "Best Buy", "Costco", "IKEA", "REI",
          "a consignment store", "eBay"]
PRICES = ["120", "350", "80", "1200", "45", "890", "230", "60", "520", "75"]

PREF_VERBS = ["prefer", "love", "hate", "avoid", "enjoy", "dislike"]
PREF_OBJECTS = ["spicy food", "early mornings", "jazz music", "long flights",
                "crowded restaurants", "rainy weather", "minimalist design",
                "group travel", "loud offices", "aisle seats"]
REASONS = ["it gives me energy", "it helps me focus", "my back hurts",
           "I sleep poorly", "I grew up with it", "it's cheaper",
           "it's more reliable", "my partner insists"]

SYNONYMS = {
    "flew": "traveled", "from": "departing from", "to": "arriving in",
    "bought": "purchased", "meeting": "discussion", "about": "regarding",
    "prefer": "choose", "love": "adore", "hate": "can't stand",
    "for": "because of", "on": "in",
}

NOISE_TOKENS = ["honestly", "by the way", "as I recall", "I think",
                "incidentally", "weirdly enough", "it was hot that day",
                "my coffee was cold", "traffic was heavy"]


# ---------- generators ----------


def _trip_text(origin: str, dest: str, month: str, purpose: str) -> str:
    return f"I flew from {origin} to {dest} in {month} for {purpose}."


def _meeting_text(mt: str, person: str, topic: str, month: str) -> str:
    return f"I had a {mt} meeting with {person} about {topic} in {month}."


def _purchase_text(item: str, store: str, price: str, month: str) -> str:
    return f"I bought a {item} from {store} for ${price} in {month}."


def _preference_text(verb: str, obj: str, reason: str) -> str:
    return f"I {verb} {obj} because {reason}."


def _make_sibling_trip(base: dict[str, str], rng: random.Random) -> dict[str, str]:
    attrs = dict(base)
    # Change 1-2 attributes; keep origin/dest stable to mark the "family".
    if rng.random() < 0.5:
        attrs["month"] = rng.choice([m for m in MONTHS if m != base["month"]])
    else:
        attrs["purpose"] = rng.choice([p for p in PURPOSES if p != base["purpose"]])
    if rng.random() < 0.3:
        attrs["dest"] = rng.choice([c for c in CITIES if c != base["dest"] and c != base["origin"]])
    return attrs


def _make_sibling_meeting(base: dict[str, str], rng: random.Random) -> dict[str, str]:
    attrs = dict(base)
    if rng.random() < 0.5:
        attrs["month"] = rng.choice([m for m in MONTHS if m != base["month"]])
    else:
        attrs["topic"] = rng.choice([t for t in TOPICS if t != base["topic"]])
    if rng.random() < 0.3:
        attrs["meeting_type"] = rng.choice(
            [t for t in MEETING_TYPES if t != base["meeting_type"]],
        )
    return attrs


def _make_sibling_purchase(base: dict[str, str], rng: random.Random) -> dict[str, str]:
    attrs = dict(base)
    if rng.random() < 0.5:
        attrs["month"] = rng.choice([m for m in MONTHS if m != base["month"]])
    else:
        attrs["price"] = rng.choice([p for p in PRICES if p != base["price"]])
    if rng.random() < 0.3:
        attrs["store"] = rng.choice([s for s in STORES if s != base["store"]])
    return attrs


def _make_sibling_preference(base: dict[str, str], rng: random.Random) -> dict[str, str]:
    attrs = dict(base)
    # Preferences: only change the reason, keep verb+object (that's the identity).
    attrs["reason"] = rng.choice([r for r in REASONS if r != base["reason"]])
    return attrs


def generate_events(
    n_families: int = 50,
    siblings_per_family: int = 5,
    seed: int = 42,
) -> list[Event]:
    """Generate families of near-duplicate events.

    Each family shares a "family_key" — siblings differ only in a few attributes.
    """
    rng = random.Random(seed)
    events: list[Event] = []
    next_id = 0

    per_family_func = {
        "trip": (_trip_text, _make_sibling_trip),
        "meeting": (_meeting_text, _make_sibling_meeting),
        "purchase": (_purchase_text, _make_sibling_purchase),
        "preference": (_preference_text, _make_sibling_preference),
    }

    families_cycle = list(per_family_func.keys())

    for i in range(n_families):
        family = families_cycle[i % len(families_cycle)]
        text_fn, sib_fn = per_family_func[family]

        if family == "trip":
            origin, dest = rng.sample(CITIES, 2)
            base_attrs = {
                "origin": origin,
                "dest": dest,
                "month": rng.choice(MONTHS),
                "purpose": rng.choice(PURPOSES),
            }
            family_key = f"trip::{origin}→{dest}"
        elif family == "meeting":
            base_attrs = {
                "meeting_type": rng.choice(MEETING_TYPES),
                "person": rng.choice(PEOPLE),
                "topic": rng.choice(TOPICS),
                "month": rng.choice(MONTHS),
            }
            family_key = f"meeting::{base_attrs['person']}::{base_attrs['topic']}"
        elif family == "purchase":
            base_attrs = {
                "item": rng.choice(ITEMS),
                "store": rng.choice(STORES),
                "price": rng.choice(PRICES),
                "month": rng.choice(MONTHS),
            }
            family_key = f"purchase::{base_attrs['item']}"
        else:  # preference
            base_attrs = {
                "verb": rng.choice(PREF_VERBS),
                "object": rng.choice(PREF_OBJECTS),
                "reason": rng.choice(REASONS),
            }
            family_key = f"preference::{base_attrs['verb']}::{base_attrs['object']}"

        # Base event
        base_text = _render(family, base_attrs, text_fn)
        events.append(Event(event_id=next_id, family=family, text=base_text,
                            attrs=base_attrs, family_key=family_key))
        next_id += 1

        # Siblings
        for _ in range(siblings_per_family - 1):
            sib_attrs = sib_fn(base_attrs, rng)
            sib_text = _render(family, sib_attrs, text_fn)
            events.append(Event(event_id=next_id, family=family, text=sib_text,
                                attrs=sib_attrs, family_key=family_key))
            next_id += 1

    return events


def _render(family: str, attrs: dict[str, str], text_fn) -> str:  # type: ignore[no-untyped-def]
    if family == "trip":
        return text_fn(attrs["origin"], attrs["dest"], attrs["month"], attrs["purpose"])
    if family == "meeting":
        return text_fn(attrs["meeting_type"], attrs["person"], attrs["topic"], attrs["month"])
    if family == "purchase":
        return text_fn(attrs["item"], attrs["store"], attrs["price"], attrs["month"])
    return text_fn(attrs["verb"], attrs["object"], attrs["reason"])


# ---------- query generation ----------


def _build_siblings_map(events: list[Event]) -> dict[int, list[int]]:
    by_family_key: dict[str, list[int]] = {}
    for e in events:
        by_family_key.setdefault(e.family_key, []).append(e.event_id)
    return {e.event_id: by_family_key[e.family_key] for e in events}


def _partial(event: Event, rng: random.Random) -> str:
    a = event.attrs
    family = event.family
    if family == "trip":
        # Drop origin and purpose, keep dest + month.
        return f"I flew to {a['dest']} in {a['month']}."
    if family == "meeting":
        # Keep person + topic; drop meeting type + month.
        return f"A meeting with {a['person']} about {a['topic']}."
    if family == "purchase":
        # Keep item + month; drop store + price.
        return f"I bought a {a['item']} in {a['month']}."
    # preference: drop the reason.
    return f"I {a['verb']} {a['object']}."


def _paraphrase(event: Event, rng: random.Random) -> str:
    text = event.text
    for k, v in SYNONYMS.items():
        if rng.random() < 0.5 and k in text:
            text = text.replace(k, v, 1)
    return text


def _fragment(event: Event, rng: random.Random) -> str:
    a = event.attrs
    family = event.family
    if family == "trip":
        return f"{a['dest']} trip, {a['month']}"
    if family == "meeting":
        return f"{a['person']} — {a['topic']}"
    if family == "purchase":
        return f"{a['item']}, {a['month']}"
    return f"{a['verb']} {a['object']}"


def _noisy(event: Event, rng: random.Random) -> str:
    base = event.text
    tokens = rng.sample(NOISE_TOKENS, 2)
    return f"{tokens[0]}, {base} {tokens[1]}."


def generate_queries(
    events: list[Event],
    siblings_map: dict[int, list[int]],
    noise_modes: tuple[NoiseMode, ...] = ("partial", "paraphrase", "fragment", "noisy"),
    seed: int = 7,
) -> list[Query]:
    """For each event, emit one query per noise mode."""
    rng = random.Random(seed)
    generators = {
        "partial": _partial,
        "paraphrase": _paraphrase,
        "fragment": _fragment,
        "noisy": _noisy,
    }
    queries: list[Query] = []
    for e in events:
        for mode in noise_modes:
            text = generators[mode](e, rng)
            queries.append(Query(
                text=text,
                target_id=e.event_id,
                sibling_ids=siblings_map[e.event_id],
                noise_mode=mode,
                family=e.family,
            ))
    return queries


def build_dataset(
    n_families: int = 50,
    siblings_per_family: int = 5,
    seed: int = 42,
) -> Dataset:
    events = generate_events(n_families=n_families, siblings_per_family=siblings_per_family, seed=seed)
    siblings_map = _build_siblings_map(events)
    queries = generate_queries(events, siblings_map, seed=seed + 1)
    return Dataset(events=events, queries=queries, siblings_map=siblings_map)
