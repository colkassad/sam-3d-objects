from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PromptCatalog:
    """Validated query prompts and their canonical output categories."""

    prompts: tuple[str, ...]
    synonym_to_canonical: Mapping[str, str]
    synonym_groups: tuple[tuple[str, tuple[str, ...]], ...]
    categories: tuple[str, ...]

    def normalized_synonyms(self) -> dict[str, list[str]]:
        return {canonical: list(members) for canonical, members in self.synonym_groups}


def parse_prompt_csv(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError("--prompts must be a comma-separated string")
    raw = value.split(",")
    prompts = tuple(item.strip() for item in raw)
    if not prompts or any(not item for item in prompts):
        raise ValueError("--prompts must contain nonempty comma-separated prompts")
    seen: dict[str, str] = {}
    for prompt in prompts:
        folded = prompt.casefold()
        if folded in seen:
            raise ValueError(
                f"duplicate prompt {prompt!r}; prompts are compared case-insensitively"
            )
        seen[folded] = prompt
    return prompts


def build_prompt_catalog(
    prompts: Sequence[str], synonyms: str = ""
) -> PromptCatalog:
    normalized = tuple(str(value).strip() for value in prompts)
    if not normalized or any(not value for value in normalized):
        raise ValueError("at least one nonempty prompt is required")
    by_folded: dict[str, str] = {}
    for prompt in normalized:
        folded = prompt.casefold()
        if folded in by_folded:
            raise ValueError(
                f"duplicate prompt {prompt!r}; prompts are compared case-insensitively"
            )
        by_folded[folded] = prompt

    mapping: dict[str, str] = {prompt: prompt for prompt in normalized}
    member_owner: dict[str, str] = {}
    canonical_names: dict[str, str] = {}
    groups: list[tuple[str, tuple[str, ...]]] = []
    text = str(synonyms or "").strip()
    for raw_group in text.split(";") if text else ():
        group = raw_group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(
                f"invalid --synonyms group {group!r}; expected "
                "canonical:member1,member2"
            )
        canonical, members_text = group.split(":", 1)
        canonical = canonical.strip()
        if not canonical:
            raise ValueError(f"invalid --synonyms group {group!r}; canonical is empty")
        canonical_folded = canonical.casefold()
        if canonical_folded in canonical_names:
            raise ValueError(f"duplicate synonym category {canonical!r}")
        canonical_names[canonical_folded] = canonical
        raw_members = tuple(item.strip() for item in members_text.split(","))
        if not raw_members or any(not item for item in raw_members):
            raise ValueError(
                f"invalid --synonyms group {group!r}; members must be nonempty"
            )
        members: list[str] = []
        local_seen: set[str] = set()
        for member in raw_members:
            folded = member.casefold()
            if folded in local_seen:
                raise ValueError(
                    f"duplicate synonym member {member!r} in {canonical!r}"
                )
            local_seen.add(folded)
            query_prompt = by_folded.get(folded)
            if query_prompt is None:
                raise ValueError(
                    f"synonym member {member!r} is not present in --prompts"
                )
            prior = member_owner.get(folded)
            if prior is not None:
                raise ValueError(
                    f"synonym member {member!r} belongs to both {prior!r} "
                    f"and {canonical!r}"
                )
            member_owner[folded] = canonical
            mapping[query_prompt] = canonical
            members.append(query_prompt)
        groups.append((canonical, tuple(members)))

    categories: list[str] = []
    category_seen: set[str] = set()
    for prompt in normalized:
        canonical = mapping[prompt]
        folded = canonical.casefold()
        if folded not in category_seen:
            categories.append(canonical)
            category_seen.add(folded)
    return PromptCatalog(
        prompts=normalized,
        synonym_to_canonical=mapping,
        synonym_groups=tuple(groups),
        categories=tuple(categories),
    )


def parse_prompt_catalog(prompts: str, synonyms: str = "") -> PromptCatalog:
    return build_prompt_catalog(parse_prompt_csv(prompts), synonyms)
