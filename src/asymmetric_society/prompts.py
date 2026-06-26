"""Text-based prompt templates with ``{placeholder}`` variables.

Pattern borrowed from FAIRGAME (arXiv:2504.14325): prompts are plain text with
named placeholders filled per round. We add strict missing-key validation so a
typo in a template surfaces immediately instead of leaking ``{placeholder}``
literals into an LLM call.
"""

from __future__ import annotations

import re
import string

_PLACEHOLDER_RE = re.compile(r"{(\w+)}")


class PromptTemplate:
    """A reusable prompt with ``{name}`` placeholders.

    Example:
        >>> t = PromptTemplate("Round {round} of {total}.")
        >>> t.render(round=1, total=10)
        'Round 1 of 10.'
    """

    def __init__(self, template: str) -> None:
        """Initialize the template.

        Args:
            template: Text containing zero or more ``{name}`` placeholders.
        """
        self.template = template
        self.placeholders: frozenset[str] = frozenset(_PLACEHOLDER_RE.findall(template))

    def render(self, **values: object) -> str:
        """Fill every placeholder and return the resulting string.

        Args:
            **values: One keyword argument per placeholder in the template.

        Returns:
            The template with all placeholders substituted.

        Raises:
            KeyError: If a placeholder has no corresponding value, or an extra
                value is supplied that the template does not use.
        """
        provided = set(values)
        missing = self.placeholders - provided
        if missing:
            raise KeyError(f"Missing placeholder values: {sorted(missing)}")
        extra = provided - self.placeholders
        if extra:
            raise KeyError(f"Unused values supplied: {sorted(extra)}")
        return string.Formatter().vformat(self.template, (), dict(values))
