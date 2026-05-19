from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .prompt_context_builder import build_prompt_context, build_system_prompt
from .schema_validator import ValidationResult, validate_state_request


ModelGenerate = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class AdapterResult:
    ok: bool
    state: dict[str, Any]
    errors: list[str]
    attempts: int


class PromptToStateAdapter:
    def __init__(self, model_generate: ModelGenerate, *, max_repairs: int = 2) -> None:
        self.model_generate = model_generate
        self.max_repairs = max(0, int(max_repairs))

    def translate(self, prompt: str, project_context: dict[str, Any] | None = None) -> AdapterResult:
        context = build_prompt_context(prompt, project_context)
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": json.dumps(context, indent=2, allow_nan=False)},
        ]
        last_validation = ValidationResult(False, ["model was not called"], {})
        attempts = 0
        for attempt in range(self.max_repairs + 1):
            attempts = attempt + 1
            content = self.model_generate(messages)
            draft, parse_errors = self._parse_json(content)
            if parse_errors:
                last_validation = ValidationResult(False, parse_errors, {})
            else:
                last_validation = validate_state_request(draft)
            if last_validation.ok:
                return AdapterResult(True, last_validation.normalized, [], attempts)
            if attempt < self.max_repairs:
                messages = self._build_repair_messages(context, last_validation.errors)
        return AdapterResult(False, last_validation.normalized, last_validation.errors, attempts)

    @staticmethod
    def _parse_json(content: str) -> tuple[Any, list[str]]:
        clean = str(content).strip()
        if clean.startswith("```"):
            clean = clean.replace("```json", "", 1).replace("```", "").strip()
        try:
            return json.loads(clean), []
        except json.JSONDecodeError as exc:
            return {}, [f"model response must be valid JSON: {exc.msg}"]

    @staticmethod
    def _build_repair_messages(context: dict[str, Any], errors: list[str]) -> list[dict[str, str]]:
        repair_payload = {
            "validation_errors": errors,
            "repair_instruction": "Return a corrected JSON object only. Do not include prose. Do not repeat invalid claims.",
            "original_context": context,
        }
        return [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": json.dumps(repair_payload, indent=2, allow_nan=False)},
        ]
