"""Text-generation providers and Azure authentication configuration."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

try:
    from .story_generation import SYSTEM_PROMPT
except ImportError:
    from story_generation import SYSTEM_PROMPT

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable


class TextGenerator(Protocol):
    async def generate(
        self, prompts: Sequence[str], batch_size: int
    ) -> list["TextGenerationResult"]:
        """Generate one response for each prompt, preserving input order."""


class WebSearcher(Protocol):
    async def search_web(self, query: str) -> str:
        """Retrieve grounded web context for one query."""


@dataclass(frozen=True)
class TextGenerationResult:
    text: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class AzureOpenAISettings:
    endpoint: str
    deployment: str
    api_version: str
    scope: str
    api_key: str | None = None

    @classmethod
    def from_env(
        cls,
        model_name: str,
        *,
        prefix: str = "AZURE_OPENAI",
        endpoint_fallback: str | None = None,
        deployment_fallback: str | None = None,
    ) -> "AzureOpenAISettings":
        endpoint = (
            os.getenv(f"{prefix}_ENDPOINT")
            or endpoint_fallback
            or os.getenv("TRAPI_ENDPOINT")
        )
        if not endpoint:
            raise ValueError(
                f"{prefix}_ENDPOINT is required for the Azure provider"
            )
        deployment = (
            os.getenv(f"{prefix}_DEPLOYMENT")
            or deployment_fallback
            or os.getenv("TRAPI_MODEL")
            or model_name
        )
        api_version = (
            os.getenv(f"{prefix}_API_VERSION")
            or os.getenv("TRAPI_API_VERSION")
            or "2025-04-01-preview"
        )
        default_scope = (
            "https://ai.azure.com/.default"
            if endpoint.rstrip("/").endswith("/openai/v1")
            else "https://cognitiveservices.azure.com/.default"
        )
        scope = (
            os.getenv(f"{prefix}_SCOPE")
            or os.getenv("TRAPI_SCOPE")
            or default_scope
        )
        api_key = os.getenv(f"{prefix}_API_KEY")
        return cls(
            endpoint=endpoint.rstrip("/"),
            deployment=deployment,
            api_version=api_version,
            scope=scope,
            api_key=api_key,
        )


def create_azure_client(settings: AzureOpenAISettings):
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Azure text generation requires the openai package"
        ) from exc

    is_v1_endpoint = settings.endpoint.endswith("/openai/v1")
    if settings.api_key and is_v1_endpoint:
        return OpenAI(base_url=f"{settings.endpoint}/", api_key=settings.api_key)
    if settings.api_key:
        return AzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=settings.api_version,
        )

    try:
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Microsoft Entra authentication requires azure-identity"
        ) from exc

    credential = ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    )
    token_provider = get_bearer_token_provider(credential, settings.scope)
    if is_v1_endpoint:
        return OpenAI(base_url=f"{settings.endpoint}/", api_key=token_provider)
    return AzureOpenAI(
        azure_endpoint=settings.endpoint,
        azure_ad_token_provider=token_provider,
        api_version=settings.api_version,
    )


class AzureStoryGenerator:
    """Concurrent Azure OpenAI or authorized TRAPI chat-completions provider."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int,
        temperature: float,
        max_concurrent: int,
    ) -> None:
        self.settings = AzureOpenAISettings.from_env(model_name)
        self.client = create_azure_client(self.settings)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _generate_one(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Azure returned an empty story response")
        return content

    async def _generate_indexed(
        self, index: int, prompt: str
    ) -> tuple[int, TextGenerationResult]:
        try:
            async with self.semaphore:
                response = await asyncio.to_thread(self._generate_one, prompt)
            return index, TextGenerationResult(text=response)
        except Exception as exc:
            return index, TextGenerationResult(error=str(exc))

    async def generate(
        self, prompts: Sequence[str], batch_size: int
    ) -> list[TextGenerationResult]:
        del batch_size
        results = [TextGenerationResult(error="No response")] * len(prompts)
        tasks = [
            asyncio.create_task(self._generate_indexed(index, prompt))
            for index, prompt in enumerate(prompts)
        ]
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating Azure stories",
            unit="story",
        ):
            index, result = await task
            results[index] = result
        return results

    def _search_one(self, query: str) -> str:
        response = self.client.responses.create(
            model=self.settings.deployment,
            input=f"Search the web and return sourced, concise evidence for: {query}",
            tools=[
                {
                    "type": os.getenv(
                        "AZURE_WEB_SEARCH_TOOL", "web_search"
                    )
                }
            ],
        )
        if not any(
            getattr(item, "type", None) == "web_search_call"
            for item in response.output
        ):
            raise RuntimeError("The model returned text without performing web search")
        output = response.output_text
        if not isinstance(output, str) or not output.strip():
            raise ValueError("Azure web search returned an empty response")
        return output

    async def search_web(self, query: str) -> str:
        async with self.semaphore:
            return await asyncio.to_thread(self._search_one, query)


class HuggingFaceStoryGenerator:
    """Lazy local Hugging Face text-generation pipeline."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face dependencies are missing; install requirements.txt"
            ) from exc

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        kwargs: dict[str, Any] = {
            "model": model_name,
            "device_map": "auto",
            "torch_dtype": "auto",
        }
        if token:
            kwargs["token"] = token
        self.pipeline = pipeline("text-generation", **kwargs)
        tokenizer = self.pipeline.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _generate_sync(
        self, prompts: Sequence[str], batch_size: int
    ) -> list[TextGenerationResult]:
        responses: list[TextGenerationResult] = []
        starts = range(0, len(prompts), batch_size)
        for start in tqdm(
            starts,
            total=(len(prompts) + batch_size - 1) // batch_size,
            desc="Generating Hugging Face stories",
            unit="batch",
        ):
            batch = [
                f"{SYSTEM_PROMPT}\n\n{prompt}"
                for prompt in prompts[start : start + batch_size]
            ]
            generation_args: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "return_full_text": False,
                "batch_size": batch_size,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generation_args["temperature"] = self.temperature
            try:
                outputs = self.pipeline(batch, **generation_args)
                for output in outputs:
                    item = output[0] if isinstance(output, list) else output
                    generated_text = item.get("generated_text")
                    if not isinstance(generated_text, str):
                        raise ValueError("Hugging Face returned an unexpected output")
                    responses.append(TextGenerationResult(text=generated_text))
            except Exception as exc:
                responses.extend(
                    TextGenerationResult(error=str(exc)) for _ in batch
                )
        return responses

    async def generate(
        self, prompts: Sequence[str], batch_size: int
    ) -> list[TextGenerationResult]:
        return await asyncio.to_thread(self._generate_sync, prompts, batch_size)


def create_text_generator(
    model_type: str,
    model_name: str,
    *,
    max_new_tokens: int,
    temperature: float,
    max_concurrent: int,
) -> TextGenerator:
    if model_type == "azure":
        return AzureStoryGenerator(
            model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_concurrent=max_concurrent,
        )
    if model_type == "huggingface":
        return HuggingFaceStoryGenerator(
            model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported model type: {model_type}")
