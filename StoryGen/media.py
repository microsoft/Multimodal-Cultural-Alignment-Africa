"""Image and audio generation providers."""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Mapping
from urllib.request import urlopen

try:
    from .providers import AzureOpenAISettings, create_azure_client
except ImportError:
    from providers import AzureOpenAISettings, create_azure_client

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable


IDENTITY_EDIT_PROMPT = (
    "Preserve the exact identity, face, hair, age, and body proportions of each "
    "person in the reference image. Keep each person distinct. Change only the "
    "clothing, activity, objects, and setting needed for this scene. "
)

DEFAULT_VOICES = {
    "en-ZA": {"female": "en-ZA-LeahNeural", "male": "en-ZA-LukeNeural"},
    "pt-PT": {"female": "pt-PT-RaquelNeural", "male": "pt-PT-DuarteNeural"},
    "en-NG": {"female": "en-NG-EzinneNeural", "male": "en-NG-AbeoNeural"},
    "en-KE": {"female": "en-KE-AsiliaNeural", "male": "en-KE-ChilembaNeural"},
}


def normalized_gender(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"female", "woman", "feminino", "mulher"}:
        return "female"
    return "male"


def voice_for(language: str, gender: str) -> str:
    override = os.getenv("AZURE_SPEECH_VOICE")
    if override:
        return override
    voices = DEFAULT_VOICES.get(language)
    if not voices:
        raise ValueError(
            f"No default Azure Speech voice for {language}; set AZURE_SPEECH_VOICE"
        )
    return voices[normalized_gender(gender)]


class LocalFluxImageGenerator:
    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Local FLUX generation requires torch and diffusers"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("Local FLUX image generation requires a CUDA GPU")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        kwargs = {"torch_dtype": torch.bfloat16}
        if token:
            kwargs["token"] = token
        self.torch = torch
        self.pipeline = FluxPipeline.from_pretrained(model_name, **kwargs)
        self.pipeline.enable_model_cpu_offload()

    def _generate_sync(
        self,
        prompts: Mapping[str, str],
        output_dir: Path,
        base_name: str,
        seed: int,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        images = []
        ordered = [(key, prompts[key]) for key in ("1", "2", "3", "4")]
        for offset, (paragraph, prompt) in enumerate(
            tqdm(ordered, desc=f"Generating FLUX images for {base_name}", unit="image")
        ):
            generator = self.torch.Generator("cpu").manual_seed(seed + offset)
            image = self.pipeline(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=generator,
            ).images[0]
            path = output_dir / f"{base_name}_{paragraph}.png"
            image.save(path)
            images.append(path)
        return images

    async def generate(
        self,
        prompts: Mapping[str, str],
        output_dir: Path,
        base_name: str,
        seed: int,
    ) -> list[Path]:
        return await asyncio.to_thread(
            self._generate_sync, prompts, output_dir, base_name, seed
        )


class AzureImageGenerator:
    def __init__(self, model_name: str) -> None:
        text_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("TRAPI_ENDPOINT")
        text_scope = os.getenv("AZURE_OPENAI_SCOPE") or os.getenv("TRAPI_SCOPE")
        settings = AzureOpenAISettings.from_env(
            model_name,
            prefix="AZURE_IMAGE",
            endpoint_fallback=text_endpoint,
            deployment_fallback=os.getenv("AZURE_IMAGE_DEPLOYMENT") or model_name,
        )
        settings = AzureOpenAISettings(
            endpoint=settings.endpoint,
            deployment=settings.deployment,
            api_version=(
                os.getenv("AZURE_IMAGE_API_VERSION")
                or os.getenv("AZURE_OPENAI_API_VERSION")
                or settings.api_version
            ),
            scope=(
                os.getenv("AZURE_IMAGE_SCOPE")
                or text_scope
                or settings.scope
            ),
            api_key=(
                os.getenv("AZURE_IMAGE_API_KEY")
                or os.getenv("AZURE_OPENAI_API_KEY")
                or settings.api_key
            ),
        )
        self.settings = settings
        self.client = create_azure_client(settings)

    @staticmethod
    def _save_response(response, path: Path) -> None:
        item = response.data[0]
        if item.b64_json:
            content = base64.b64decode(item.b64_json)
        elif item.url:
            with urlopen(item.url, timeout=120) as source:
                content = source.read()
        else:
            raise ValueError("Azure image response contained neither base64 nor URL")
        path.write_bytes(content)

    def _generate_sync(
        self,
        prompts: Mapping[str, str],
        output_dir: Path,
        base_name: str,
        seed: int,
    ) -> list[Path]:
        del seed
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        reference_path: Path | None = None
        for paragraph in tqdm(
            ("1", "2", "3", "4"),
            desc=f"Generating Azure images for {base_name}",
            unit="image",
        ):
            path = output_dir / f"{base_name}_{paragraph}.png"
            if reference_path is None:
                response = self.client.images.generate(
                    model=self.settings.deployment,
                    prompt=prompts[paragraph],
                    n=1,
                    size="1024x1024",
                )
                reference_path = path
            else:
                with reference_path.open("rb") as reference:
                    response = self.client.images.edit(
                        model=self.settings.deployment,
                        image=[reference],
                        prompt=IDENTITY_EDIT_PROMPT + prompts[paragraph],
                        n=1,
                        size="1024x1024",
                    )
            self._save_response(response, path)
            paths.append(path)
        return paths

    async def generate(
        self,
        prompts: Mapping[str, str],
        output_dir: Path,
        base_name: str,
        seed: int,
    ) -> list[Path]:
        return await asyncio.to_thread(
            self._generate_sync, prompts, output_dir, base_name, seed
        )


class AzureSpeechAudioGenerator:
    def __init__(self, max_concurrent: int) -> None:
        key = os.getenv("AZURE_SPEECH_KEY") or os.getenv("SPEECH_KEY")
        region = os.getenv("AZURE_SPEECH_REGION") or os.getenv("SPEECH_REGION")
        if not key or not region:
            raise ValueError(
                "--generate-audio requires AZURE_SPEECH_KEY and "
                "AZURE_SPEECH_REGION"
            )
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise RuntimeError(
                "Audio generation requires azure-cognitiveservices-speech"
            ) from exc
        self.speechsdk = speechsdk
        self.key = key
        self.region = region
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _synthesize(
        self, text: str, voice: str, output_path: Path
    ) -> None:
        speech_config = self.speechsdk.SpeechConfig(
            subscription=self.key, region=self.region
        )
        speech_config.speech_synthesis_voice_name = voice
        speech_config.set_speech_synthesis_output_format(
            self.speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        audio_config = self.speechsdk.audio.AudioOutputConfig(
            filename=str(output_path)
        )
        synthesizer = self.speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason != self.speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = getattr(result, "error_details", "") or ""
            raise RuntimeError(f"Azure Speech synthesis failed: {details}")

    async def _generate_one(
        self, text: str, voice: str, path: Path
    ) -> Path:
        async with self.semaphore:
            await asyncio.to_thread(self._synthesize, text, voice, path)
        return path

    async def generate(
        self,
        paragraphs: Mapping[str, str],
        output_dir: Path,
        base_name: str,
        *,
        language: str,
        gender: str,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        voice = voice_for(language, gender)
        paths = [
            output_dir / f"{base_name}_{paragraph}.mp3"
            for paragraph in ("1", "2", "3", "4")
        ]
        tasks = [
            asyncio.create_task(
                self._generate_one(paragraphs[key], voice, path)
            )
            for key, path in zip(("1", "2", "3", "4"), paths)
        ]
        completed = []
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Generating audio for {base_name}",
            unit="clip",
        ):
            completed.append(await task)
        return sorted(completed)


def create_image_generator(provider: str, model_name: str):
    if provider == "azure":
        return AzureImageGenerator(model_name)
    if provider == "huggingface":
        return LocalFluxImageGenerator(model_name)
    raise ValueError(f"Unsupported image provider: {provider}")
