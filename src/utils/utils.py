import base64
import os
import time
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr

from .llm import DeepSeekR1ChatOllama, DeepSeekR1ChatOpenAI

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "intern": "Intern",
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "alibaba": "Alibaba",
    "moonshot": "MoonShot",
}


def get_llm_model(provider: str, **kwargs: Any) -> BaseChatModel:
    """获取LLM 模型

    :param provider: 模型类型
    :param kwargs: 其他参数
    :return: LLM模型实例
    """
    api_key: SecretStr | None = None

    if provider not in ["ollama"]:
        env_var = f"{provider.upper()}_API_KEY"
        api_key_str = kwargs.get("api_key", "") or os.getenv(env_var, "")
        if not api_key_str:
            raise MissingAPIKeyError(provider, env_var)
        api_key = SecretStr(api_key_str)

    if provider == "anthropic":
        base_url = kwargs.get("base_url") or "https://api.anthropic.com"

        # ChatAnthropic requires a non-None api_key
        if api_key is None:
            raise ValueError("API key cannot be None for Anthropic")

        return ChatAnthropic(
            model_name=kwargs.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
            timeout=kwargs.get("timeout", 60),
            stop=kwargs.get("stop"),
        )
    elif provider == "mistral":
        base_url = kwargs.get("base_url") or os.getenv(
            "MISTRAL_ENDPOINT", "https://api.mistral.ai/v1"
        )

        return ChatMistralAI(
            name=kwargs.get("model_name", "mistral-large-latest"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == "openai":
        base_url = kwargs.get("base_url") or os.getenv(
            "OPENAI_ENDPOINT", "https://api.openai.com/v1"
        )

        return ChatOpenAI(
            name=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "intern":
        base_url = kwargs.get("base_url") or os.getenv(
            "INTERN_ENDPOINT", "https://chat.intern-ai.org.cn/api/v1"
        )

        return ChatOpenAI(
            model=kwargs.get("model_name", "internlm3-latest"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == "deepseek":
        base_url = kwargs.get("base_url") or os.getenv("DEEPSEEK_ENDPOINT", "")

        if kwargs.get("model_name", "deepseek-chat") == "deepseek-reasoner":
            return DeepSeekR1ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-reasoner"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
        else:
            return ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-chat"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash-exp"),
            temperature=kwargs.get("temperature", 0.0),
            api_key=api_key,
        )
    elif provider == "ollama":
        base_url = kwargs.get("base_url") or os.getenv(
            "OLLAMA_ENDPOINT", "http://localhost:11434"
        )

        if "deepseek-r1" in kwargs.get("model_name", "qwen2.5:7b"):
            return DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                base_url=base_url,
            )
        else:
            return ChatOllama(
                model=kwargs.get("model_name", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )
    elif provider == "azure_openai":
        base_url = kwargs.get("base_url") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_version = kwargs.get("api_version", "") or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"
        )
        return AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=api_key,
        )
    elif provider == "alibaba":
        base_url = kwargs.get("base_url") or os.getenv(
            "ALIBABA_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        return ChatOpenAI(
            model=kwargs.get("model_name", "qwen-plus"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "moonshot":
        base_url = kwargs.get("base_url") or os.getenv("MOONSHOT_ENDPOINT")

        return ChatOpenAI(
            model=kwargs.get("model_name", "moonshot-v1-32k-vision-preview"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Predefined model names for common providers
model_names: dict[str, list[str]] = {
    "intern": ["internlm3-latest", "internlm2.5-latest", "internvl-latest"],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
    ],
    "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "google": [
        "gemini-2.0-flash",
        "gemini-2.0-flash-thinking-exp",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b-latest",
        "gemini-2.0-flash-thinking-exp-01-21",
        "gemini-2.0-pro-exp-02-05",
    ],
    "ollama": [
        "qwen2.5:7b",
        "qwen2.5:14b",
        "qwen2.5:32b",
        "qwen2.5-coder:14b",
        "qwen2.5-coder:32b",
        "llama2:7b",
        "deepseek-r1:14b",
        "deepseek-r1:32b",
    ],
    "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "mistral": [
        "pixtral-large-latest",
        "mistral-large-latest",
        "mistral-small-latest",
        "ministral-8b-latest",
    ],
    "alibaba": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"],
    "moonshot": ["moonshot-v1-32k-vision-preview", "moonshot-v1-8k-vision-preview"],
}


def update_model_dropdown(
    llm_provider: str, api_key: str | None = None, base_url: str | None = None
) -> Any:
    """Update the model name dropdown with predefined models for the selected provider."""
    import gradio as gr

    # Use API keys from .env if not provided
    if not api_key:
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY", "")
    if not base_url:
        base_url = os.getenv(f"{llm_provider.upper()}_BASE_URL", "")

    # Use predefined models for the selected provider
    if llm_provider in model_names:
        return gr.Dropdown(
            choices=model_names[llm_provider],
            value=model_names[llm_provider][0],
            interactive=True,
        )
    else:
        return gr.Dropdown(
            choices=[], value="", interactive=True, allow_custom_value=True
        )


class MissingAPIKeyError(Exception):
    """Custom exception for missing API key."""

    def __init__(self, provider: str, env_var: str) -> None:
        provider_display = PROVIDER_DISPLAY_NAMES.get(provider, provider.upper())
        super().__init__(
            f"💥 {provider_display} API key not found! 🔑 Please set the "
            f"`{env_var}` environment variable or provide it in the UI."
        )


def encode_image(img_path: str | None) -> str | None:
    """编码图片为base64字符串"""
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data


def get_latest_files(
    directory: str, file_types: list[str] = [".webm", ".zip"]
) -> dict[str, str | None]:
    """Get the latest recording and trace files"""
    latest_files: dict[str, str | None] = dict.fromkeys(file_types)

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")

    return latest_files


async def capture_screenshot(browser_context: Any) -> str | None:
    """Capture and encode a screenshot"""
    # Extract the Playwright browser instance
    playwright_browser = (
        browser_context.browser.playwright_browser
    )  # Ensure this is correct.

    # Check if the browser instance is valid and if an existing context can be reused
    if playwright_browser and playwright_browser.contexts:
        playwright_context = playwright_browser.contexts[0]
    else:
        return None

    # Access pages in the context
    pages = None
    if playwright_context:
        pages = playwright_context.pages

    # Use an existing page or create a new one if none exist
    if pages:
        active_page = pages[0]
        for page in pages:
            if page.url != "about:blank":
                active_page = page
    else:
        return None

    # Take screenshot
    try:
        screenshot = await active_page.screenshot(type="jpeg", quality=75, scale="css")
        encoded = base64.b64encode(screenshot).decode("utf-8")
        return encoded
    except Exception:
        return None
