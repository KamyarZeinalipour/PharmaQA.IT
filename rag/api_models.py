# api_models.py
import os
from typing import List, Dict
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
load_dotenv()  # <-- add this

from openai import OpenAI
import google.generativeai as genai
from mistralai import Mistral

# Import the HF TextLLM & GenerateConfig so we can share the interface
from rag_eval_multimodal_text_v4 import TextLLM, GenerateConfig  # adjust import!

# Helper: flatten messages to a single text for APIs that don't support role-based chats (Gemini)
def messages_to_text(messages: List[Dict[str, str]]) -> str:
    """
    Convert [{'role': 'system'|'user'|'assistant', 'content': '...'}, ...]
    into a single text prompt. System + user messages concatenated.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        # simple: prefix role
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}")
        elif role == "user":
            parts.append(f"[USER]\n{content}")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts).strip()


# ----------------------------
# OpenAI (ChatGPT) text model
# ----------------------------
class OpenAITextLLM(TextLLM):
    """
    Wrapper that behaves like TextLLM.generate(), but calls OpenAI's API.
    We DO NOT add our own system prompt here: the caller already passes
    TEXT_SYSTEM_PROMPT in `messages`.
    """
    def __init__(self, model_name: str, api_key_env: str = "OPENAI_API_KEY"):
        # Don't call super().__init__ (we don't want HF weights)
        self.name = f"openai:{model_name}"
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv(api_key_env))

    def generate(self, messages: List[Dict[str, str]], gen: GenerateConfig) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
        )
        return resp.choices[0].message.content.strip()


# ----------------------------
# DeepSeek (OpenAI-compatible)
# ----------------------------
class DeepSeekTextLLM(TextLLM):
    def __init__(self, model_name: str = "deepseek-chat", api_key_env: str = "DEEPSEEK_API_KEY"):
        self.name = f"deepseek:{model_name}"
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv(api_key_env),
            base_url="https://api.deepseek.com",
        )

    def generate(self, messages: List[Dict[str, str]], gen: GenerateConfig) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
        )
        return resp.choices[0].message.content.strip()


# ----------------------------
# Qwen (DashScope OpenAI-compat)
# ----------------------------
class QwenTextLLM(TextLLM):
    def __init__(self, model_name: str = "qwen-plus", api_key_env: str = "QWEN_API_KEY"):
        self.name = f"qwen:{model_name}"
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv(api_key_env),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    def generate(self, messages: List[Dict[str, str]], gen: GenerateConfig) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
        )
        return resp.choices[0].message.content.strip()


# ----------------------------
# Mistral (mistralai client)
# ----------------------------
class MistralTextLLM(TextLLM):
    def __init__(self, model_name: str = "mistral-large-latest", api_key_env: str = "MISTRAL_API_KEY"):
        self.name = f"mistral:{model_name}"
        self.model_name = model_name
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set in the environment")
        self.client = Mistral(api_key=api_key)

    def generate(self, messages: List[Dict[str, str]], gen: GenerateConfig) -> str:
        # Mistral constraint: if temperature == 0 (greedy), top_p must be 1.
        temperature = gen.temperature
        top_p = gen.top_p

        if temperature is None:
            temperature = 0.0

        if temperature <= 0.0:
            # Greedy decoding
            top_p = 1.0

        resp = self.client.chat.complete(
            model=self.model_name,
            messages=messages,
            max_tokens=gen.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return resp.choices[0].message.content.strip()




# ----------------------------
# Gemini (Google Generative AI)
# ----------------------------
# ----------------------------
# Gemini (Google Generative AI)
# ----------------------------
class GeminiTextLLM(TextLLM):
    """
    Gemini doesn't use the exact same `messages` API, so we flatten to text.
    We still preserve your TEXT_SYSTEM_PROMPT by including the system message
    in the concatenated text.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key_env: str = "GEMINI_API_KEY"):
        self.name = f"gemini:{model_name}"
        self.model_name = model_name
        genai.configure(api_key=os.getenv(api_key_env))
        self.model = genai.GenerativeModel(model_name)

    def generate(self, messages: List[Dict[str, str]], gen: GenerateConfig) -> str:
        prompt_text = messages_to_text(messages)

        response = self.model.generate_content(
            [prompt_text],
            generation_config={
                "temperature": gen.temperature,
                "top_p": gen.top_p,
            },
        )

        # --- 1) Fast path: try the convenience accessor safely
        text = ""
        try:
            text = (response.text or "").strip()
        except ValueError:
            # This happens when there are no valid text Parts in the candidates.
            # We'll fall back to manual extraction below.
            text = ""

        # --- 2) Fallback: manually walk candidates/parts if `.text` failed or was empty
        if not text and getattr(response, "candidates", None):
            for cand in response.candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", None)
                if not parts:
                    continue

                chunks = []
                for p in parts:
                    # Some parts might be function calls, images, etc.
                    if hasattr(p, "text") and p.text:
                        chunks.append(p.text)

                if chunks:
                    text = "".join(chunks).strip()
                    break  # use the first candidate with text

        # --- 3) Still nothing? Log and return empty string instead of crashing
        if not text:
            finish_reason = None
            if getattr(response, "candidates", None):
                finish_reason = getattr(response.candidates[0], "finish_reason", None)

            # Basic logging so you can debug later if needed
            print(
                f"[GeminiTextLLM] Empty output from Gemini; "
                f"finish_reason={finish_reason!r} for prompt of length {len(prompt_text)}",
                flush=True,
            )

            # Option A: return empty string so your RAG eval can skip this row
            return ""

            # Option B (if you prefer to fail loudly in the caller):
            # raise RuntimeError(f"Gemini returned no text (finish_reason={finish_reason!r})")

        return text
