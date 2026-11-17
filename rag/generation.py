"""Generation: extractive + optional LLM providers."""
from __future__ import annotations

import re
import os
from typing import Any, Dict, List, Optional, Tuple

from rag.chunking import Chunk
from rag.config import GenerationConfig


DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided context. "
    "If the answer is not in the context, say you don't know. Cite sources with [n]."
)


class BaseGenerator:
    def generate(self, query: str, contexts: List[Chunk], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class ExtractiveGenerator(BaseGenerator):
    """Selects the most relevant sentences from retrieved context."""
    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    def generate(self, query, contexts, **kwargs):
        # score sentences by overlap with query
        q_terms = set(re.findall(r"\w+", query.lower()))
        cands = []
        for i, c in enumerate(contexts):
            for sent in re.split(r"(?<=[.!?])\s+", c.text):
                if not sent.strip():
                    continue
                terms = set(re.findall(r"\w+", sent.lower()))
                score = len(q_terms & terms)
                cands.append((score, sent.strip(), i))
        cands.sort(key=lambda x: -x[0])
        chosen = []
        seen = set()
        for score, sent, idx in cands:
            if sent in seen:
                continue
            seen.add(sent)
            chosen.append((sent, idx))
            if len(chosen) >= self.max_sentences:
                break
        if not chosen:
            # fallback: first sentences
            for c in contexts[:1]:
                chosen.append((c.text[:300], 0))
        answer = " ".join(s for s, _ in chosen)
        citations = sorted(set(i + 1 for _, i in chosen))
        return {
            "answer": answer,
            "citations": citations,
            "contexts": [c.text for c in contexts],
            "provider": "extractive",
        }


class OpenAIGenerator(BaseGenerator):
    def __init__(self, model="gpt-4o-mini", temperature=0.2, max_tokens=512, system_prompt=None, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_SYSTEM

    def _build_prompt(self, query, contexts):
        ctx = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(contexts))
        return f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"

    def generate(self, query, contexts, **kwargs):
        prompt = self._build_prompt(query, contexts)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = resp.choices[0].message.content
        return {"answer": answer, "citations": [], "contexts": [c.text for c in contexts], "provider": "openai"}


class AnthropicGenerator(BaseGenerator):
    def __init__(self, model="claude-3-5-sonnet-latest", temperature=0.2, max_tokens=512, system_prompt=None, api_key=None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_SYSTEM

    def generate(self, query, contexts, **kwargs):
        ctx = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(contexts))
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.content[0].text
        return {"answer": answer, "citations": [], "contexts": [c.text for c in contexts], "provider": "anthropic"}


class HuggingFaceGenerator(BaseGenerator):
    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct", temperature: float = 0.2, max_tokens: int = 256, system_prompt: Optional[str] = None, device: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        self.model_name = model

    def _build_messages(self, query, contexts):
        ctx = "\n\n".join(f"[{i+1}] {c.text if hasattr(c,'text') else c}" for i, c in enumerate(contexts))
        user = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def generate(self, query, contexts, **kwargs):
        import torch
        messages = self._build_messages(query, contexts)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = self.system_prompt + "\n\n" + messages[-1]["content"]
        else:
            prompt = self.system_prompt + "\n\n" + messages[-1]["content"]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"answer": gen.strip(), "citations": [], "contexts": [c.text if hasattr(c, "text") else str(c) for c in contexts], "provider": "huggingface"}


def get_generator(config: GenerationConfig) -> BaseGenerator:
    if config.provider == "extractive":
        return ExtractiveGenerator()
    if config.provider == "openai":
        return OpenAIGenerator(config.model, config.temperature, config.max_tokens, config.system_prompt)
    if config.provider == "anthropic":
        return AnthropicGenerator(config.model, config.temperature, config.max_tokens, config.system_prompt)
    if config.provider == "huggingface":
        return HuggingFaceGenerator(config.model, config.temperature, config.max_tokens, config.system_prompt)
    raise ValueError(f"Unknown generation provider: {config.provider}")