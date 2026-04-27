"""LLM 调用封装 — 支持 DeepSeek / Qwen / OpenAI 兼容接口，以及本地模型"""

import json
import os
import re
from openai import OpenAI


class LocalLLMClient:
    """加载本地合并好的模型（SFT 训练后），接口和 LLMClient 一致"""

    def __init__(self, model_path: str, device: str = "auto"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("请先安装: pip install transformers torch")

        print(f"📦 加载本地模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("✅ 模型加载完成")

    def generate(
        self,
        prompt: str,
        system: str = "你是一个专业的文学分析助手。",
        temperature: float = 0.3,
        max_tokens: int = 512,
        **kwargs,  # 忽略 API 专有参数
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return response.strip()

    def generate_json(self, prompt: str, **kwargs) -> dict:
        """本地模型不保证 JSON 输出，加 fallback 解析"""
        raw = self.generate(prompt, **{k: v for k, v in kwargs.items()
                                       if k in ("system", "temperature", "max_tokens")})
        json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)
        else:
            positions = [p for p in (raw.find("{"), raw.find("[")) if p >= 0]
            if positions:
                raw = raw[min(positions):]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}


class LLMClient:
    def __init__(
        self,
        provider: str = "deepseek",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        PROVIDER_DEFAULTS = {
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
                "env_key": "DEEPSEEK_API_KEY",
            },
            "qwen": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-plus",
                "env_key": "DASHSCOPE_API_KEY",
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "env_key": "OPENAI_API_KEY",
            },
            "groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "model": "llama-3.1-8b-instant",
                "env_key": "GROQ_API_KEY",
            },

        }

        defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["deepseek"])
        self.model = model or defaults["model"]
        api_key = api_key or os.getenv(defaults["env_key"], "")
        print("这是一个api_key: ", api_key)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or defaults["base_url"],
        )

    def _is_deepseek_v4(self) -> bool:
        return self.model.startswith("deepseek-v4-")

    def generate(
        self,
        prompt: str,
        system: str = "你是一个专业的文学分析助手。",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        thinking: str | None = None,
        response_format: dict | None = None,
    ) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": thinking}}
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def generate_json(
        self,
        prompt: str,
        system: str = "你是一个专业的文学分析助手。请严格按照要求输出JSON格式，不要输出其他内容。",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> dict:
        """生成并解析 JSON，带容错处理"""
        last_error = None

        for _ in range(retries):
            is_deepseek_v4 = self._is_deepseek_v4()
            raw = self.generate(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=None if is_deepseek_v4 else None,
                thinking="disabled" if is_deepseek_v4 else None,
                response_format={"type": "json_object"} if is_deepseek_v4 else None,
            )

            if not raw or not raw.strip():
                last_error = ValueError("Model returned empty content for JSON request")
                continue

            json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(1)
            else:
                positions = [p for p in (raw.find("{"), raw.find("[")) if p >= 0]
                if positions:
                    raw = raw[min(positions):]

            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                last_error = e
                continue

        raise last_error or ValueError("Failed to parse JSON response")
