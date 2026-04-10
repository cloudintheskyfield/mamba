#!/usr/bin/env python3
"""OpenAI-compatible inference API for a local Mamba HF checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stream: bool = False


class MambaServer:
    def __init__(self, model_path: str, device: str, dtype: str):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.load_seconds = 0.0
        self.model = None
        self.tokenizer = None
        self._lock = Lock()

    def load(self) -> None:
        load_kwargs: dict[str, Any] = {"dtype": DTYPE_MAP[self.dtype]}
        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.load_seconds = time.time() - start

    def build_prompt(self, messages: list[Message]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.role.strip().lower()
            parts.append(f"{role}: {message.content.strip()}")
        parts.append("assistant:")
        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> tuple[str, dict[str, int | float]]:
        assert self.model is not None
        assert self.tokenizer is not None
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        do_sample = temperature > 0
        with self._lock, torch.inference_mode():
            started = time.time()
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            elapsed = time.time() - started
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        usage = {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(new_tokens.shape[0]),
            "total_tokens": int(output.shape[1]),
            "generate_seconds": round(elapsed, 3),
        }
        return text, usage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Mamba checkpoint with an OpenAI-like API")
    parser.add_argument("--model-path", required=True, help="Local model directory")
    parser.add_argument("--served-model-name", default="mamba-2.8b-hf")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--api-key", default=os.environ.get("MAMBA_API_KEY"))
    return parser.parse_args()


def build_app(args: argparse.Namespace) -> FastAPI:
    server = MambaServer(args.model_path, args.device, args.dtype)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        server.load()
        yield

    app = FastAPI(title="Mamba OpenAI API", version="0.1.0", lifespan=lifespan)

    def require_api_key(auth_header: str | None) -> None:
        if not args.api_key:
            return
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = auth_header.split(" ", 1)[1].strip()
        if token != args.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def validate_model(requested_model: str | None) -> None:
        if requested_model and requested_model != args.served_model_name:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model '{requested_model}', expected '{args.served_model_name}'",
            )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_path": args.model_path,
            "served_model_name": args.served_model_name,
            "device": args.device,
            "dtype": args.dtype,
            "load_seconds": round(server.load_seconds, 3),
        }

    @app.get("/v1/models")
    def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_api_key(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "owned_by": "cloudintheskyfield",
                }
            ],
        }

    @app.post("/v1/completions")
    def completions(
        request: CompletionRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_api_key(authorization)
        validate_model(request.model)
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        text, usage = server.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": args.served_model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": usage,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        request: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_api_key(authorization)
        validate_model(request.model)
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        prompt = server.build_prompt(request.messages)
        text, usage = server.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "mamba-openai-api",
            "health": "/health",
            "models": "/v1/models",
            "completions": "/v1/completions",
            "chat_completions": "/v1/chat/completions",
        }

    return app


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
