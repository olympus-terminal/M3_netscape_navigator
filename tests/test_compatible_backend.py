import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web

import codex_backend
import compatible_backend as backend


class CompatibleBackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.reply = {
            "model": "local-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
        }
        self.status = 200

        async def models(request):
            return web.json_response({"data": [{"id": "local-test"}]})

        async def complete(request):
            self.requests.append(await request.json())
            return web.json_response(self.reply, status=self.status)

        app = web.Application()
        app.router.add_get("/v1/models", models)
        app.router.add_post("/v1/chat/completions", complete)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.patches = [
            patch.object(backend, "LLM_BASE_URL", f"http://127.0.0.1:{port}/v1"),
            patch.object(backend, "LLM_MODEL", "local-test"),
            patch.object(codex_backend, "LLM_BACKEND", "openai_compatible"),
        ]
        for p in self.patches:
            p.start()

    async def asyncTearDown(self):
        for p in reversed(self.patches):
            p.stop()
        await self.runner.cleanup()

    async def test_preflight_and_structured_request_keep_data_separate(self):
        self.assertEqual((await codex_backend.preflight_codex())[0], "local-test")
        with tempfile.TemporaryDirectory() as tmp:
            schema = Path(tmp) / "judge.schema.json"
            schema.write_text('{"type":"object"}')
            result = await codex_backend.run_codex(
                "trusted instructions", "untrusted source text", timeout_seconds=2,
                output_schema=schema,
            )
        self.assertEqual(json.loads(result), {"ok": True})
        req = self.requests[0]
        self.assertEqual(req["messages"], [
            {"role": "system", "content": "trusted instructions"},
            {"role": "user", "content": "untrusted source text"},
        ])
        self.assertEqual(req["response_format"]["json_schema"]["schema"], {"type": "object"})
        self.assertNotIn("tools", req)

    async def test_unavailable_or_wrong_model_never_falls_back(self):
        with patch.object(backend, "LLM_MODEL", "missing"):
            with self.assertRaisesRegex(codex_backend.CodexExecError, "not advertised"):
                await codex_backend.preflight_codex()
        self.status = 503
        with self.assertRaisesRegex(codex_backend.CodexExecError, "HTTP 503"):
            await codex_backend.run_codex("instruction", "data", timeout_seconds=2)
        self.assertEqual(len(self.requests), 1)

    async def test_rejects_incomplete_wrong_model_and_tool_responses(self):
        cases = [
            ({"model": "wrong", "choices": self.reply["choices"]}, "model"),
            ({"model": "local-test", "choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}, "incomplete"),
            ({"model": "local-test", "choices": [{"finish_reason": "stop", "message": {"content": "", "tool_calls": [{}]}}]}, "tool call"),
            ({"model": "local-test", "choices": []}, "exactly one"),
        ]
        for reply, error in cases:
            self.reply = reply
            with self.subTest(error=error), self.assertRaisesRegex(backend.ModelError, error):
                await backend.transform("instruction", "data", timeout_seconds=2)
