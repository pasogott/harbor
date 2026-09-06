"""Unit tests for the codemode Boost module."""

import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chat as ch
import codemode_sandbox
import config
import tools.registry as tool_registry
from modules import codemode
from state import request as request_state
from tools.registry import LOCAL_TOOL_PREFIX


@contextmanager
def request_context():
  req = MagicMock()
  req.state = type("State", (), {})()
  token_req = request_state.set(req)
  try:
    yield req
  finally:
    request_state.reset(token_req)
    if hasattr(req.state, "local_tools"):
      delattr(req.state, "local_tools")
    if hasattr(req.state, codemode.HIDDEN_STORE):
      delattr(req.state, codemode.HIDDEN_STORE)


@contextmanager
def config_overrides(**overrides):
  saved = {key: getattr(config, key).__value__ for key in overrides}
  for key, value in overrides.items():
    getattr(config, key).__value__ = value
  try:
    yield
  finally:
    for key, value in saved.items():
      getattr(config, key).__value__ = value


def make_chat(content: str = "What time is it in Warsaw?") -> ch.Chat:
  return ch.Chat.from_conversation([{"role": "user", "content": content}])


def make_llm(on_final=None):
  llm = MagicMock()
  llm.boost_params = {}

  async def _final():
    if on_final is not None:
      return on_final()
    return "final"

  llm.stream_final_completion = _final
  return llm


def system_messages(chat: ch.Chat) -> list[str]:
  return [
    message["content"] for message in chat.history() if message["role"] == "system"
  ]


async def failing_tool() -> str:
  """
  Always raises so error propagation can be observed.
  """
  raise ValueError("nope")


async def echo_tool(value: str = "x") -> str:
  """
  Echo the given value back.

  Args:
    value (str): Value to echo.
  """
  return value * 100


class TestAdvertisedTools:
  @pytest.mark.asyncio
  async def test_advertises_single_tool(self):
    seen = {}

    with request_context(), config_overrides(TOOLS=["current_time", "add_note", "finish"]):
      chat = make_chat()
      llm = make_llm(on_final=lambda: seen.update(defs=tool_registry.collect_tool_defs()))
      await codemode.apply(chat, llm, {})

    defs = seen["defs"]
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "__tool_execute_code"
    assert list(defs[0]["function"]["parameters"]["properties"]) == ["code"]

  @pytest.mark.asyncio
  async def test_catalog_absorbs(self):
    async def earlier_step_tool(query: str) -> str:
      """
      Registered by an earlier workflow step.

      Args:
        query (str): Anything.
      """
      return query

    with request_context(), config_overrides(TOOLS=["current_time", "finish"]):
      tool_registry.set_local_tool("earlier_step_tool", earlier_step_tool)
      hidden = codemode.hide(codemode.catalog({}))

      assert set(hidden) == {
        LOCAL_TOOL_PREFIX + "current_time",
        LOCAL_TOOL_PREFIX + "finish",
        LOCAL_TOOL_PREFIX + "earlier_step_tool",
      }
      assert list(tool_registry.get_local_tools()) == [
        LOCAL_TOOL_PREFIX + "execute_code"
      ]

  @pytest.mark.asyncio
  async def test_client_tools_untouched(self):
    import llm as llm_mod

    client_tool = {
      "type": "function",
      "function": {"name": "client_side_lookup", "parameters": {}},
    }

    with request_context(), config_overrides(TOOLS=["current_time"]):
      backend = llm_mod.LLM.__new__(llm_mod.LLM)
      backend.model = "test-model"
      backend.params = {"tools": [client_tool]}

      hidden = codemode.hide(codemode.catalog({}))
      params = await backend.resolve_request_params()

      names = [tool["function"]["name"] for tool in params["tools"]]
      assert names == ["client_side_lookup", "__tool_execute_code"]
      assert LOCAL_TOOL_PREFIX + "client_side_lookup" not in hidden
      assert "client_side_lookup" not in codemode.render_prompt(hidden)


class TestPrompt:
  @pytest.mark.asyncio
  async def test_prompt_renders_signatures(self):
    with request_context(), config_overrides(TOOLS=["current_time", "finish"]):
      chat = make_chat()
      await codemode.apply(chat, make_llm(), {})

    prompt = "\n".join(system_messages(chat))
    assert "current_time(timezone: str = 'UTC') -> str" in prompt
    assert "Return the current date and time in a named timezone." in prompt
    assert "finish(answer: str) -> str" in prompt
    assert "execute_code(code)" in prompt
    assert "print(...)" in prompt
    assert "result" in prompt


class TestExecution:
  @pytest.mark.asyncio
  async def test_roundtrip(self):
    with request_context(), config_overrides(TOOLS=["current_time"]):
      codemode.hide(codemode.catalog({}))
      output = await codemode.execute_code(
        "stamp = current_time('UTC')\n"
        "print('now:', stamp)\n"
        "result = len(stamp) > 0\n"
      )

    assert output.startswith("now: ")
    assert "T" in output
    assert "result: True" in output
    assert "error:" not in output

  @pytest.mark.asyncio
  async def test_timeout_kills(self):
    with request_context(), config_overrides(TOOLS=["current_time"], CODEMODE_TIMEOUT=1):
      codemode.hide(codemode.catalog({}))
      output = await codemode.execute_code("while True:\n  pass\n")

    assert "error: timeout after 1s" in output

  @pytest.mark.asyncio
  async def test_program_error(self):
    with request_context(), config_overrides(TOOLS=["current_time"]):
      hidden = codemode.hide(codemode.catalog({}))
      hidden[LOCAL_TOOL_PREFIX + "failing_tool"] = failing_tool

      crash = await codemode.execute_code("raise ValueError('boom')")
      caught = await codemode.execute_code(
        "try:\n"
        "  failing_tool()\n"
        "except RuntimeError as exc:\n"
        "  print('caught', exc)\n"
      )

      assert "Traceback (most recent call last):" in crash
      assert "ValueError: boom" in crash
      assert '<codemode>' in crash
      assert "caught ValueError: nope" in caught

      # The tool loop keeps working after both failures.
      assert list(tool_registry.get_local_tools()) == [
        LOCAL_TOOL_PREFIX + "execute_code"
      ]

  @pytest.mark.asyncio
  async def test_output_and_call_caps(self):
    with request_context(), config_overrides(TOOLS=["current_time"], CODEMODE_MAX_CALLS=2):
      hidden = codemode.hide(codemode.catalog({}))
      hidden[LOCAL_TOOL_PREFIX + "echo_tool"] = echo_tool

      with config_overrides(CODEMODE_MAX_OUTPUT=50):
        capped = await codemode.execute_code("print('x' * 500)")

      limited = await codemode.execute_code(
        "for i in range(4):\n"
        "  try:\n"
        "    echo_tool('a')\n"
        "    print(i, 'ok')\n"
        "  except RuntimeError as exc:\n"
        "    print(i, exc)\n"
      )

    assert "...[truncated to 50 chars]" in capped
    assert len(capped.split("\n")[0]) == 50
    assert "0 ok" in limited and "1 ok" in limited
    assert "2 call limit 2 reached" in limited
    assert "3 call limit 2 reached" in limited


class TestRegistryLifecycle:
  @pytest.mark.asyncio
  async def test_restores_registry(self):
    expected = {
      LOCAL_TOOL_PREFIX + "current_time",
      LOCAL_TOOL_PREFIX + "finish",
    }

    with request_context(), config_overrides(TOOLS=["current_time", "finish"]):
      await codemode.apply(make_chat(), make_llm(), {})
      assert set(tool_registry.get_local_tools()) == expected

    def _boom():
      raise RuntimeError("backend down")

    with request_context(), config_overrides(TOOLS=["current_time", "finish"]):
      with pytest.raises(RuntimeError):
        await codemode.apply(make_chat(), make_llm(on_final=_boom), {})
      assert set(tool_registry.get_local_tools()) == expected

    with request_context(), config_overrides(
      TOOLS=["current_time", "finish"],
      CODEMODE_TIMEOUT=1,
    ):
      hidden = codemode.hide(codemode.catalog({}))
      timed_out = await codemode.execute_code("while True:\n  pass\n")
      assert "timeout" in timed_out
      codemode.restore(hidden)
      assert set(tool_registry.get_local_tools()) == expected


class TestSandboxHelpers:
  def test_truncate_is_noop_under_cap(self):
    assert codemode_sandbox.truncate("abc", 10) == "abc"

  def test_render_signature_optional_types(self):
    parameters = {
      "properties": {
        "pattern": {"type": "string"},
        "glob": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "max_matches": {"type": "integer", "default": 5},
      },
      "required": ["pattern"],
    }
    assert codemode.render_signature("grep_workspace", parameters) == (
      "grep_workspace(pattern: str, glob: str | None = None, max_matches: int = 5) -> str"
    )
