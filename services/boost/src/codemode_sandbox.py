"""Host side of the `codemode` sandbox.

Spawns `codemode_runner.py` in an isolated interpreter, feeds it the program
plus the names of the hidden local tools, and services every tool call the
program makes over a JSON-lines channel.

This is process isolation, not a security boundary: the child shares Boost's
filesystem and network.
"""

import asyncio
import inspect
import json
import sys
from pathlib import Path

import log
from tools.registry import LOCAL_TOOL_PREFIX

logger = log.setup_logger("codemode")

RUNNER_PATH = Path(__file__).with_name("codemode_runner.py")


def truncate(text: str, max_chars: int) -> str:
  if max_chars > 0 and len(text) > max_chars:
    return text[:max_chars] + f"\n...[truncated to {max_chars} chars]"
  return text


def tool_params(fn) -> list[str]:
  """Positional parameter order of a local tool, for positional stub calls."""
  try:
    return list(inspect.signature(fn).parameters)
  except (TypeError, ValueError):
    return []


def _encode(value):
  """Make a tool result JSON-safe without losing plain strings."""
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  try:
    json.dumps(value)
  except (TypeError, ValueError):
    return repr(value)
  return value


async def _invoke(fn, args: dict):
  result = fn(**args)
  if inspect.iscoroutinefunction(fn):
    result = await result
  return result


async def _exchange(proc, code, hidden_tools, max_calls, captured):
  names = [key[len(LOCAL_TOOL_PREFIX):] for key in hidden_tools]
  params = {
    key[len(LOCAL_TOOL_PREFIX):]: tool_params(fn)
    for key, fn in hidden_tools.items()
  }

  proc.stdin.write(
    (
      json.dumps(
        {
          "code": code,
          "tools": names,
          "params": params,
        },
        ensure_ascii=False,
      ) + "\n"
    ).encode("utf-8")
  )
  await proc.stdin.drain()

  calls = 0
  while True:
    line = await proc.stdout.readline()
    if not line:
      stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
      return {
        "stdout": captured["stdout"],
        "result": None,
        "error": f"runner exited unexpectedly: {stderr or 'no output'}",
      }

    try:
      message = json.loads(line.decode("utf-8"))
    except json.JSONDecodeError:
      return {
        "stdout": captured["stdout"],
        "result": None,
        "error": f"malformed runner message: {line.decode('utf-8', errors='replace').strip()}",
      }

    if message.get("done"):
      captured["stdout"] = message.get("stdout") or ""
      return {
        "stdout": captured["stdout"],
        "result": message.get("result"),
        "error": message.get("error"),
      }

    name = message.get("call")
    args = message.get("args") or {}
    calls += 1

    if calls > max_calls:
      reply = {"error": f"call limit {max_calls} reached"}
    else:
      fn = hidden_tools.get(LOCAL_TOOL_PREFIX + str(name))
      if fn is None:
        reply = {"error": f"unknown tool: {name}"}
      else:
        try:
          reply = {"result": _encode(await _invoke(fn, args))}
        except Exception as exc:  # noqa: BLE001 - surfaced inside the program
          reply = {"error": f"{type(exc).__name__}: {exc}"}

    proc.stdin.write(
      (json.dumps(reply, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    )
    await proc.stdin.drain()


async def run(
  code: str,
  hidden_tools: dict,
  *,
  timeout: int,
  max_output: int,
  max_calls: int,
) -> dict:
  """Run `code` in the sandbox, returning `{stdout, result, error}`."""
  captured = {"stdout": ""}

  try:
    proc = await asyncio.create_subprocess_exec(
      sys.executable,
      "-I",
      "-S",
      str(RUNNER_PATH),
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
  except OSError as exc:
    return {"stdout": "", "result": None, "error": f"failed to start sandbox: {exc}"}

  try:
    outcome = await asyncio.wait_for(
      _exchange(proc, code, hidden_tools, max_calls, captured),
      timeout=timeout,
    )
  except asyncio.TimeoutError:
    logger.warning(f"codemode: program exceeded {timeout}s, killing sandbox")
    outcome = {
      "stdout": captured["stdout"],
      "result": None,
      "error": f"timeout after {timeout}s",
    }
  except Exception as exc:  # noqa: BLE001 - never raises into the tool loop
    outcome = {
      "stdout": captured["stdout"],
      "result": None,
      "error": f"sandbox failure: {type(exc).__name__}: {exc}",
    }
  finally:
    if proc.returncode is None:
      proc.kill()
      await proc.wait()

  outcome["stdout"] = truncate(outcome.get("stdout") or "", max_output)
  return outcome
