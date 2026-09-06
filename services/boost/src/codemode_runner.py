"""Child process for the `codemode` module.

Reads one JSON line describing the program and the tool names it may call,
executes the program with a stub per tool, and writes a final JSON line with
the captured stdout, the `result` variable and any traceback.

Protocol lines travel on the real stdout; the program's own `print()` output is
captured into a buffer so it can never corrupt the channel.
"""

import io
import json
import sys
import traceback

PROGRAM_FILENAME = "<codemode>"


def _emit(channel, message):
  channel.write(json.dumps(message, ensure_ascii=False, default=repr) + "\n")
  channel.flush()


def _make_stub(name, params, channel):
  def stub(*args, **kwargs):
    bound = dict(kwargs)
    for index, value in enumerate(args):
      if index >= len(params):
        raise TypeError(
          f"{name}() takes at most {len(params)} positional argument(s)"
        )
      bound[params[index]] = value

    _emit(channel, {"call": name, "args": bound})
    line = sys.stdin.readline()
    if not line:
      raise RuntimeError(f"{name}: host closed the connection")

    reply = json.loads(line)
    if "error" in reply and reply["error"] is not None:
      raise RuntimeError(reply["error"])
    return reply.get("result")

  stub.__name__ = name
  stub.__qualname__ = name
  return stub


def _program_traceback(exc: BaseException) -> str:
  """Format a traceback with the runner's own frames removed."""
  frames = [
    frame for frame in traceback.extract_tb(exc.__traceback__)
    if frame.filename != __file__
  ]
  lines = ["Traceback (most recent call last):\n"]
  lines.extend(traceback.StackSummary.from_list(frames).format())
  lines.extend(traceback.format_exception_only(type(exc), exc))
  return "".join(lines)


def main() -> int:
  channel = sys.stdout
  first_line = sys.stdin.readline()
  if not first_line:
    return 1

  request = json.loads(first_line)
  code = request.get("code", "")
  params = request.get("params", {})

  scope = {"__name__": "__main__"}
  for name in request.get("tools", []):
    scope[name] = _make_stub(name, params.get(name, []), channel)

  buffer = io.StringIO()
  sys.stdout = buffer
  error = None
  try:
    exec(compile(code, PROGRAM_FILENAME, "exec"), scope)
  except BaseException as exc:  # noqa: BLE001 - reported back to the model
    error = _program_traceback(exc)
  finally:
    sys.stdout = channel

  _emit(
    channel, {
      "done": True,
      "stdout": buffer.getvalue(),
      "result": scope.get("result"),
      "error": error,
    }
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
