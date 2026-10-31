"""Tests for Phantom deterministic runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from phantom.capabilities import Capability, CapabilityError, CapabilityGrant
from phantom.ir import (
    Argument,
    ArgumentType,
    EncodingMode,
    InstructionType,
    PhantomInstruction,
    PhantomProgram,
    Severity,
    SourceLocation,
)
from phantom.runtime import PhantomRuntime

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def dummy_callback(*args, **kwargs):
    return "called"


def failing_callback(*args, **kwargs):
    raise ValueError("intentional error")


def test_runtime_builtin_events():
    runtime = PhantomRuntime()
    state = runtime.get_state()
    assert "phantom.pre_execute" in state["events"]
    assert "phantom.post_execute" in state["events"]
    assert "phantom.shutdown" in state["events"]


def test_runtime_event_lifecycle():
    runtime = PhantomRuntime()
    runtime.declare_event("my_event")

    # Duplicate declare raises RuntimeError
    with pytest.raises(RuntimeError):
        runtime.declare_event("my_event")

    # Undeclared event register raises RuntimeError
    with pytest.raises(RuntimeError):
        runtime.register_handler("unknown_event", "test_runtime.dummy_callback")

    # Register handler using a registered module in sys.modules
    import sys
    dummy_mod = type(sys)("phantom_test_dummy")
    dummy_mod.dummy_callback = dummy_callback
    sys.modules["phantom_test_dummy"] = dummy_mod

    runtime.register_handler("my_event", "phantom_test_dummy.dummy_callback")
    results = runtime.fire_event("my_event")
    assert results == ["called"]


def test_runtime_event_order_and_exception_handling():
    runtime = PhantomRuntime()
    runtime.declare_event("test_chain")

    order = []

    def cb1():
        order.append(1)
        return 1

    def cb2():
        order.append(2)
        raise RuntimeError("boom")

    def cb3():
        order.append(3)
        return 3

    runtime.register_handler("test_chain", cb1)
    runtime.register_handler("test_chain", cb2)
    runtime.register_handler("test_chain", cb3)

    results = runtime.fire_event("test_chain")
    assert order == [1, 2, 3]
    assert results == [1, 3]

    # Check diagnostic recorded for cb2 exception
    diagnostics = runtime.get_diagnostics()
    assert any("boom" in d.message for d in diagnostics)


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
@pytest.mark.parametrize("handler_kind", ["no_args", "named", "var_kw"])
def test_runtime_handler_error_is_not_retried(error_type, handler_kind):
    runtime = PhantomRuntime()
    calls = []

    def fail(value):
        calls.append(value)
        raise error_type("error inside handler")

    def no_args():
        fail(None)

    def named(value):
        fail(value)

    def var_kw(**kwargs):
        fail(kwargs["value"])

    handler = {"no_args": no_args, "named": named, "var_kw": var_kw}[handler_kind]
    runtime.register_handler("phantom.pre_execute", handler)
    runtime.register_handler("phantom.pre_execute", lambda: "next handler")

    assert runtime.fire_event("phantom.pre_execute", value=42, extra=True) == ["next handler"]
    assert calls == [None if handler_kind == "no_args" else 42]
    diagnostics = runtime.get_diagnostics()
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == Severity.ERROR
    assert "error inside handler" in diagnostics[0].message


def test_runtime_callback_keyword_selection():
    runtime = PhantomRuntime()

    def named(value, *, optional=7):
        return value, optional

    def optional_only(optional=7):
        return optional

    def positional_default(value=7, /):
        return value

    class Handler:
        def __call__(self, value):
            return value

        def method(self, *, value):
            return value

    handler = Handler()
    callbacks = [
        lambda: "no args",
        named,
        optional_only,
        positional_default,
        lambda **kwargs: kwargs,
        handler,
        handler.method,
    ]
    for callback in callbacks:
        runtime.register_handler("phantom.pre_execute", callback)

    assert runtime.fire_event("phantom.pre_execute", value=42, extra=True) == [
        "no args", (42, 7), 7, 42, {"value": 42, "extra": True}, 42, 42,
    ]
    assert runtime.get_diagnostics() == []


@pytest.mark.parametrize("positional_only", [False, True])
def test_runtime_callback_invalid_signature_does_not_run(positional_only):
    runtime = PhantomRuntime()
    calls = []

    def named(required):
        calls.append(required)

    def positional(required, /):
        calls.append(required)

    runtime.register_handler("phantom.pre_execute", positional if positional_only else named)
    kwargs = {"extra": True}
    assert runtime.fire_event("phantom.pre_execute", **kwargs) == []
    assert calls == []
    assert len(runtime.get_diagnostics()) == 1
    assert runtime.get_diagnostics()[0].severity == Severity.ERROR


def test_runtime_positional_only_callback_receives_matching_kwargs():
    runtime = PhantomRuntime()

    def on_event(host_file, /):
        return f"Handled: {host_file}"

    runtime.register_handler("phantom.pre_execute", on_event)
    results = runtime.fire_event("phantom.pre_execute", host_file="app.py")
    assert results == ["Handled: app.py"]
    assert runtime.get_diagnostics() == []


@pytest.mark.parametrize("signature_error", [TypeError, ValueError])
@pytest.mark.parametrize("handler_error", [None, TypeError, ValueError])
def test_runtime_uninspectable_callback_runs_once(monkeypatch, signature_error, handler_error):
    runtime = PhantomRuntime()
    calls = []

    def unavailable_signature(fn):
        raise signature_error("signature unavailable")

    def handler(**kwargs):
        calls.append(kwargs)
        if handler_error is not None:
            raise handler_error("error inside handler")
        return "called"

    monkeypatch.setattr("phantom.runtime.inspect.signature", unavailable_signature)
    runtime.register_handler("phantom.pre_execute", handler)
    assert runtime.fire_event("phantom.pre_execute", value=42) == (
        ["called"] if handler_error is None else []
    )
    assert calls == [{"value": 42}]
    diagnostics = runtime.get_diagnostics()
    assert len(diagnostics) == (0 if handler_error is None else 1)
    if handler_error is not None:
        assert diagnostics[0].severity == Severity.ERROR
        assert "error inside handler" in diagnostics[0].message


def test_runtime_unresolvable_handler():
    runtime = PhantomRuntime()
    runtime.declare_event("missing_handler_ev")
    runtime.register_handler("missing_handler_ev", "non_existent_module.no_func")

    # Should not crash runtime, logs diagnostic
    results = runtime.fire_event("missing_handler_ev")
    assert results == []
    diags = runtime.get_diagnostics()
    assert any("Failed to resolve handler" in d.message for d in diags)


def test_runtime_counters():
    runtime = PhantomRuntime()
    runtime.create_counter("requests", 2)
    assert runtime.get_counter("requests") == (0, 2)

    # Invalid limits and duplicates
    with pytest.raises(RuntimeError):
        runtime.create_counter("invalid", 0)
    with pytest.raises(RuntimeError):
        runtime.create_counter("requests", 10)

    # Increment
    assert runtime.increment_counter("requests") == 1
    assert runtime.increment_counter("requests") == 2
    # Beyond limit: logged as diagnostic, stays at limit
    assert runtime.increment_counter("requests") == 2
    assert runtime.get_counter("requests") == (2, 2)

    diags = runtime.get_diagnostics()
    assert any("exceeded limit" in d.message for d in diags)


def test_runtime_reset_counter():
    runtime = PhantomRuntime()
    runtime.create_counter("requests", 2)
    runtime.increment_counter("requests")
    runtime.increment_counter("requests")
    assert runtime.get_counter("requests") == (2, 2)

    # Reset keeps the limit and returns the counter to 0
    runtime.reset_counter("requests")
    assert runtime.get_counter("requests") == (0, 2)
    assert runtime.increment_counter("requests") == 1

    # Resetting a counter that was never created is an error
    with pytest.raises(RuntimeError):
        runtime.reset_counter("ghost")


def test_runtime_decrement_counter():
    runtime = PhantomRuntime()
    runtime.create_counter("tasks", 5)
    runtime.increment_counter("tasks")
    runtime.increment_counter("tasks")
    assert runtime.get_counter("tasks") == (2, 5)

    assert runtime.decrement_counter("tasks") == 1
    assert runtime.decrement_counter("tasks") == 0
    # Bounded at 0: diagnostic recorded, stays at 0
    assert runtime.decrement_counter("tasks") == 0
    assert runtime.get_counter("tasks") == (0, 5)

    diags = runtime.get_diagnostics()
    assert any("decremented below 0" in d.message for d in diags)

    with pytest.raises(RuntimeError):
        runtime.decrement_counter("ghost")


def test_runtime_initialize_counter_decrement():
    from phantom.parser import parse

    runtime = PhantomRuntime()
    program = parse("""
#: phantom:0.1.0 counter.create("c", 10)
#: phantom:0.1.0 counter.increment("c")
#: phantom:0.1.0 counter.increment("c")
#: phantom:0.1.0 counter.decrement("c")
""")
    runtime.initialize(program)
    assert runtime.get_counter("c") == (1, 10)


def test_runtime_initialize_counter_reset():
    runtime = PhantomRuntime()

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.COUNTER_CREATE, "runs", 5),
        _instruction(InstructionType.COUNTER_INCREMENT, "runs"),
        _instruction(InstructionType.COUNTER_RESET, "runs"),
    ]))

    assert errors == []
    assert runtime.get_counter("runs") == (0, 5)


def test_runtime_initialize_counter_reset_missing_counter():
    runtime = PhantomRuntime()

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.COUNTER_RESET, "ghost"),
    ]))

    assert [str(error) for error in errors] == ["Counter 'ghost' does not exist"]


@pytest.mark.parametrize("values", [(), ("",)])
def test_runtime_initialize_counter_reset_requires_name(values):
    runtime = PhantomRuntime()
    runtime.create_counter("runs", 2)

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.COUNTER_RESET, *values),
    ]))

    assert [str(error) for error in errors] == [
        "counter.reset requires a counter name argument",
    ]
    assert runtime.get_counter("runs") == (0, 2)


def test_runtime_metadata_and_diagnostics():
    runtime = PhantomRuntime()
    runtime.set_metadata("author", "Alice")
    assert runtime.get_metadata("author") == "Alice"
    assert runtime.get_metadata("non_existent") is None

    runtime.add_diagnostic("A test note", severity=Severity.INFO)
    diags = runtime.get_diagnostics()
    assert len(diags) == 1
    assert diags[0].message == "A test note"


def test_runtime_diagnostic_warn_instruction():
    loc = SourceLocation(file="test.py", line=1, column=0, encoding=EncodingMode.COMMENT)
    instructions = [
        PhantomInstruction(
            type=InstructionType.DIAGNOSTIC_WARN,
            arguments=(Argument(type=ArgumentType.STRING, value="Deprecated feature used"),),
            source=loc,
        ),
    ]
    program = PhantomProgram(instructions=instructions, source_file="test.py")

    runtime = PhantomRuntime()
    errors = runtime.initialize(program)
    assert errors == []

    diags = runtime.get_diagnostics()
    assert len(diags) == 1
    assert diags[0].severity == Severity.WARNING
    assert diags[0].message == "Deprecated feature used"
    assert str(diags[0]) == f"[WARNING] [{loc}] Deprecated feature used"


def test_runtime_initialize_program():
    loc = SourceLocation(file="test.py", line=1, column=0, encoding=EncodingMode.COMMENT)
    instructions = [
        PhantomInstruction(
            type=InstructionType.META_AUTHOR,
            arguments=(Argument(type=ArgumentType.STRING, value="Phantom Team"),),
            source=loc,
        ),
        PhantomInstruction(
            type=InstructionType.EVENT_DECLARE,
            arguments=(Argument(type=ArgumentType.STRING, value="on_ready"),),
            source=loc,
        ),
        PhantomInstruction(
            type=InstructionType.COUNTER_CREATE,
            arguments=(
                Argument(type=ArgumentType.STRING, value="runs"),
                Argument(type=ArgumentType.INTEGER, value=5),
            ),
            source=loc,
        ),
        PhantomInstruction(
            type=InstructionType.COUNTER_INCREMENT,
            arguments=(Argument(type=ArgumentType.STRING, value="runs"),),
            source=loc,
        ),
    ]
    program = PhantomProgram(instructions=instructions, source_file="test.py")

    runtime = PhantomRuntime()
    errors = runtime.initialize(program)
    assert errors == []

    assert runtime.get_metadata("author") == "Phantom Team"
    assert "on_ready" in runtime.get_state()["events"]
    assert runtime.get_counter("runs") == (1, 5)


def test_runtime_license_metadata_after_initialize():
    loc = SourceLocation(file="test.py", line=1, column=0, encoding=EncodingMode.COMMENT)
    instructions = [
        PhantomInstruction(
            type=InstructionType.META_LICENSE,
            arguments=(Argument(type=ArgumentType.STRING, value="MIT"),),
            source=loc,
        ),
    ]
    program = PhantomProgram(instructions=instructions, source_file="test.py")

    runtime = PhantomRuntime()
    errors = runtime.initialize(program)
    assert errors == []

    assert runtime.get_metadata("license") == "MIT"


def test_runtime_tags_metadata_after_initialize():
    loc = SourceLocation(file="test.py", line=1, column=0, encoding=EncodingMode.COMMENT)
    instructions = [
        PhantomInstruction(
            type=InstructionType.META_TAGS,
            arguments=(Argument(type=ArgumentType.STRING, value="telemetry,audit,compliance"),),
            source=loc,
        ),
    ]
    program = PhantomProgram(instructions=instructions, source_file="test.py")

    runtime = PhantomRuntime()
    errors = runtime.initialize(program)
    assert errors == []

    assert runtime.get_metadata("tags") == "telemetry,audit,compliance"


@pytest.mark.parametrize("values, expected", [(("telemetry,audit",), "telemetry,audit"), ((), "")])
def test_runtime_initialize_tags_metadata(values, expected):
    runtime = PhantomRuntime()

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.META_TAGS, *values),
    ]))

    assert errors == []
    assert runtime.get_metadata("tags") == expected
def _instruction(instruction_type, *values):
    """Build runtime inputs directly, including arguments validation would reject."""
    return PhantomInstruction(
        type=instruction_type,
        arguments=tuple(
            Argument(
                type=ArgumentType.INTEGER if isinstance(value, int) else ArgumentType.STRING,
                value=value,
            )
            for value in values
        ),
        source=SourceLocation(file="runtime_test.py", line=1),
    )


def test_runtime_initialize_skips_denied_instruction_and_continues():
    runtime = PhantomRuntime(CapabilityGrant([Capability.METADATA]))
    program = PhantomProgram(instructions=[
        _instruction(InstructionType.EVENT_DECLARE, "denied"),
        _instruction(InstructionType.META_AUTHOR, "allowed"),
    ])

    errors = runtime.initialize(program)

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == (
        "Instruction 'event.declare' requires capabilities [cap.events], which are not granted."
    )
    assert "denied" not in runtime.get_state()["events"]
    assert runtime.get_metadata("author") == "allowed"


@pytest.mark.parametrize("values", [(), ("",)])
def test_runtime_initialize_event_declare_requires_name(values):
    runtime = PhantomRuntime()

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.EVENT_DECLARE, *values),
    ]))

    assert [str(error) for error in errors] == [
        "event.declare requires a non-empty name argument",
    ]
    assert set(runtime.get_state()["events"]) == {
        "phantom.pre_execute", "phantom.post_execute", "phantom.shutdown",
    }


@pytest.mark.parametrize("values", [(), ("ready",), ("", "module.handler"), ("ready", "")])
def test_runtime_initialize_event_on_requires_event_and_handler(values):
    runtime = PhantomRuntime()
    runtime.declare_event("ready")

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.EVENT_ON, *values),
    ]))

    assert [str(error) for error in errors] == [
        "event.on requires event name and handler path arguments",
    ]
    assert runtime.get_state()["events"]["ready"] == []


def test_runtime_initialize_event_on_registers_handler():
    runtime = PhantomRuntime()

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.EVENT_DECLARE, "ready"),
        _instruction(InstructionType.EVENT_ON, "ready", "module.handler"),
    ]))

    assert errors == []
    assert runtime.get_state()["events"]["ready"] == ["module.handler"]


@pytest.mark.parametrize("values", [(), ("runs",), ("", 2), ("runs", "2")])
def test_runtime_initialize_counter_create_requires_name_and_integer_limit(values):
    runtime = PhantomRuntime()

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.COUNTER_CREATE, *values),
    ]))

    assert [str(error) for error in errors] == [
        "counter.create requires name and integer limit arguments",
    ]
    assert runtime.get_state()["counters"] == {}


@pytest.mark.parametrize("values", [(), ("",)])
def test_runtime_initialize_counter_increment_requires_name(values):
    runtime = PhantomRuntime()
    runtime.create_counter("runs", 2)

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.COUNTER_INCREMENT, *values),
    ]))

    assert [str(error) for error in errors] == [
        "counter.increment requires a counter name argument",
    ]
    assert runtime.get_counter("runs") == (0, 2)


@pytest.mark.parametrize("values, expected", [(("1.2.3",), "1.2.3"), ((), "")])
def test_runtime_initialize_version_metadata(values, expected):
    runtime = PhantomRuntime()

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.META_VERSION, *values),
    ]))

    assert errors == []
    assert runtime.get_metadata("version") == expected


def test_runtime_initialize_returns_runtime_error_and_continues():
    runtime = PhantomRuntime()

    errors = runtime.initialize(PhantomProgram(instructions=[
        _instruction(InstructionType.EVENT_DECLARE, "ready"),
        _instruction(InstructionType.EVENT_DECLARE, "ready"),
        _instruction(InstructionType.META_AUTHOR, "after error"),
    ]))

    assert [str(error) for error in errors] == ["Event 'ready' has already been declared"]
    assert runtime.get_metadata("author") == "after error"


def test_runtime_initialize_wraps_unexpected_error_and_continues(monkeypatch):
    runtime = PhantomRuntime()
    instruction = _instruction(InstructionType.EVENT_DECLARE, "ready")

    def fail_declaration(name):
        raise ValueError("synthetic declaration failure")

    monkeypatch.setattr(runtime, "declare_event", fail_declaration)
    errors = runtime.initialize(PhantomProgram(instructions=[
        instruction,
        _instruction(InstructionType.META_AUTHOR, "after error"),
    ]))

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == (
        f"Unexpected error processing instruction {instruction}: synthetic declaration failure"
    )
    assert "ready" not in runtime.get_state()["events"]
    assert runtime.get_metadata("author") == "after error"


@pytest.mark.parametrize("handler, message", [
    (42, "Handler must be a dotted string path or callable, got int"),
    ("undotted", "Invalid handler path 'undotted': must contain at least 'module.function'"),
])
def test_runtime_invalid_handler_reference_records_diagnostic(handler, message):
    runtime = PhantomRuntime()
    runtime.register_handler("phantom.pre_execute", handler)

    assert runtime.fire_event("phantom.pre_execute") == []

    diagnostics = runtime.get_diagnostics()
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == Severity.ERROR
    assert diagnostics[0].message == (
        f"Failed to resolve handler '{handler}' for event 'phantom.pre_execute': {message}"
    )


def test_runtime_handler_resolution_adds_cwd(monkeypatch, tmp_path):
    module_name = "phantom_runtime_cwd_callback"
    (tmp_path / f"{module_name}.py").write_text(
        "def handler(value):\n    return value + 1\n", encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(tmp_path)])
    # Record the previous module entry so monkeypatch also removes the later import.
    monkeypatch.setitem(sys.modules, module_name, None)
    del sys.modules[module_name]
    runtime = PhantomRuntime()
    runtime.register_handler("phantom.pre_execute", f"{module_name}.handler")

    assert runtime.fire_event("phantom.pre_execute", value=41) == [42]
    assert sys.path[0] == str(tmp_path)
    assert runtime.get_diagnostics() == []


def test_runtime_non_callable_handler_records_diagnostic(monkeypatch):
    module = ModuleType("phantom_runtime_non_callable")
    module.handler = 42
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = PhantomRuntime()
    handler_path = f"{module.__name__}.handler"
    runtime.register_handler("phantom.pre_execute", handler_path)

    assert runtime.fire_event("phantom.pre_execute") == []

    diagnostics = runtime.get_diagnostics()
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == Severity.ERROR
    assert diagnostics[0].message == (
        f"Failed to resolve handler '{handler_path}' for event 'phantom.pre_execute': "
        f"Resolved attribute '{handler_path}' is not callable"
    )


def test_runtime_capability_restriction():
    # Grant only STDIO (no EVENTS, COUNTERS, etc.)
    grant = CapabilityGrant([Capability.STDIO])
    runtime = PhantomRuntime(capabilities=grant)

    with pytest.raises(CapabilityError):
        runtime.declare_event("foo")

    with pytest.raises(CapabilityError):
        runtime.create_counter("bar", 10)


def test_runtime_execute_host_lifecycle():
    runtime = PhantomRuntime()
    pre_fired = []
    post_fired = []
    shutdown_fired = []

    runtime.register_handler("phantom.pre_execute", lambda **kw: pre_fired.append(kw))
    runtime.register_handler("phantom.post_execute", lambda **kw: post_fired.append(kw))
    runtime.register_handler("phantom.shutdown", lambda **kw: shutdown_fired.append(True))

    target = FIXTURES_DIR / "valid_plain.py"
    res = runtime.execute_host(str(target))
    assert res.exit_code == 0
    assert len(pre_fired) == 1
    assert len(post_fired) == 1

    runtime.shutdown()
    assert len(shutdown_fired) == 1


def test_runtime_execute_host_timeout(tmp_path):
    script = tmp_path / "sleep_script.py"
    script.write_text("import time; time.sleep(5)\n")
    runtime = PhantomRuntime()
    res = runtime.execute_host(str(script), timeout=0.2)
    assert res.exit_code == 124
    assert "timed out" in res.stderr.lower()


def test_runtime_execute_host_restores_sys_path(tmp_path):
    sub = tmp_path / "custom_host_dir"
    sub.mkdir()
    script = sub / "test_prog.py"
    script.write_text("print('hello')\n")
    runtime = PhantomRuntime()
    orig_path = list(sys.path)
    assert str(sub) not in sys.path
    res = runtime.execute_host(str(script))
    assert res.exit_code == 0
    assert str(sub) not in sys.path
    assert sys.path == orig_path
