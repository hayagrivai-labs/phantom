"""Phantom Deterministic Runtime.

Manages execution state, event subscriptions and dispatch, bounded counters,
structured diagnostics, and host program execution.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from phantom.capabilities import (
    Capability,
    CapabilityError,
    CapabilityGrant,
)
from phantom.handler_policy import HandlerPolicy
from phantom.host import HostResult, execute_host_program
from phantom.ir import (
    InstructionType,
    PhantomProgram,
    Severity,
    SourceLocation,
)


@dataclass
class DiagnosticRecord:
    """A diagnostic recorded by the Phantom runtime."""

    message: str
    source: SourceLocation | None = None
    severity: Severity = Severity.INFO
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostic record to dictionary."""
        return {
            "message": self.message,
            "source": str(self.source) if self.source else None,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        loc = f" [{self.source}]" if self.source else ""
        return f"[{self.severity.value.upper()}]{loc} {self.message}"


class DiagnosticCollector:
    """Collects structured diagnostics emitted during runtime operations."""

    def __init__(self) -> None:
        self._diagnostics: list[DiagnosticRecord] = []

    def add(
        self,
        message: str,
        source: SourceLocation | None = None,
        severity: Severity = Severity.INFO,
    ) -> None:
        """Add a diagnostic record."""
        self._diagnostics.append(
            DiagnosticRecord(message=message, source=source, severity=severity)
        )

    def get_all(self) -> list[DiagnosticRecord]:
        """Get all recorded diagnostics."""
        return list(self._diagnostics)

    def clear(self) -> None:
        """Clear all recorded diagnostics."""
        self._diagnostics.clear()

    def __len__(self) -> int:
        return len(self._diagnostics)


class PhantomRuntime:
    """Deterministic runtime for Phantom programs and host execution."""

    def __init__(
        self,
        capabilities: CapabilityGrant | None = None,
        handler_policy: HandlerPolicy | None = None,
    ) -> None:
        self._capabilities: CapabilityGrant = (
            capabilities if capabilities is not None else CapabilityGrant()
        )
        self._handler_policy: HandlerPolicy = handler_policy or HandlerPolicy()
        self._events: dict[str, list[str | Callable[..., Any]]] = {}
        self._counters: dict[str, tuple[int, int]] = {}
        self._diagnostics: DiagnosticCollector = DiagnosticCollector()
        self._metadata: dict[str, str] = {}
        self._is_shutdown: bool = False

        # Built-in lifecycle events (auto-declared)
        self._events["phantom.pre_execute"] = []
        self._events["phantom.post_execute"] = []
        self._events["phantom.shutdown"] = []

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

    def initialize(self, program: PhantomProgram) -> list[RuntimeError]:
        """Process all instructions: declare events, register handlers,
        create counters, collect diagnostics, store metadata.
        Validates capabilities before each operation.
        Returns any runtime errors encountered.
        """
        errors: list[RuntimeError] = []

        for inst in program.instructions:
            try:
                self._capabilities.require_instruction(inst.type)
            except CapabilityError as exc:
                errors.append(RuntimeError(str(exc)))
                continue

            try:
                if inst.type == InstructionType.EVENT_DECLARE:
                    name = inst.get_string_arg(0)
                    if not name:
                        errors.append(
                            RuntimeError("event.declare requires a non-empty name argument")
                        )
                    else:
                        self.declare_event(name)

                elif inst.type == InstructionType.EVENT_ON:
                    event_name = inst.get_string_arg(0)
                    handler_path = inst.get_string_arg(1)
                    if not event_name or not handler_path:
                        errors.append(
                            RuntimeError(
                                "event.on requires event name and handler path arguments"
                            )
                        )
                    else:
                        self.register_handler(event_name, handler_path)

                elif inst.type == InstructionType.COUNTER_CREATE:
                    name = inst.get_string_arg(0)
                    limit = inst.get_int_arg(1)
                    if not name or limit is None:
                        errors.append(
                            RuntimeError(
                                "counter.create requires name and integer limit arguments"
                            )
                        )
                    else:
                        self.create_counter(name, limit)

                elif inst.type == InstructionType.COUNTER_INCREMENT:
                    name = inst.get_string_arg(0)
                    if not name:
                        errors.append(
                            RuntimeError("counter.increment requires a counter name argument")
                        )
                    else:
                        self.increment_counter(name)

                elif inst.type == InstructionType.COUNTER_DECREMENT:
                    name = inst.get_string_arg(0)
                    if not name:
                        errors.append(
                            RuntimeError("counter.decrement requires a counter name argument")
                        )
                    else:
                        self.decrement_counter(name)

                elif inst.type == InstructionType.COUNTER_RESET:
                    name = inst.get_string_arg(0)
                    if not name:
                        errors.append(
                            RuntimeError("counter.reset requires a counter name argument")
                        )
                    else:
                        self.reset_counter(name)

                elif inst.type == InstructionType.DIAGNOSTIC_NOTE:
                    message = inst.get_string_arg(0) or ""
                    self.add_diagnostic(message, source=inst.source)

                elif inst.type == InstructionType.DIAGNOSTIC_WARN:
                    message = inst.get_string_arg(0) or ""
                    self.add_diagnostic(message, source=inst.source, severity=Severity.WARNING)

                elif inst.type == InstructionType.META_AUTHOR:
                    author = inst.get_string_arg(0) or ""
                    self.set_metadata("author", author)

                elif inst.type == InstructionType.META_VERSION:
                    version = inst.get_string_arg(0) or ""
                    self.set_metadata("version", version)

                elif inst.type == InstructionType.META_LICENSE:
                    self.set_metadata("license", inst.get_string_arg(0) or "")

                elif inst.type == InstructionType.META_TAGS:
                    self.set_metadata("tags", inst.get_string_arg(0) or "")

            except RuntimeError as exc:
                errors.append(exc)
            except Exception as exc:
                errors.append(
                    RuntimeError(f"Unexpected error processing instruction {inst}: {exc}")
                )

        return errors

    def execute_host(
        self,
        host_file: str,
        args: list[str] | None = None,
        timeout: float = 30.0,
    ) -> HostResult:
        """Run the host Python program in a subprocess.

        Fires lifecycle events: phantom.pre_execute, phantom.post_execute.
        Captures stdout, stderr, exit code.
        """
        self._capabilities.require(Capability.STDIO)

        # Dotted handler paths resolve relative to the host program's directory.
        import os

        host_dir = os.path.dirname(os.path.abspath(host_file))
        orig_sys_path = list(sys.path)
        inserted = False
        if host_dir not in sys.path:
            sys.path.insert(0, host_dir)
            inserted = True

        try:
            # Fire pre-execute lifecycle event
            self.fire_event("phantom.pre_execute", host_file=host_file, args=args or [])

            result = execute_host_program(host_file, args=args, timeout=timeout)

            # Fire post-execute lifecycle event
            self.fire_event("phantom.post_execute", host_file=host_file, result=result)

            return result
        finally:
            if inserted:
                try:
                    sys.path.remove(host_dir)
                except ValueError:
                    sys.path[:] = orig_sys_path

    def shutdown(self) -> None:
        """Fire phantom.shutdown event, clean up resources."""
        if not self._is_shutdown:
            try:
                self.fire_event("phantom.shutdown")
            finally:
                self._is_shutdown = True

    # ──────────────────────────────────────────────────────────────────
    # Event API
    # ──────────────────────────────────────────────────────────────────

    def declare_event(self, name: str) -> None:
        """Declare a named event in the Phantom runtime."""
        self._capabilities.require(Capability.EVENTS)
        if name in self._events:
            raise RuntimeError(f"Event '{name}' has already been declared")
        self._events[name] = []

    def register_handler(
        self, event_name: str, handler_path: str | Callable[..., Any]
    ) -> None:
        """Register a callback handler for a declared event.

        Handlers are identified by dotted path strings but resolved at fire time.
        """
        self._capabilities.require(Capability.EVENTS)
        if event_name not in self._events:
            raise RuntimeError(
                f"Cannot register handler for undeclared event '{event_name}'"
            )
        self._events[event_name].append(handler_path)

    def _resolve_handler(
        self, handler_ref: str | Callable[..., Any]
    ) -> Callable[..., Any]:
        """Resolve a dotted path handler string to a callable attribute at fire time."""
        if callable(handler_ref):
            return handler_ref

        if not isinstance(handler_ref, str):
            raise TypeError(
                f"Handler must be a dotted string path or callable, got {type(handler_ref).__name__}"
            )

        parts = handler_ref.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid handler path '{handler_ref}': must contain at least 'module.function'"
            )

        # Ensure cwd is in sys.path for local module resolution
        import os
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

        # Decide before importing anything (importing runs the module's top-level code).
        # This must come after sys.path is final, or find_spec would not see local modules.
        denial = self._handler_policy.denial(handler_ref)
        if denial is not None:
            raise PermissionError(denial)

        last_error: Exception | None = None
        for i in range(len(parts) - 1, 0, -1):
            mod_name = ".".join(parts[:i])
            attr_chain = parts[i:]
            try:
                mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
                obj: Any = mod
                for attr in attr_chain:
                    obj = getattr(obj, attr)
                if not callable(obj):
                    raise TypeError(f"Resolved attribute '{handler_ref}' is not callable")
                return obj
            except (ImportError, AttributeError) as exc:
                last_error = exc
                continue

        raise ImportError(f"Could not resolve handler '{handler_ref}': {last_error}")

    def _invoke_callback(
        self, fn: Callable[..., Any], kwargs: dict[str, Any]
    ) -> Any:
        """Select matching keyword arguments, then invoke the callback exactly once."""
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            # Some extension callables have no signature. Call once with the
            # event payload; an exception from the handler must never trigger a retry.
            return fn(**kwargs)

        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        pos_args: list[Any] = []
        kw_args: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                if name in kwargs:
                    pos_args.append(kwargs[name])
            elif param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                if name in kwargs:
                    kw_args[name] = kwargs[name]

        if has_var_kw:
            for k, v in kwargs.items():
                if k not in sig.parameters:
                    kw_args[k] = v

        bound = sig.bind(*pos_args, **kw_args)
        return fn(*bound.args, **bound.kwargs)

    def fire_event(self, event_name: str, **kwargs: Any) -> list[Any]:
        """Fire a declared event and execute all registered handlers in order.

        Callbacks fire in registration order. Exceptions in callbacks are caught
        and logged as diagnostics; other callbacks continue to fire.
        Returns the results of all handlers that executed successfully.
        """
        self._capabilities.require(Capability.EVENTS)
        if event_name not in self._events:
            raise RuntimeError(f"Event '{event_name}' is not declared")

        results: list[Any] = []
        handlers = list(self._events[event_name])
        for handler_ref in handlers:
            # Resolve handler at fire time
            try:
                fn = self._resolve_handler(handler_ref)
            except Exception as exc:
                self.add_diagnostic(
                    f"Failed to resolve handler '{handler_ref}' for event '{event_name}': {exc}",
                    severity=Severity.ERROR,
                )
                continue

            # Deterministic dispatch
            try:
                res = self._invoke_callback(fn, kwargs)
                results.append(res)
            except Exception as exc:
                self.add_diagnostic(
                    f"Error in handler '{handler_ref}' for event '{event_name}': {exc}",
                    severity=Severity.ERROR,
                )

        return results

    # ──────────────────────────────────────────────────────────────────
    # Counter API
    # ──────────────────────────────────────────────────────────────────

    def create_counter(self, name: str, limit: int) -> None:
        """Create a bounded counter in the Phantom runtime."""
        self._capabilities.require(Capability.COUNTERS)
        if limit <= 0:
            raise RuntimeError(f"Counter limit must be > 0, got {limit}")
        if name in self._counters:
            raise RuntimeError(f"Counter '{name}' already exists")
        self._counters[name] = (0, limit)

    def increment_counter(self, name: str) -> int:
        """Increment a bounded counter by one, returning the new value.

        If incrementing would exceed limit, a RuntimeError message is logged
        as a diagnostic and the counter remains at its limit.
        """
        self._capabilities.require(Capability.COUNTERS)
        if name not in self._counters:
            raise RuntimeError(f"Counter '{name}' does not exist")

        current, limit = self._counters[name]
        if current >= limit:
            msg = f"RuntimeError: Counter '{name}' exceeded limit ({limit})"
            self.add_diagnostic(msg, severity=Severity.ERROR)
            return limit

        new_val = current + 1
        self._counters[name] = (new_val, limit)
        return new_val

    def decrement_counter(self, name: str) -> int:
        """Decrement a bounded counter by one, returning the new value.

        If decrementing would go below 0, a diagnostic is recorded and the
        counter remains at 0.
        """
        self._capabilities.require(Capability.COUNTERS)
        if name not in self._counters:
            raise RuntimeError(f"Counter '{name}' does not exist")

        current, limit = self._counters[name]
        if current <= 0:
            msg = f"RuntimeError: Counter '{name}' decremented below 0"
            self.add_diagnostic(msg, severity=Severity.WARNING)
            return 0

        new_val = current - 1
        self._counters[name] = (new_val, limit)
        return new_val

    def reset_counter(self, name: str) -> None:
        """Reset a bounded counter to 0, keeping its limit unchanged.

        Raises RuntimeError if the counter does not exist.
        """
        self._capabilities.require(Capability.COUNTERS)
        if name not in self._counters:
            raise RuntimeError(f"Counter '{name}' does not exist")
        _, limit = self._counters[name]
        self._counters[name] = (0, limit)

    def get_counter(self, name: str) -> tuple[int, int]:
        """Get (current, limit) for a named counter."""
        if name not in self._counters:
            raise RuntimeError(f"Counter '{name}' does not exist")
        return self._counters[name]

    # ──────────────────────────────────────────────────────────────────
    # Diagnostic API
    # ──────────────────────────────────────────────────────────────────

    def add_diagnostic(
        self,
        message: str,
        source: SourceLocation | None = None,
        severity: Severity = Severity.INFO,
    ) -> None:
        """Attach a structured diagnostic note."""
        self._capabilities.require(Capability.DIAGNOSTICS)
        self._diagnostics.add(message=message, source=source, severity=severity)

    def get_diagnostics(self) -> list[DiagnosticRecord]:
        """Return all recorded diagnostics."""
        return self._diagnostics.get_all()

    # ──────────────────────────────────────────────────────────────────
    # Metadata API
    # ──────────────────────────────────────────────────────────────────

    def set_metadata(self, key: str, value: str) -> None:
        """Store a metadata key-value pair."""
        self._capabilities.require(Capability.METADATA)
        self._metadata[key] = value

    def get_metadata(self, key: str) -> str | None:
        """Retrieve a metadata value by key, or None if not set."""
        return self._metadata.get(key)

    # ──────────────────────────────────────────────────────────────────
    # Inspection
    # ──────────────────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Full runtime state for inspection."""
        return {
            "capabilities": sorted(c.value for c in self._capabilities.granted),
            "events": {
                name: [
                    h if isinstance(h, str) else getattr(h, "__name__", str(h))
                    for h in handlers
                ]
                for name, handlers in self._events.items()
            },
            "counters": {
                name: {"current": cur, "limit": lim}
                for name, (cur, lim) in self._counters.items()
            },
            "metadata": dict(self._metadata),
            "diagnostics": [d.to_dict() for d in self._diagnostics.get_all()],
            "is_shutdown": self._is_shutdown,
        }
