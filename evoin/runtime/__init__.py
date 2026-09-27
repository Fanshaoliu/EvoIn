"""Lightweight reference runtime that executes a harness directory.

A harness is a directory containing ``base_harness.yaml``, ``systemprompt.md``,
``tools/*.tool.yaml``, harness-local tool implementations (``tool_impl/*.py``)
and middleware (``middleware/*.py``).  The runtime loads that directory, lets a
pluggable target model (``evoin.models.AgentModel``) choose tool calls, executes
the bound tool functions against a per-task document workspace, and records the
trajectory in the row format consumed by ``evoin.evolve`` and ``evoin.tailor``.
"""
