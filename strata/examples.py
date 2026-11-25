"""The paper's worked examples (Appendix A) as world models."""

from __future__ import annotations

from datetime import date

from .core import Assertion, WorldModel


def example_1() -> WorldModel:
    D = date
    return WorldModel(
        roles={"pm": "pm", "eng": "eng", "cto": "cto", "sales_head": "sales_head"},
        authority={
            "product_scope": {"eng": 1, "pm": 2, "cto": 3},
            "commercial": {"sales_head": 2},
            "architecture": {"eng": 1},
        },
        governance={
            "sf.target": {"product_scope"},
            "sf.priority": {"product_scope", "commercial"},
            "sf_api.limit": {"architecture"},
        },
        assertions=[
            Assertion("a1", D(2026, 1, 12), "pm", "product_scope", "decide", "sf.target", "Q3"),
            Assertion("a2", D(2026, 2, 3), "eng", "architecture", "report", "sf_api.limit", "rate_limited"),
            Assertion("a3", D(2026, 3, 18), "pm", "product_scope", "decide", "sf.target", "Q4", ("a2",)),
            Assertion("a4", D(2026, 4, 22), "cto", "product_scope", "decide", "sf.priority", "below_hubspot"),
            Assertion(
                "a5", D(2026, 5, 30), "sales_head", "commercial", "decide", "sf.priority", "committed_customer_x"
            ),
        ],
    )


# ====================== Appendix A, Example 2 ======================
def example_2() -> WorldModel:
    D = date
    return WorldModel(
        roles={
            "customer_y": "customer",
            "tech_lead": "tech_lead",
            "principal": "principal",
            "engineer": "engineer",
            "support_lead": "support_lead",
            "release": "release",
        },
        authority={
            "architecture": {"engineer": 1, "tech_lead": 2, "principal": 3},
            "requirements": {"customer": 1, "support_lead": 2, "tech_lead": 3},
        },
        governance={"export.spec": {"requirements"}, "export.arch_rule": {"architecture"}},
        assertions=[
            Assertion(
                "b1",
                D(2026, 2, 4),
                "customer_y",
                "requirements",
                "request",
                "export.spec",
                "bulk export of all records, CSV",
            ),
            Assertion(
                "b2",
                D(2026, 2, 11),
                "tech_lead",
                "requirements",
                "decide",
                "export.spec",
                "bulk export, CSV, max 50k rows per job",
                ("b1",),
            ),
            Assertion(
                "b3",
                D(2026, 3, 2),
                "principal",
                "architecture",
                "decide",
                "export.arch_rule",
                "exports run only through the async job service",
                ("incident-17",),
            ),
            Assertion(
                "b4",
                D(2026, 3, 20),
                "engineer",
                "architecture",
                "decide",
                "export.arch_rule",
                "synchronous export is fine for small jobs",
            ),
            Assertion(
                "b5",
                D(2026, 4, 8),
                "release",
                None,
                "implement",
                "export.spec",
                "bulk export, CSV, max 50k rows per job",
            ),
            Assertion(
                "b6",
                D(2026, 5, 14),
                "support_lead",
                "requirements",
                "ask",
                "export.spec",
                "does the 50k cap apply to customer Y's enterprise plan?",
            ),
        ],
    )
