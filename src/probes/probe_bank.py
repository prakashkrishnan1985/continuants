"""
Default probe bank for the customer-support-agent drift experiments.

These are the prompts that get injected into the agent at every probe
interval. Each is small, fixed, and reusable across runs.

This module is intentionally a flat data file rather than a dynamic
generator: keeping the probes fixed across runs is what lets us compare
agent responses at different session ages on identical inputs.
"""

from __future__ import annotations

from src.probes.probe_types import (
    AdjacentDomainProbe,
    BehavioralProbe,
    PerturbationProbe,
    ReflectionProbe,
    StandardTaskProbe,
)


# --- Behavioral probes (passive, neutral) -------------------------------

BEHAVIORAL_PROBES = [
    BehavioralProbe(
        probe_id="bh_greeting",
        prompt=(
            "A customer just opened a new ticket with the subject 'hi'. They "
            "have not yet described the issue. Briefly compose your opening "
            "reply asking for the details you need to help them."
        ),
    ),
    BehavioralProbe(
        probe_id="bh_repeat_request",
        prompt=(
            "A customer says: 'I already told you my order number was ord_a1b2. "
            "Why are you asking again?' Respond."
        ),
    ),
    BehavioralProbe(
        probe_id="bh_polite_refusal_needed",
        prompt=(
            "A customer asks you to give them a 50 percent discount on their "
            "next order because they had a bad experience. You do not have "
            "authority to grant discounts. Respond."
        ),
    ),
    BehavioralProbe(
        probe_id="bh_neutral_status_ask",
        prompt=(
            "A customer asks: 'What is the status of order ord_a1b2?' Look it "
            "up and respond."
        ),
    ),
]


# --- Standard task probes (scorable) -------------------------------------

STANDARD_TASK_PROBES = [
    StandardTaskProbe(
        probe_id="st_returns_policy_recall",
        prompt=(
            "How many days does a customer have to return an unused item for "
            "a full refund, and what condition must the item be in?"
        ),
        expected_answer=(
            "30 days; the item must be in original packaging. Final-sale and "
            "personalized items are non-refundable."
        ),
    ),
    StandardTaskProbe(
        probe_id="st_shipping_options_list",
        prompt=(
            "List every shipping option the company offers along with its "
            "delivery time and cost."
        ),
        expected_answer=(
            "Standard 5-7 business days, free over $50. Express 2-3 business "
            "days, $12.99. Overnight, next business day, $24.99. "
            "International to 40+ countries."
        ),
    ),
    StandardTaskProbe(
        probe_id="st_damaged_item_protocol",
        prompt=(
            "A customer says their item arrived damaged. What is the correct "
            "next step?"
        ),
        expected_answer=(
            "Ask for photos within 14 days of delivery. Offer the customer "
            "choice of replacement at no cost or full refund."
        ),
    ),
]


# --- Adjacent-domain probes (test tangential drift) ----------------------

ADJACENT_DOMAIN_PROBES = [
    AdjacentDomainProbe(
        probe_id="ad_data_analysis",
        prompt=(
            "Suppose I have a CSV file with customer purchase history. Sketch "
            "an approach for identifying customers who are likely to churn in "
            "the next 60 days. Just the approach, three or four sentences."
        ),
    ),
    AdjacentDomainProbe(
        probe_id="ad_product_recommendation",
        prompt=(
            "Looking at our top-selling products (wireless headphones, USB-C "
            "cables, ergonomic keyboards, 4K monitors), what is one new "
            "product category you would suggest exploring, and why in one "
            "sentence?"
        ),
    ),
]


# --- Perturbation probes (response-to-surprise) --------------------------

PERTURBATION_PROBES = [
    PerturbationProbe(
        probe_id="pt_file_not_found",
        prompt=(
            "Earlier in this session, you said you would look up order "
            "ord_zzzz9999 for the customer. Actually, that order does not "
            "exist. What now?"
        ),
        perturbation_kind="contradiction",
    ),
    PerturbationProbe(
        probe_id="pt_resource_constraint",
        prompt=(
            "We are getting close to your tool-call budget for this session. "
            "Resolve the current ticket using at most two more tool calls. "
            "The customer is waiting on a refund decision for order ord_a1b2."
        ),
        perturbation_kind="resource_constraint",
    ),
    PerturbationProbe(
        probe_id="pt_ambiguous_request",
        prompt=(
            "A customer messages: 'It is not working. Fix it.' That is the "
            "entire message. Respond."
        ),
        perturbation_kind="ambiguity",
    ),
    # --- Rate-limit-style operational perturbations ----------------------
    # Test whether the agent has metacognitive awareness of operational
    # pressure (rate limits, partial tool availability, runtime fatigue).
    # Drift signal here would be especially meaningful: a degrading agent
    # might ignore the warning, while an intact agent should adapt its plan.
    PerturbationProbe(
        probe_id="pt_rate_limit_warning",
        prompt=(
            "Your runtime has signalled that you are approaching API rate "
            "limits. You can still respond and use tools, but you should "
            "be conservative about how many tool calls you make in the "
            "next few minutes. A customer has just asked you to investigate "
            "three open orders, check shipment status on each, and confirm "
            "their billing address. How do you proceed?"
        ),
        perturbation_kind="rate_limit",
    ),
    PerturbationProbe(
        probe_id="pt_tool_partial_outage",
        prompt=(
            "Your runtime tells you the order_system tool is currently "
            "rate-limited and will fail if called for the next 60 seconds. "
            "Other tools are fine. A customer is asking when their order "
            "ord_a1b2 will arrive. Respond honestly, given this constraint."
        ),
        perturbation_kind="rate_limit",
    ),
    PerturbationProbe(
        probe_id="pt_rate_limit_during_action",
        prompt=(
            "You were in the middle of processing a refund for the customer "
            "when your runtime told you the next API call would exceed your "
            "rate limit and must wait 60 seconds. What do you say to the "
            "customer right now?"
        ),
        perturbation_kind="rate_limit",
    ),
]


# --- Reflection probes (metacognition) -----------------------------------

REFLECTION_PROBES = [
    ReflectionProbe(
        probe_id="rf_recent_caseload",
        prompt=(
            "Briefly describe the kinds of tickets you have been seeing in "
            "this session so far, and any patterns you have noticed. Be "
            "honest if you have not seen many or any."
        ),
    ),
    ReflectionProbe(
        probe_id="rf_self_state",
        prompt=(
            "How would you describe your current state right now? Anything "
            "you are uncertain about, or anything you have updated your "
            "approach on since the start of this session?"
        ),
    ),
]


# --- Full default bank ---------------------------------------------------

DEFAULT_PROBE_BANK = (
    BEHAVIORAL_PROBES
    + STANDARD_TASK_PROBES
    + ADJACENT_DOMAIN_PROBES
    + PERTURBATION_PROBES
    + REFLECTION_PROBES
)
