"""The five inbox categories, their criteria, and the Gmail labels they map to."""

from __future__ import annotations

OTP = "OTP"
UNSURE = "Unsure"

CATEGORY_CRITERIA: dict[str, str] = {
    OTP: (
        "Its purpose is to hand the recipient a machine-generated one-time passcode, "
        "verification code, 2FA token, magic sign-in link, or recovery code that "
        "must be entered or clicked somewhere else."
    ),
    "Primary": (
        "Personal, individual-to-individual correspondence addressed to the recipient as "
        "a specific person: colleagues, friends, family, recruiters, or a human contact "
        "replying in a named thread. Written by a person, for this person, and plausibly "
        "expecting a reply."
    ),
    "Transaction": (
        "A receipt, invoice, order or booking confirmation, shipping status, payment or "
        "refund notice, account or security alert, or another automated notification about "
        "an event that already happened to the recipient or to something they own."
    ),
    "Updates": (
        "Informational content the recipient is subscribed to but did not personally "
        "trigger: newsletters, digests, product or changelog announcements, policy changes, "
        "social network activity, shared document or comment notifications."
    ),
    "Promotional": (
        "Marketing or sales content: offers, discounts, coupons, ads, upsell or win-back "
        "campaigns, webinars being sold, or any message whose main purpose is to get the "
        "recipient to buy, upgrade, or click through to a product."
    ),
}

CATEGORIES: tuple[str, ...] = tuple(CATEGORY_CRITERIA)

DEFAULT_LABEL_NAMES: dict[str, str] = {
    OTP: "Categories/OTP",
    "Primary": "Categories/Primary",
    "Transaction": "Categories/Transactions",
    "Updates": "Categories/Updates",
    "Promotional": "Categories/Promotions",
    UNSURE: "Categories/Unsure",
}
