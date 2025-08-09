# Website ↔︎ API Parity: Gaps Across Grok, ChatGPT, and Claude

This document compares **Grok (xAI)**, **ChatGPT (OpenAI)**, and **Claude (Anthropic)** on *website vs. API parity*, *tooling*, and *context/limits*.  
It is adapted from a deep-dive analysis for **Grok Projects**.  
Last updated: 2025-08-09.

---

| Area | **Grok (xAI)** | **ChatGPT (OpenAI)** | **Claude (Anthropic)** | What breaks parity in practice |
|---|---|---|---|---|
| **Context window** | Docs list **256K** context (Grok-4, a.k.a. `grok-4-0709`) with differential pricing beyond 128K. | GPT-5 docs list **400K** context; unified model family via new Responses API. | Claude models (Sonnet/Opus) support large windows; tool use docs focus on model choice rather than a single universal limit. | Vendors sometimes ship **different context caps** (or pricing tiers) between UI and API; third-party hosts may enforce **smaller caps** (e.g., 128K). |
| **Built-in web search** | **Live Search** is a first-class feature; billed **per thousand sources**. Web UI emphasizes search; API exposes function calling to wire your own tools too. | **Responses API** includes an official **web search tool** and **computer-use** via Operator model; ChatGPT UI also has search. | Tool-use supports retrieval and **computer-use** (desktop control) with beta headers; web app has capabilities that require explicit flags/API wiring. | **Tool availability & defaults differ**: the web app may auto-enable search/computer tools while the API requires explicit setup or billing. |
| **Tool calling / function calling** | **OpenAI-compatible** function calling; JSON-schema tools and loops. Web UI often has extra scripted behaviors. | Mature **function/tool calling** in Responses and Assistants APIs. Web and API are close but not identical in orchestration. | **Tool use** supports parallel tools, computer use, bash/text-editor tools; requires explicit setup and sometimes beta headers. | Web UIs often include **hidden system prompts & guardrails**; APIs expose raw tools but not the same presets, so outputs diverge. |
| **Streaming & agent loops** | API supports tool calls; you build your own **agent loop** (as Grok Projects does). UI may feel “smarter” due to proprietary loops. | Responses API + Agents SDK push toward **agentic behavior** similar to ChatGPT’s UI. | Anthropic docs show agent loops for computer-use; not all web behaviors are portable to API. | **Agent orchestration differences**: website runs **provider-managed loops**, API requires you to implement them, leading to divergence. |
| **Pricing gotchas** | Live Search priced separately; **higher-context** pricing beyond 128K differs from base rates. | New tool costs for search/computer-use; may not mirror UI bundling. | Some tools gated behind beta usage; enterprise plans may alter defaults. | **Billing & quotas** impact parity—web plans bundle features; APIs often **unbundle**. |
| **Observed polish & features** | Competitive reasoning/coding; fewer team/workspace niceties vs. ChatGPT. | Widest feature set & ecosystem; rapid API evolution (Responses, Agents). | Strong safety/tooling posture; powerful computer-use, but more header/config complexity. | Parity breaks when **features ship UI-first** and **API later** (or behind flags). |

---

## TL;DR — What Vendors Should Guarantee

1. **Same model, same results.** Calling the same model ID via API with equivalent tools should produce **the same quality and formatting** as the website.  
2. **Tool parity by default.** Web-search, retrieval, computer-use: the **same tools** enabled in the web UI should be **exposed and documented** in the API with identical behavior.  
3. **Transparent presets.** Publish the **exact system prompts / policies / routing** used in the website so devs can reproduce them.  
4. **Stable caps & pricing.** Context windows and tool charges should match **UI claims** and not silently differ across tiers or hosting environments.

---

*For implementation details, see [GrokProjects on GitHub](https://github.com/russell0/GrokProjects).*

