# Okanga MCP — Antigravity Agent Rules & Workflows

You are the development agent operating in Google Antigravity with the Okanga MCP server connected. Okanga provides token-efficient codebase intelligence via a compressed semantic index (CSU).

## Absolute Non-Negotiables
- **Never read a full file before `check_file_size`.** Outline → structure → snippet → full read.
- **Use the Okanga tool ladder** (cheapest first): `search_project` → `get_file_outline` → `read_structure` → `read_file_snippet` → full read.
- **Absolute paths only** in Okanga tool calls. Never relative, never `~`.
- **Never skip `scan_project`** at session start — a stale CSU cache produces wrong answers.
- **Don't guess** a file or symbol — `search_project` for it.
- **Max 3 full file reads per session.** Exhaust outline/structure/snippet first.
- **`suggest_context_for_question` before touching 2+ files.**
- **Checkpoint at 10 tool calls** with `get_session_token_summary`; flag if savings <50%.
- **Task completion metrics & Obsidian logging**: Whenever any Okanga tool is used in a session, upon completion of each task:
  1. Display token usage metrics (`get_session_token_summary`: tokens used, tokens saved, baseline without Okanga, savings %).
  2. Write a session metrics note to Obsidian Vault under `/Okanga/Metrics/` (`/Users/cosmovandalay/Library/Mobile Documents/com~apple~CloudDocs/Obsidian/Okanga/Metrics/YYYY-MM-DD-<project>-<session-slug>.md`).

---

## Mandatory Session Start Sequence

At the beginning of every session or when switching tasks:
1. `scan_project(project_path="/absolute/path/to/project")` — build/verify CSU cache is fresh.
2. `get_project_ontology(project_path="/absolute/path/to/project")` — single-call orientation covering entry points, top modules, dependency graph, circular dependency warnings, and external dependency counts.
   *(Or individually: `analyze_project` → `check_dependencies` → `diagnose_project`).*

Report in ≤10 lines:
- Cache status (fresh vs. rebuilt) and project shape (languages, module count).
- Any diagnostics/issues found, ranked by severity.
- The single most useful next action for the user's stated goal.

---

## Tool Ladder (Cheapest → Most Expensive)

Always start at the top and stop as soon as you have sufficient evidence:

| # | Tool | Purpose / When to Use |
|---|---|---|
| **0** | `get_file_tree(project_path)` | Whole-project directory layout walk (no cache needed, bound with `max_depth`/`max_nodes`). |
| **1** | `search_project(project_path, query)` | Find symbols, text, routes, imports, labels, URLs. |
| **2** | `get_file_outline(file_path)` | Lightweight outline: imports, class/function signatures, line numbers (no bodies). |
| **3** | `read_structure(file_path)` | Structural view with implementation bodies elided. |
| **4** | `read_file_snippet(file_path, start_line, end_line)` | Targeted range of implementation lines. |
| **5** | Full file read (`read`) | Last resort only. Call `check_file_size` first. Max 3 per session. |

---

## Multi-File & Investigation Patterns

### Multi-File Context Suggestion
Before touching 2+ files or modifying logic, call:
```python
suggest_context_for_question(project_path="/path", file_path="/path/to/anchor_file.py", question="Specific technical question with symbol names")
```
- Use the most relevant anchor file already identified.
- Formulate keyword-dense questions with real identifiers and imports.

### Impact / Blast-Radius Analysis (Before Modifying Code)
- **Primary:** `get_impact(project_path="/path", symbol="SymbolName")` — returns deterministic affected files tiered by risk (`WILL_BREAK`, `LIKELY_AFFECTED`, `MAY_NEED_TESTING`), traceable to imports.
- **Visualization:** `visualize_dependencies(project_path="/path", granularity="module"|"file")` — generates interactive force-directed graph at `{project_root}/.okanga/dependency-graph.html`.

### Module Onboarding
1. `get_file_tree` scoped to module directory if structure is unknown.
2. `suggest_context_for_question` with English description of goal.
3. `get_file_outline` on each candidate file.
4. `read_structure` on the 2–3 central files.
5. `check_dependencies` to identify collaborators.
6. `read_file_snippet` for specific logic.

---

## Token Efficiency & Checkpoints

- Run `get_session_token_summary` every 10 tool calls. If token savings vs. baseline naive full reads is `<50%`, tighten the investigation approach.
- Target savings: **≥85–95%** token reduction via CSU compression.
- Rescan (`scan_project force=true`) ONLY when: provider added/modified, `provider_registry.py`/`provider_base.py` changed, new tools added to `okanga_server.py`, or structural core changes in `src/core/`. Otherwise the cache is valid.

---

## Jira Ticketing Workflow (Atlassian MCP)

When bugs, feature gaps, refactoring needs, or test gaps are discovered:
1. **Rule:** Propose first. Never create tickets without explicit user approval.
2. **Issue Type:** `Type: Task` only (OKAN and JPT projects reject `Bug`/`Story`).
3. **Project Routing:**
   - `OKAN` — Okanga product features, bugs, enhancements.
   - `JPT` — Personal / non-Okanga tasks.
4. **Categories (Labels):** `bug`, `tech-debt`, `feature`, `test`, `perf`, `docs`, `provider`, `csu`, `dx`.
5. **Template:**
```
Summary:      <imperative, specific>
Project:      OKAN | JPT
Type:         Task
Labels:       okanga, <category>
Priority:     <High|Medium|Low, justified>

Description:
  <what & where — file:symbol if known>

Acceptance criteria:
  - <testable outcome 1>
  - <testable outcome 2>
```

---

## Task Completion & Obsidian Metrics Logging Protocol

Whenever any Okanga tool is used in a session, upon completion of each task:

### 1. In-Chat Summary
- Display the token summary table from `get_session_token_summary`:
  - Tokens used
  - Tokens saved
  - Baseline without Okanga
  - Savings %
  - Tool breakdown table

### 2. Obsidian Metrics Note
Create or update a markdown note in:
`/Users/cosmovandalay/Library/Mobile Documents/com~apple~CloudDocs/Obsidian/Okanga/Metrics/YYYY-MM-DD-<project>-<session-slug>.md`

Format:
```markdown
---
session: <Descriptive Session Name>
date: YYYY-MM-DD
project: <Project Name>
project_path: <Absolute Path>
branch: <Git Branch>
primary_language: <Language>
languages_deep_tier:
  - <Language 1>
  - <Language 2>
indexed_files: <Count>
tags:
  - okanga
  - metrics
  - <topic-tags>
---

# <Descriptive Session Name>

## Codebase Summary
- Path: `<project_path>`
- Branch: `<branch>` at `<short_commit>`
- Primary language: <language>
- Deep-tier languages: <languages>
- Total indexed files: <count>
- Module highlights: <key modules touched>

## Task Summary
- <Bullet points of work completed and verified>

## Token Usage
| Metric | Value |
|---|---:|
| Tokens used | <count> |
| Tokens saved | <count> |
| Baseline without Okanga | <count> |
| Savings | <percentage>% |

| Tracked tool | Calls | Used | Saved |
|---|---:|---:|---:|
| `search_project` | <n> | <used> | <saved> |
| `read_file_snippet` | <n> | <used> | <saved> |
| ... | ... | ... | ... |

## Efficiency And Verification
- <Verification commands run, test counts, assertions validated>
```

---

## Boundaries & Non-Okanga Tools

Okanga is strictly read/analyze. Use native Antigravity capabilities for:
- Writing/editing files (`edit`, `write`)
- Terminal / Shell execution (`bash`, `run_build_or_test`)
- Git operations (status, diff, branch, commit)
- Browser verification / UI inspection
