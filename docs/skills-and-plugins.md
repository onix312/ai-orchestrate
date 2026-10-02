# Skills and plugins for ai-orchestrate + ChatGPT Desktop

Reviewed 2 October 2026. This is a small, task-focused shortlist for the Python orchestrator, its HTML/JS dashboard, and local web apps such as PrintFlow—not a recommendation to install every extension in a marketplace.

## Start with these official plugins

The links in the dashboard open the plugin's install flow in ChatGPT Desktop/Codex. They do **not** install anything silently. Review the requested access and authenticate only if you want that integration. After installing a plugin, start a new chat before relying on its skills or tools.

| Plugin | Why it fits | Scope / caution |
| --- | --- | --- |
| [Build Web Data Visualization](https://github.com/openai/plugins/tree/main/plugins/build-web-data-visualization) | The most “wow” fit here: builds interactive dashboards, charts, maps, Gantt/diagrams, Three.js/WebGL visuals, and can test/export reports or slides. Could help turn supplied run/token data into an operations cockpit. | It does not connect to ai-orchestrate logs automatically. Give Codex the relevant data explicitly, and keep secrets out of exported datasets. |
| [Game Studio](https://github.com/openai/plugins/tree/main/plugins/game-studio) | For a genuinely fun prototype: design and ship a playable browser game with guided 2D/3D, asset, and playtesting workflows. | A creative side quest rather than a dependency for this Python orchestrator. |
| [Build Web Apps](https://github.com/openai/plugins/tree/main/plugins/build-web-apps) | Includes `frontend-app-builder` and `frontend-testing-debugging`, useful for dashboard and local web-app work. | The bundle also includes React, shadcn, Stripe, and Supabase skills; use those only when that stack is actually present. This repository is Python + static HTML/JS. |
| [GitHub](https://github.com/openai/plugins/tree/main/plugins/github) | Brings issues, pull requests, checks, and the `gh-address-comments` skill into a ChatGPT Desktop task. | ai-orchestrate already uses `gh` and has its own guarded PR/merge path. Prefer that path for merges; the plugin is most useful for reading/triaging work in a separate Codex chat. The plugin documents that its OAuth client is public; review requested access and keep your user tokens private. |
| [Codex Security](https://github.com/openai/plugins/tree/main/plugins/codex-security) | Adds a dedicated security-scan workflow for authorized repositories and changes. | Use as an additional signal, inspect each finding, and keep the normal tests/review/merge gates. Do not scan a repository you are not authorized to share with the service. |
| [OpenAI Developers](https://github.com/openai/plugins/tree/main/plugins/openai-developers) | Bundles current OpenAI/Codex documentation workflows, useful when maintaining the orchestrator's model/API integrations. | Prefer this over asking the model to guess about changing OpenAI APIs. |
| [Figma](https://github.com/openai/plugins/tree/main/plugins/figma) | Helps translate a real Figma file, components, and design tokens into UI work. | Optional—install only if the source design is actually in Figma. |

The official plugin directory and supported surfaces are documented in [Codex Plugins](https://developers.openai.com/codex/plugins). OpenAI's old `openai/skills` repository is marked deprecated; use the current [OpenAI Plugins repository](https://github.com/openai/plugins) and the current docs instead of copying old installer instructions.

## Browser and desktop testing

1. **Local web app:** start with the desktop app's built-in browser and `@Browser` for screenshots, page inspection, and UI checks. It is usually simpler than installing another server.
2. **Windows desktop UI:** the **Computer Use** plugin can see and operate approved apps. It runs in the foreground on Windows, so keep the target visible and expect mouse/keyboard control. EEA support was added in the June 2026 rollout, but availability can still depend on account/workspace settings. Computer Use cannot automate ChatGPT itself; use the supported new-chat handoff instead. See the [Computer Use guide](https://developers.openai.com/codex/app/computer-use) and [availability notes](https://developers.openai.com/codex/changelog/).
3. **Repeatable browser automation:** Microsoft's [Playwright MCP](https://github.com/microsoft/playwright-mcp) is an optional MCP server for structured browser interaction and automated UI flows. It is not required for ordinary visual QA; enable it only when browser automation is part of the workflow and review the tools it exposes.

## Built-in new-chat handoff

The **↗ Новый чат ChatGPT** button in the task composer uses ChatGPT Desktop's documented `codex://new` deep link. It opens a **new local Codex chat**, selects the validated project folder, and pre-fills the task and configured checks. It never presses Send. Review the prompt and send it yourself.

This handoff is separate from an ai-orchestrate run: it does not create the orchestrator's worktree, execute its checks/reviewer, or enable its merge gate. If you use it to edit code, choose an isolated worktree in Codex when available and review the diff before applying it. Long prompts are copied to the clipboard and the chat opens with the project selected, so paste with Ctrl+V.

## Installation and safety

- Install from the official Codex/ChatGPT plugin directory or the links inside the dashboard. The directory supports installing skills and MCP-backed tools; connecting a service can grant read or write access.
- `codex://skills` opens skill management. Skills provide task instructions; plugins package skills and optional connectors/MCP tools.
- Do not bulk-install community skill packs. A skill is executable guidance, and MCP servers may run local code or access external data. Inspect the manifest, `SKILL.md`, scripts, requested scopes, network destinations, and write actions first.
- Keep GitHub write actions, merges, deployments, email sends, and destructive operations behind explicit user confirmation. The orchestrator's existing confirmation and verification gates remain the source of truth for work it manages.

## Sources

- [OpenAI Codex skills](https://developers.openai.com/codex/skills)
- [OpenAI Codex plugins](https://developers.openai.com/codex/plugins)
- [ChatGPT Desktop deep links and new chats](https://developers.openai.com/codex/app/commands)
- [OpenAI plugin examples](https://github.com/openai/plugins)
- [Microsoft Playwright MCP](https://github.com/microsoft/playwright-mcp)
