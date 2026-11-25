# AGENTS.md

Guidance for anyone changing this repo, people and coding agents alike.
[README.md](README.md) explains what the library does and how to call it.

geo-cloudify converts geospatial data to cloud-native formats and checks the results.
Other projects call it as a library.
Its first user is the ingest step of [servir-data-catalog](https://github.com/SERVIR-AI/servir-data-catalog), but it must stay usable by any project.

## Setup and checks

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest -q
```

Run these checks before opening a pull request.
`uv run ruff check --fix` and `uv run ruff format` fix most of what the lint and format checks report.
CI runs the same checks after `uv sync --locked`, which fails if `uv.lock` does not match `pyproject.toml`, so commit both together.

## Scope

- **Convert and check, nothing else.**
  The library knows nothing about STAC, Source Coop, consent or any catalog.
  The calling project decides which files make up an output, where the output goes and what gets published.
- **Only what a caller needs now.**
  COG is the only format for now.
  Add a format, function or option only when a calling project needs it.
  Removed code stays in git history; bring it back when a use case needs it.
- **Settings are arguments.**
  Library code reads no config file and no environment variables, so any project can call it.

## Rules

- **Keep the data as it is.**
  Never set a nodata value, a CRS or any other field the input does not have, and never drop one it has.
  Refuse input the library cannot keep, naming the file and the field, instead of changing it silently.
  Accept only lossless compression.
- **Same input, same bytes.**
  The same input and arguments give byte-identical output with the same library versions, so callers can compare outputs by sha256.
  Never write timestamps, run IDs or other per-run values into an output.
- **Large files.**
  Read and write in blocks or windows; never hold a whole raster in memory.
  Write each output under a temporary name and move it into place only when it is complete, so a failure leaves nothing behind.
- **Errors and logging.**
  Every error a public function raises subclasses one base error of this package, so a caller catches one type.
  Wrap GDAL and rasterio errors in it and keep the original as the cause.
  Errors and problems name the file and the field they are about.
  Library code logs through `logging`, never prints and never configures logging.
- **Releases are tags.**
  Callers pin a release tag such as `v0.1.0`, never a branch.
- **Add a dependency only with a stated reason,** through `pyproject.toml`, and commit the updated `uv.lock`.
  Install one from git only at a release tag, never a branch, so a later `uv lock` still works once the branch is deleted.
- **Never commit** data files, credentials or tokens.
  Tests write the small rasters they need; large sample data stays out of git.

## Code changes

- Understand the goal, constraints, and existing behavior before changing code.
- Read existing code first.
  Reuse current files, utilities, patterns, and conventions.
- Prefer editing existing files over creating new ones.
- Make the smallest complete change that solves the current problem.
- Keep one theme per pull request, and put refactors in their own.
  Split a pull request with more than about 400 changed lines of non-test code.
- Do not add abstractions, configurability, or complexity for hypothetical future requirements.
- Keep every setting, path, and derived value defined in one source of truth.
  If changing a value requires editing multiple places, improve the design.
- Validate data at system boundaries (arguments from callers, files, URLs).
  Refuse input you cannot use; never drop fields or fill in data silently.
- Do not place secrets, tokens, credentials, or sensitive values in source code, logs, errors, fixtures, or examples.
- Clean up files, connections, locks, and other resources on success, failure, and cancellation.
- Define each function above the code that calls it, so a module reads from its helpers up to its public functions.

## Naming

- Name the domain thing, not its role in a sentence: `tile`, `output`, `nodata`, `problem`, not `thing`, `kind`, `data` or `done`.
- Use one word per concept everywhere: a **problem** is a message saying what is wrong, and an **error** is an exception.
- A function says what it does to which domain thing, such as `read_header` or `merge_to_cog`, or names the value it returns.
- A public name is unique across modules, so importing it by its bare name is never ambiguous.
- Anything callers should not use starts with `_`, and a public function never returns a private type.
- Keep short names to a scope of a few lines, such as a comprehension or a loop index.

## Testing

- Tests should earn their place.
  Prefer fewer high-value tests over many low-value tests.
- Each test should cover a distinct behavior, failure mode, edge case, or regression risk.
- Do not write tests that verify standard library, GDAL or framework behavior.
- Cover important edge cases: empty inputs, zero values, null/None, boundary values, off-by-one.
- Use specific assertions with meaningful failure messages.
- Prefer behavior-focused tests over implementation-detail tests.
- Tests never use the network.

## Writing conventions

- Plain language and short sentences.
- One sentence per line in Markdown.
- Diagrams in Mermaid, so GitHub renders them and diffs stay readable.
- Keep commit messages to one short line.
- PR descriptions should include only what changed, how to validate, and any risk or follow-up.
  Do not make them verbose or promotional.

## Documentation

- When a change likely requires doc updates, call that out and ask first.
- Do not add boilerplate file headers such as `@file`, `@author`, or `@date`.
- Comments should explain why or intent, not restate what the code does.
- Describe each behavior in one place: the docstring, with the README showing how to call it.
- Do not manually edit `CHANGELOG.md` or files marked as generated.

## Reviews

- When reviewing code, be proactive and broad.
- Surface issues across correctness, reliability, security, testing, and maintainability.
- State the verdict first, then findings.
- List must-fix findings first, each with its file and line and the rule it breaks.
  Skip anything a linter would catch.

## Completion

- Distinguish completed, partial, and blocked work.
  Do not describe partial or blocked work as complete.
- When required validation has not run, say so and provide the commands.
- Do not claim work is production-ready until checks have passed.

## Agent-specific rules

- Do not run test, lint, build, or dev server commands.
  Provide the exact command for the user to run.
- Do not run git state-changing commands (commit, push, merge, rebase, reset).
  Provide the command and wait for the user.
- Do not create, update, merge, or close pull requests.
  Provide the PR title, description, and commands.
- Do not run destructive actions (deleting files, force-pushing, overwriting user work).
  Explain the impact, provide the command, and get explicit approval.
- Read-only inspection is allowed when needed to understand the task.
