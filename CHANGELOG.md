# Changelog

All notable changes to this project are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/);
each section corresponds to a git version tag (which is also the release
published to PyPI). Entries are commit subjects and PR titles, verbatim.

## [0.1.4] - 2026-06-04

- translators: split extraction from rendering (add *_from_spec cores) ([#4](https://github.com/thorwhalen/aw/pull/4))

## [0.1.3] - 2026-05-18

- ci: migrate old CI to uv (wads-migrate setup-to-pyproject + ci-to-uv); drop legacy setup.cfg/setup.py
- Merge pull request [#2](https://github.com/thorwhalen/aw/pull/2) from thorwhalen/claude/document-repo-structure-ztb33
- Add OPENAI_API_KEY to CI workflow
- Enhance claude_desktop_config to auto-create missing config files and sections; update test cases for improved coverage
- Enhance Claude Desktop configuration to auto-create missing files and sections; update setup requirements for testing
- Enhance CI configuration and update project metadata
- Refactor utility functions and routing for agentic workflows
- Add comprehensive test suite for aw package
- gitignore and ci
- 0.1.1:
- 0.0.2:
- Add AI Agentic Workflow section to README

### Fixed

- fix(ci): wire OPENAI_API_KEY via [tool.wads.ci.env.test_envvars]
