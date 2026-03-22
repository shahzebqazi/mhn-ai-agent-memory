# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately via
[GitHub Security Advisories](https://github.com/shahzebqazi/mhn-ai-agent-memory/security/advisories/new)
rather than opening a public issue.

You should receive an initial response within 48 hours.

## Scope

This library runs entirely in-process with no network calls (unless using
`OpenAIEncoder`). The primary attack surface is malicious input to `load()`
or the MCP server. Both deserialize JSON — never pickle.
