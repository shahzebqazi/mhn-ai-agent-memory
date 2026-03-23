# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, report it privately through
[GitHub Security Advisories](https://github.com/shahzebqazi/mhn-ai-agent-memory/security/advisories/new)
instead of opening a public issue.

You should receive an initial response within 48 hours.

## Scope

This library runs entirely in-process with no network calls unless you use
`OpenAIEncoder`. The main attack surface is malicious input to `load()` or the
MCP server. Both deserialize JSON; neither should ever use pickle.
