# Enterprise API Gateway Security & Token Specification

## 1. Authentication & Bearer Tokens
All client communication through the external ingress gateway requires an RFC 6750 Bearer Token. Tokens are validated deterministically by the gateway before routing to internal microservices:
- Algorithm: Asymmetric RS256 with 2048-bit RSA keys rotated every 90 days.
- Claims Requirement: Must contain `iss` (Identity Provider URI), `sub` (User UUID), `aud` (Gateway Audience), and `exp` (Expiry).
- Expiration Policy: Access tokens expire strictly after 15 minutes (900 seconds). Refresh tokens possess a 7-day sliding session lifespan.
- Error Code `ERR_AUTH_EXPIRED`: Returned with HTTP status 401 when the access token timestamp has elapsed.

## 2. Rate Limiting & Denial of Service Protection
The gateway enforces a distributed Token Bucket rate limiter backed by Redis clusters:
- Default Quota: 100 requests per minute per IP address for unauthenticated public endpoints.
- Authenticated Quota: 2,500 requests per minute per verified Organization ID.
- Burst Capacity: Maximum burst limit allows +25% overhead for 5-second intervals.
- Error Code `ERR_RATE_LIMIT_EXCEEDED`: Returned with HTTP status 429 and `Retry-After` header when quota is exhausted.

## 3. Data Sanitization & OWASP Top 10 Safeguards
1. SQL Injection Prevention: Dynamic string concatenation inside database queries is strictly forbidden. All relational queries must execute via parameterized prepared statements.
2. Cross-Site Scripting (XSS): All response bodies set `Content-Security-Policy: default-src 'self'` and `X-Content-Type-Options: nosniff`.
3. Secret Scrubbing: Gateway middleware scans incoming payloads for AWS access keys, GitHub personal access tokens, and private keys, dropping flagged requests with HTTP 400 (`ERR_POTENTIAL_CREDENTIAL_LEAK`).
