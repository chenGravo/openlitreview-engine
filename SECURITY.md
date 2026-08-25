# Security policy

## Supported version

Only the latest tagged release is supported. The private workspace must pin the public engine to
an audited commit SHA; do not run an unreviewed branch with model secrets.

## Reporting

Report suspected vulnerabilities privately to the repository owner. Do not open a public issue
that contains API keys, private task files, model responses, downloaded full text, or personal
data. Revoke a possibly exposed provider key before investigating the code path.

## Operating boundaries

- Keep the execution workspace private and enable GitHub secret scanning.
- Never submit personal, sensitive, confidential, patient, or secret material to Phase 0.
- Keep GitHub Actions billing at zero and provider balances within the approved combined ceiling.
- Only process full text whose access and cloud-processing rights are confirmed.
- Treat generated reviews as drafts requiring human academic review.

