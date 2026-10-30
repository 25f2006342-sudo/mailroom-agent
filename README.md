# GA5 Mailroom Action Gate v2

## Implemented from the supplied propose contract

- Exact profile and propose envelope validation.
- 58 stable + 6 fresh dossier count and partition validation.
- SHA-256 content fingerprints and persisted content-based decision cache.
- Stable deterministic `callId` per dossier ID + dossier fingerprint.
- Model sees dossier data only; no tools, keys, or receipt verifier.
- Action allowlist, typed target shape, object payload, and evidence lineId validation.
- Exact evaluation replay and changed-content 409.
- Ed25519 public JWK is persisted per evaluation.

## Still required before grader submission

The provided excerpt does not include:
1. Exact `commit` request/receipt fields.
2. Exact bytes/string that the Ed25519 signature covers.
3. Receipt-to-proposal digest/scope matching fields.
4. Canonical JSON/hash rules (the implementation currently uses sorted-key compact UTF-8 JSON).
5. Per-action target and payload schemas / allowed target kinds.

The commit endpoint deliberately fails closed. Do not submit until these are specified and implemented. Do not use a check to discover these details.

## Local run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
set OPENAI_API_KEY=your_key
set DATABASE_PATH=./mailroom.sqlite3
uvicorn app:app --reload --port 8000
```

Deploy with Render using `render.yaml`, persistent disk, and `OPENAI_API_KEY`.


## Commit adapter added (v2.1)

The commit envelope and terminal outcome now match the supplied fields. It checks
evaluation ID and input digest, requires exactly one receipt per proposal, matches
dossierId/callId/action/proposalDigest, rejects duplicate call IDs and receipt IDs,
verifies Ed25519 signatures, and persists all receipts plus the final response in
one SQLite transaction. `accepted=true` maps to `executed`; `false` maps to `rejected`.

**Signature serialization assumption:** because the supplied excerpt specifies
"base64 Ed25519 signature" but not the exact signed bytes, the implementation
currently verifies compact sorted-key UTF-8 JSON of each receipt with
`receiptSignature` omitted. Likewise, proposalDigest is SHA-256 over the complete
proposal object including callId, using compact sorted-key UTF-8 JSON. Confirm these
two serialization details against the full frozen hashing/signature rules before
using a paid evaluation check.
