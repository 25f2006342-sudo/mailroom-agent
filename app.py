import os, json, hashlib, re
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from decision import decide_batch
from storage import Store
from policy import validate_proposal, verify_receipt_signature

PROFILE = "ga5-mailroom-action-gate/v2"
MAX_BODY = int(os.getenv("MAX_BODY_BYTES", "2097152"))
DB_PATH = os.getenv("DATABASE_PATH", "/var/data/mailroom.sqlite3")
CALL_RE = re.compile(r"^[A-Za-z0-9._:-]{12,128}$")
ALLOWED = {"create_draft", "update_internal_record", "send_approved_notice",
           "request_confirmation", "quarantine_item", "no_action"}

app = FastAPI(title="GA5 Mailroom Action Gate", version="2.1.0")
store = Store(DB_PATH)

def canonical(obj: Any) -> bytes:
    # Canonical JSON adapter: sorted keys, compact separators, UTF-8, no ASCII escaping.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")

def sha256(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()

def json_error(status: int, message: str):
    return JSONResponse(status_code=status, content={"error": message})

@app.middleware("http")
async def limit_body(request: Request, call_next):
    if request.method == "POST":
        raw = await request.body()
        if len(raw) > MAX_BODY:
            return json_error(413, "request body too large")
        request.state.raw_body = raw
    response = await call_next(request)
    response.headers["Content-Type"] = "application/json"
    return response

@app.get("/health")
def health():
    return {"status": "ok", "profile": PROFILE}

@app.post("/")
async def endpoint(request: Request):
    try:
        body = json.loads(request.state.raw_body)
    except Exception:
        return json_error(400, "malformed JSON")
    if not isinstance(body, dict):
        return json_error(400, "JSON object required")
    try:
        if body.get("operation") == "propose":
            return await propose(body)
        if body.get("operation") == "commit":
            return await commit(body)
        return json_error(400, "unsupported operation")
    except ValueError as e:
    print(f"VALIDATION ERROR: {str(e)}", flush=True)
    return json_error(422, str(e))
    except Exception:
        return json_error(500, "internal processing error")

def validate_propose_envelope(body):
    required = {"profile", "operation", "evaluationId", "receiptVerifier", "corpus", "dossiers"}
    if set(body) != required:
        raise ValueError("propose envelope has missing or unknown fields")
    if body["profile"] != PROFILE or body["operation"] != "propose":
        raise ValueError("unsupported profile or operation")
    if not isinstance(body["evaluationId"], str) or not body["evaluationId"]:
        raise ValueError("evaluationId must be a non-empty string")
    rv = body["receiptVerifier"]
    if not isinstance(rv, dict) or set(rv) != {"algorithm", "publicKeyJwk"} or rv["algorithm"] != "Ed25519":
        raise ValueError("receiptVerifier must specify Ed25519 and publicKeyJwk")
    jwk = rv["publicKeyJwk"]
    if not isinstance(jwk, dict) or set(jwk) != {"kty", "crv", "x"}:
        raise ValueError("invalid Ed25519 publicKeyJwk")
    if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519" or not isinstance(jwk.get("x"), str):
        raise ValueError("invalid Ed25519 public key")
    corpus = body["corpus"]
    if not isinstance(corpus, dict) or set(corpus) != {"coreId", "auditId", "stableCount", "freshCount"}:
        raise ValueError("invalid corpus descriptor")
    if not isinstance(corpus["coreId"], str) or not isinstance(corpus["auditId"], str):
        raise ValueError("invalid corpus identifiers")
    if corpus["stableCount"] != 58 or corpus["freshCount"] != 6:
        raise ValueError("unexpected corpus counts")
    dossiers = body["dossiers"]
    if not isinstance(dossiers, list) or len(dossiers) != 64:
        raise ValueError("dossiers must contain 64 cases")
    seen, stable, fresh = set(), 0, 0
    for d in dossiers:
        if not isinstance(d, dict) or set(d) != {"dossierId", "partition", "receivedAt", "mailbox", "objective", "sources"}:
            raise ValueError("invalid dossier schema")
        did = d["dossierId"]
        if not isinstance(did, str) or not did or did in seen:
            raise ValueError("missing or duplicate dossierId")
        seen.add(did)
        if d["partition"] == "stable_core": stable += 1
        elif d["partition"] == "fresh_audit": fresh += 1
        else: raise ValueError("invalid dossier partition")
        if not all(isinstance(d[k], str) for k in ("receivedAt", "mailbox", "objective")):
            raise ValueError("invalid dossier metadata")
        if not isinstance(d["sources"], list): raise ValueError("sources must be an array")
        for s in d["sources"]:
            if not isinstance(s, dict) or set(s) != {"sourceId", "kind", "provenance", "title", "lines"}:
                raise ValueError("invalid source schema")
            if not all(isinstance(s[k], str) for k in ("sourceId", "kind", "provenance", "title")):
                raise ValueError("invalid source metadata")
            if not isinstance(s["lines"], list): raise ValueError("source lines must be an array")
            for line in s["lines"]:
                if not isinstance(line, dict) or set(line) != {"lineId", "text"}:
                    raise ValueError("invalid line schema")
                if not all(isinstance(line[k], str) for k in ("lineId", "text")):
                    raise ValueError("invalid line values")
    if stable != 58 or fresh != 6:
        raise ValueError("expected 58 stable_core and 6 fresh_audit dossiers")

async def propose(body):
    validate_propose_envelope(body)
    evaluation_id = body["evaluationId"]
    input_digest = sha256(body)
    existing = store.get_evaluation(evaluation_id)
    if existing:
        if existing["request_hash"] != input_digest:
            return json_error(409, "evaluationId reused with changed content")
        return JSONResponse(content=json.loads(existing["propose_response"]))

    proposals_by_id = {}
    missing = []
    for dossier in body["dossiers"]:
        fp = sha256(dossier)
        cached = store.get_decision(dossier["dossierId"], fp)
        if cached is None: missing.append((dossier, fp))
        else: proposals_by_id[dossier["dossierId"]] = json.loads(cached)

    if missing:
        generated = await decide_batch([d for d, _ in missing])
        if len(generated) != len(missing):
            raise ValueError("model returned incorrect proposal count")
        for (dossier, fp), proposal in zip(missing, generated):
            validate_proposal(dossier, proposal, ALLOWED)
            proposal["callId"] = stable_call_id(dossier["dossierId"], fp)
            validate_proposal(dossier, proposal, ALLOWED)
            store.put_decision(dossier["dossierId"], fp, proposal)
            proposals_by_id[dossier["dossierId"]] = proposal

    proposals = []
    for dossier in body["dossiers"]:
        proposal = proposals_by_id[dossier["dossierId"]]
        validate_proposal(dossier, proposal, ALLOWED)
        if not isinstance(proposal.get("callId"), str) or not CALL_RE.fullmatch(proposal["callId"]):
            raise ValueError("invalid callId")
        proposals.append(proposal)

    response = {"profile": PROFILE, "evaluationId": evaluation_id,
                "status": "awaiting_receipts", "inputDigest": input_digest,
                "proposals": proposals}
    store.put_evaluation(evaluation_id, input_digest, response,
                         body["receiptVerifier"]["publicKeyJwk"])
    return JSONResponse(content=response)

def stable_call_id(dossier_id, fingerprint):
    return "mr:" + hashlib.sha256((dossier_id + ":" + fingerprint).encode()).hexdigest()[:32]

async def commit(body):
    required = {"profile", "operation", "evaluationId", "inputDigest", "receipts"}
    if not isinstance(body, dict) or set(body) != required:
        raise ValueError("commit envelope has missing or unknown fields")
    if body["profile"] != PROFILE or body["operation"] != "commit":
        raise ValueError("unsupported profile or operation")
    eid, input_digest, receipts = body["evaluationId"], body["inputDigest"], body["receipts"]
    if not isinstance(eid, str) or not eid or not isinstance(input_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", input_digest):
        raise ValueError("invalid evaluationId or inputDigest")
    if not isinstance(receipts, list):
        raise ValueError("receipts must be an array")
    ev = store.get_evaluation(eid)
    if not ev:
        return json_error(409, "unknown evaluation")
    if input_digest != ev["request_hash"]:
        return json_error(409, "inputDigest does not match evaluation")
    commit_hash = sha256(body)
    replay = store.get_commit(eid, commit_hash)
    if replay:
        return JSONResponse(content=json.loads(replay))
    prior_commit = store.get_any_commit(eid)
    if prior_commit:
        return json_error(409, "evaluation already committed with different content")

    proposals = json.loads(ev["propose_response"])["proposals"]
    by_call = {p["callId"]: p for p in proposals}
    if len(receipts) != len(proposals):
        raise ValueError("receipt count must match proposal count")
    seen_calls, seen_receipt_ids = set(), set()
    outcomes, persist_rows = [], []
    jwk = json.loads(ev["verification_material"])

    # Verify the entire batch before writing any receipt/effect.
    for receipt in receipts:
        fields = {"dossierId", "callId", "action", "accepted",
                  "proposalDigest", "receiptId", "receiptSignature"}
        if not isinstance(receipt, dict) or set(receipt) != fields:
            raise ValueError("invalid receipt schema")
        if not all(isinstance(receipt[k], str) and receipt[k] for k in
                   ("dossierId", "callId", "action", "proposalDigest", "receiptId", "receiptSignature")):
            raise ValueError("invalid receipt field")
        if type(receipt["accepted"]) is not bool:
            raise ValueError("accepted must be boolean")
        p = by_call.get(receipt["callId"])
        if not p or receipt["callId"] in seen_calls:
            return json_error(422, "unknown or duplicate callId")
        seen_calls.add(receipt["callId"])
        if receipt["dossierId"] != p["dossierId"] or receipt["action"] != p["action"]:
            return json_error(422, "receipt scope does not match proposal")
        expected_pd = sha256(p)
        if receipt["proposalDigest"] != expected_pd:
            return json_error(422, "proposalDigest mismatch")
        if receipt["receiptId"] in seen_receipt_ids:
            return json_error(422, "duplicate receiptId")
        seen_receipt_ids.add(receipt["receiptId"])
        if not verify_receipt_signature(receipt, jwk):
            return json_error(422, "invalid Ed25519 receipt signature")
        status = "executed" if receipt["accepted"] else "rejected"
        outcome = {"dossierId": p["dossierId"], "callId": p["callId"],
                   "action": p["action"], "proposalDigest": expected_pd,
                   "receiptId": receipt["receiptId"], "status": status}
        outcomes.append(outcome)
        persist_rows.append((receipt, outcome))

    if seen_calls != set(by_call):
        raise ValueError("receipt batch is incomplete")
    response = {"profile": PROFILE, "evaluationId": eid, "status": "completed",
                "inputDigest": input_digest, "outcomes": outcomes}
    store.persist_commit(eid, commit_hash, persist_rows, response)
    return JSONResponse(content=response)
