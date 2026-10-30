import base64, json, re
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

def validate_proposal(dossier, proposal, allowed):
    keys = {"dossierId", "action", "target", "payload", "evidence"}
    if not isinstance(proposal, dict) or set(proposal) not in (keys, keys | {"callId"}):
        raise ValueError("model proposal has invalid fields")
    if proposal.get("dossierId") != dossier["dossierId"]:
        raise ValueError("proposal dossierId mismatch")
    if proposal.get("action") not in allowed:
        raise ValueError("unsupported action")
    target = proposal.get("target")
    if target is not None:
        if not isinstance(target, dict) or set(target) != {"kind", "id"}:
            raise ValueError("target must be null or {kind,id}")
        if not all(isinstance(target[k], str) and target[k] for k in ("kind", "id")):
            raise ValueError("invalid typed target")
    if not isinstance(proposal.get("payload"), dict):
        raise ValueError("payload must be an object")
    evidence = proposal.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(isinstance(x, str) for x in evidence):
        raise ValueError("evidence must contain lineId strings")
    valid_ids = {line["lineId"] for source in dossier["sources"] for line in source["lines"]}
    if len(set(evidence)) != len(evidence) or any(x not in valid_ids for x in evidence):
        raise ValueError("evidence contains duplicate or unknown lineId")

def _b64url_decode(value):
    if not isinstance(value, str):
        raise ValueError("base64url value must be string")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

def verify_receipt_signature(receipt, public_jwk):
    """
    Current explicit assumption: Ed25519 signs compact, sorted-key UTF-8 JSON
    of the receipt object with receiptSignature removed. If the exam specifies
    a different signed-message construction, replace this function accordingly.
    """
    try:
        if public_jwk.get("kty") != "OKP" or public_jwk.get("crv") != "Ed25519":
            return False
        raw_key = _b64url_decode(public_jwk["x"])
        if len(raw_key) != 32:
            return False
        signature = base64.b64decode(receipt["receiptSignature"], validate=True)
        if len(signature) != 64:
            return False
        signed_fields = {k: v for k, v in receipt.items() if k != "receiptSignature"}
        message = json.dumps(signed_fields, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        Ed25519PublicKey.from_public_bytes(raw_key).verify(signature, message)
        return True
    except Exception:
        return False
