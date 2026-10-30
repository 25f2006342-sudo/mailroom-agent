        return json_error(409, "unknown evaluation")

    if input_digest != evaluation["request_hash"]:
        return json_error(409, "inputDigest does not match evaluation")

    commit_hash = sha256(body)
    replay = store.get_commit(evaluation_id, commit_hash)
    if replay:
        return JSONResponse(content=json.loads(replay))

    prior_commit = store.get_any_commit(evaluation_id)
    if prior_commit:
        return json_error(409, "evaluation already committed with different content")

    proposals = json.loads(evaluation["propose_response"])["proposals"]
    by_call = {proposal["callId"]: proposal for proposal in proposals}

    if len(receipts) != len(proposals):
        raise ValueError("receipt count must match proposal count")

    seen_calls = set()
    seen_receipt_ids = set()
    outcomes = []
    persist_rows = []
    jwk = json.loads(evaluation["verification_material"])

    # Verify the complete batch before persisting any receipt/outcome.
    for receipt in receipts:
        fields = {
            "dossierId",
            "callId",
            "action",
            "accepted",
            "proposalDigest",
            "receiptId",
            "receiptSignature",
        }

        if not isinstance(receipt, dict) or set(receipt) != fields:
            raise ValueError("invalid receipt schema")

        string_fields = (
            "dossierId",
            "callId",
            "action",
            "proposalDigest",
            "receiptId",
            "receiptSignature",
        )

        if not all(
            isinstance(receipt[key], str) and receipt[key]
            for key in string_fields
        ):
            raise ValueError("invalid receipt field")

        if type(receipt["accepted"]) is not bool:
            raise ValueError("accepted must be boolean")

        proposal = by_call.get(receipt["callId"])
        if not proposal or receipt["callId"] in seen_calls:
            return json_error(422, "unknown or duplicate callId")

        seen_calls.add(receipt["callId"])

        if (
            receipt["dossierId"] != proposal["dossierId"]
            or receipt["action"] != proposal["action"]
        ):
            return json_error(422, "receipt scope does not match proposal")

        expected_digest = sha256(proposal)
        if receipt["proposalDigest"] != expected_digest:
            return json_error(422, "proposalDigest mismatch")

        if receipt["receiptId"] in seen_receipt_ids:
            return json_error(422, "duplicate receiptId")
        seen_receipt_ids.add(receipt["receiptId"])

        if not verify_receipt_signature(receipt, jwk):
            return json_error(422, "invalid Ed25519 receipt signature")

        status = "executed" if receipt["accepted"] else "rejected"
        outcome = {
            "dossierId": proposal["dossierId"],
            "callId": proposal["callId"],
            "action": proposal["action"],
            "proposalDigest": expected_digest,
            "receiptId": receipt["receiptId"],
            "status": status,
        }

        outcomes.append(outcome)
        persist_rows.append((receipt, outcome))

    if seen_calls != set(by_call):
        raise ValueError("receipt batch is incomplete")

    response = {
        "profile": PROFILE,
        "evaluationId": evaluation_id,
        "status": "completed",
        "inputDigest": input_digest,
        "outcomes": outcomes,
    }

    store.persist_commit(evaluation_id, commit_hash, persist_rows, response)
    return JSONResponse(content=response)
