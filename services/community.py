import hashlib, os
from datetime import datetime, timezone

try:
    from google.cloud import firestore as _fs
    _db = _fs.AsyncClient(project=os.environ.get("GCP_PROJECT"))
except Exception:
    _db = None

COLLECTION = "rumour_checks"

async def record_check(
    claim_hash: str, verdict: str, summary: str, harm_severity: str
) -> None:
    if _db is None:
        return
    try:
        ref = _db.collection(COLLECTION).document(claim_hash)
        await ref.set({
            "verdict": verdict,
            "summary": summary,
            "harm_severity": harm_severity,
            "check_count": _fs.Increment(1),
            "last_seen": datetime.now(timezone.utc),
        }, merge=True)
    except Exception:
        pass

async def get_trending(limit: int = 5) -> list[dict]:
    if _db is None:
        return []
    try:
        stream = (
            _db.collection(COLLECTION)
            .where("check_count", ">=", 2)
            .order_by("check_count", direction=_fs.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        results = []
        async for doc in stream:
            d = doc.to_dict()
            results.append({
                "summary": d.get("summary", ""),
                "verdict": d.get("verdict", ""),
                "harm_severity": d.get("harm_severity", "NONE"),
                "check_count": d.get("check_count", 0),
            })
        return results
    except Exception:
        return []
