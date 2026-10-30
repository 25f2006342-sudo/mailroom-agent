import sqlite3, json, os, hashlib
class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        with self.db() as c:
            c.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS decisions(
              dossier_id TEXT NOT NULL, fingerprint TEXT NOT NULL, proposal TEXT NOT NULL,
              PRIMARY KEY(dossier_id, fingerprint));
            CREATE TABLE IF NOT EXISTS evaluations(
              evaluation_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
              propose_response TEXT NOT NULL, verification_material TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS commits(
              evaluation_id TEXT NOT NULL, request_hash TEXT NOT NULL, response TEXT NOT NULL,
              PRIMARY KEY(evaluation_id, request_hash));
            CREATE TABLE IF NOT EXISTS receipts(
              evaluation_id TEXT NOT NULL, receipt_hash TEXT NOT NULL,
              receipt_json TEXT NOT NULL, outcome TEXT NOT NULL,
              PRIMARY KEY(evaluation_id, receipt_hash));
            """)
    def db(self):
        c=sqlite3.connect(self.path, timeout=25)
        c.row_factory=sqlite3.Row
        return c
    def get_evaluation(self,eid):
        with self.db() as c:
            r=c.execute("SELECT * FROM evaluations WHERE evaluation_id=?",(eid,)).fetchone()
            return dict(r) if r else None
    def put_evaluation(self,eid,rh,response,key):
        with self.db() as c:
            c.execute("INSERT INTO evaluations VALUES(?,?,?,?)",
                (eid,rh,json.dumps(response,separators=(",",":"),ensure_ascii=False),
                 json.dumps(key,separators=(",",":"))))
    def get_decision(self,did,fp):
        with self.db() as c:
            r=c.execute("SELECT proposal FROM decisions WHERE dossier_id=? AND fingerprint=?",(did,fp)).fetchone()
            return r["proposal"] if r else None
    def put_decision(self,did,fp,p):
        with self.db() as c:
            c.execute("INSERT OR IGNORE INTO decisions VALUES(?,?,?)",
                (did,fp,json.dumps(p,separators=(",",":"),ensure_ascii=False)))
    def get_commit(self,eid,rh):
        with self.db() as c:
            r=c.execute("SELECT response FROM commits WHERE evaluation_id=? AND request_hash=?",(eid,rh)).fetchone()
            return r["response"] if r else None
    def get_any_commit(self,eid):
        with self.db() as c:
            r=c.execute("SELECT request_hash,response FROM commits WHERE evaluation_id=? LIMIT 1",(eid,)).fetchone()
            return dict(r) if r else None
    def persist_commit(self,eid,commit_hash,rows,response):
        # One transaction: either all verified receipts + terminal response persist, or none.
        with self.db() as c:
            for receipt, outcome in rows:
                receipt_hash = hashlib.sha256(json.dumps(
                    receipt, sort_keys=True, separators=(",",":"), ensure_ascii=False
                ).encode()).hexdigest()
                c.execute("INSERT INTO receipts VALUES(?,?,?,?)",
                    (eid, receipt_hash,
                     json.dumps(receipt,separators=(",",":"),ensure_ascii=False),
                     json.dumps(outcome,separators=(",",":"),ensure_ascii=False)))
            c.execute("INSERT INTO commits VALUES(?,?,?)",
                (eid, commit_hash, json.dumps(response,separators=(",",":"),ensure_ascii=False)))
