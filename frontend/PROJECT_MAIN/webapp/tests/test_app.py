"""End-to-end test of CloudClear (run: python -m pytest -q webapp/tests  or  python webapp/tests/test_app.py)."""
import os, re, sys, tempfile, sqlite3

TMP = tempfile.mkdtemp()
os.environ["CLOUDCLEAR_INSTANCE"] = TMP
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import app as webapp  # noqa: E402
import inference  # noqa: E402

app = webapp.app
app.config["TESTING"] = True


def csrf(client, path):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def post(client, path, data, page=None, **kw):
    data = dict(data, csrf_token=csrf(client, page or path))
    return client.post(path, data=data, **kw)


def db():
    con = sqlite3.connect(webapp.DB_PATH); con.row_factory = sqlite3.Row; return con


def test_full_flow():
    c = app.test_client()
    # public pages
    for p in ("/", "/about", "/login", "/admin/login", "/register"):
        assert c.get(p).status_code == 200, p
    # admin seeded with hashed password
    a = db().execute("SELECT * FROM users WHERE username='admin'").fetchone()
    assert a["role"] == "admin" and a["password_hash"].startswith(("scrypt:", "pbkdf2:")) and "Admin@123" not in a["password_hash"]

    # protected pages blocked when anonymous
    for p in ("/dashboard", "/predict", "/history", "/profile", "/admin", "/admin/users", "/admin/records", "/admin/activity"):
        r = c.get(p); assert r.status_code == 302 and "login" in r.headers["Location"], p
    assert c.get("/api/my-stats").status_code == 401

    # CSRF required
    assert c.post("/register", data={"name": "X", "email": "x@y.zz", "password": "abc12345", "confirm": "abc12345"}).status_code == 400

    # validation
    r = post(c, "/register", {"name": "A", "email": "bad", "password": "short", "confirm": "nope"})
    assert r.status_code == 400 and "valid email" in r.get_data(as_text=True)

    # registration -> PENDING; extra role field ignored
    r = post(c, "/register", {"name": "Asha Rao", "email": "asha@example.com", "password": "Cloud1234", "confirm": "Cloud1234", "role": "admin", "status": "APPROVED"})
    assert r.status_code == 200 and "PENDING" in r.get_data(as_text=True)
    u = db().execute("SELECT * FROM users WHERE email='asha@example.com'").fetchone()
    assert u["role"] == "user" and u["status"] == "PENDING"
    assert post(c, "/register", {"name": "Dup", "email": "asha@example.com", "password": "Cloud1234", "confirm": "Cloud1234"}).status_code == 400

    # pending user cannot log in
    r = post(c, "/login", {"email": "asha@example.com", "password": "Cloud1234"})
    assert r.status_code == 403 and "pending" in r.get_data(as_text=True).lower()
    assert c.get("/dashboard").status_code == 302
    # wrong password / user cannot use admin login
    assert post(c, "/login", {"email": "asha@example.com", "password": "wrong1234"}).status_code == 401
    assert post(c, "/admin/login", {"username": "asha@example.com", "password": "Cloud1234"}).status_code == 401
    assert post(c, "/admin/login", {"username": "admin", "password": "wrong"}).status_code == 401

    # second user to reject
    post(c, "/register", {"name": "Ravi K", "email": "ravi@example.com", "password": "Radar5678", "confirm": "Radar5678"})
    rid = db().execute("SELECT id FROM users WHERE email='ravi@example.com'").fetchone()[0]

    # admin login + approve / reject
    adm = app.test_client()
    r = post(adm, "/admin/login", {"username": "admin", "password": "Admin@123"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/admin")
    html = adm.get("/admin").get_data(as_text=True)
    assert "asha@example.com" in html and "Pending approvals (2)" in html
    assert adm.get("/dashboard").status_code == 403          # admin is not a user
    assert post(adm, f"/admin/users/{u['id']}/approve", {}, page="/admin/users").status_code == 302
    assert post(adm, f"/admin/users/{rid}/reject", {}, page="/admin/users").status_code == 302
    assert post(adm, f"/admin/users/{a['id']}/disable", {}, page="/admin/users").status_code == 404   # admin not modifiable
    assert db().execute("SELECT status FROM users WHERE id=?", (u["id"],)).fetchone()[0] == "APPROVED"

    # rejected user cannot log in
    c2 = app.test_client()
    r = post(c2, "/login", {"email": "ravi@example.com", "password": "Radar5678"})
    assert r.status_code == 403 and "rejected" in r.get_data(as_text=True)

    # approved user login -> dashboard; user cannot access admin
    r = post(c, "/login", {"email": "asha@example.com", "password": "Cloud1234"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")
    assert "Welcome, Asha Rao" in c.get("/dashboard").get_data(as_text=True)
    for p in ("/admin", "/admin/users", "/admin/records", "/admin/activity"):
        assert c.get(p).status_code == 403, p

    # core functionality: cloud removal on a real held-out test patch (with reference)
    samples = inference.sample_patches(); assert samples
    patch = samples[0]["patch"]
    r = post(c, "/predict", {"mode": "sample", "patch": patch, "tta": "on"})
    assert r.status_code == 302, r.get_data(as_text=True)[:500]
    uid = r.headers["Location"].rsplit("/", 1)[1]
    page = c.get(f"/result/{uid}").get_data(as_text=True)
    assert patch in page and "Cloud-free prediction" in page
    p = db().execute("SELECT * FROM predictions WHERE uid=?", (uid,)).fetchone()
    assert p["psnr_pred"] > p["psnr_input"], (p["psnr_input"], p["psnr_pred"])
    assert 0 <= p["cloud_cover"] <= 1 and 0 <= p["confidence"] <= 1
    tif = c.get(f"/result/{uid}/file/cloudfree.tif"); assert tif.status_code == 200 and tif.data[:2] in (b"II", b"MM")
    assert c.get(f"/result/{uid}/file/prediction.png").status_code == 200
    assert c.get(f"/result/{uid}/file/../app.db").status_code == 404
    print(f"sample {patch}: cloud {p['cloud_cover']:.3f}, PSNR {p['psnr_input']:.2f} -> {p['psnr_pred']:.2f}, conf {p['confidence']:.3f}, {p['runtime_ms']:.0f} ms")

    # upload path with real GeoTIFFs (no TTA) + invalid upload rejected
    s1, s2, ref = inference.sample_paths(samples[1]["patch"])
    with open(s1, "rb") as f1, open(s2, "rb") as f2:
        r = post(c, "/predict", {"mode": "upload", "s1": (f1, "s1.tif"), "s2": (f2, "my_cloudy.tif")}, page="/predict", content_type="multipart/form-data")
    assert r.status_code == 302, r.get_data(as_text=True)[:800]
    uid2 = r.headers["Location"].rsplit("/", 1)[1]
    assert db().execute("SELECT psnr_pred FROM predictions WHERE uid=?", (uid2,)).fetchone()[0] is None
    with open(s2, "rb") as f1, open(s2, "rb") as f2:   # wrong band count for S1
        r = post(c, "/predict", {"mode": "upload", "s1": (f1, "a.tif"), "s2": (f2, "b.tif")}, page="/predict", content_type="multipart/form-data")
    assert r.status_code == 400 and "2 bands" in r.get_data(as_text=True)
    import io
    r = post(c, "/predict", {"mode": "upload", "s1": (io.BytesIO(b"x"), "a.exe"), "s2": (io.BytesIO(b"y"), "b.tif")}, page="/predict", content_type="multipart/form-data")
    assert r.status_code == 400
    r = post(c, "/predict", {"mode": "sample", "patch": "../../etc/passwd"}); assert r.status_code == 400

    # history: own records only
    h = c.get("/history").get_data(as_text=True)
    assert patch in h and "my_cloudy.tif" in h and "prediction" in h
    stats = c.get("/api/my-stats").get_json(); assert stats["n"] == 2 and stats["n_ref"] == 1

    # second approved user cannot see first user's records
    post(c2, "/register", {"name": "Mei Lin", "email": "mei@example.com", "password": "Optic9999", "confirm": "Optic9999"})
    mid = db().execute("SELECT id FROM users WHERE email='mei@example.com'").fetchone()[0]
    post(adm, f"/admin/users/{mid}/approve", {}, page="/admin/users")
    post(c2, "/login", {"email": "mei@example.com", "password": "Optic9999"})
    assert c2.get(f"/result/{uid}").status_code == 404
    assert c2.get(f"/result/{uid}/file/cloudfree.tif").status_code == 404
    assert patch not in c2.get("/history").get_data(as_text=True)
    assert post(c2, f"/history/{uid}/delete", {}, page="/history").status_code == 404

    # admin sees all records + activity
    rec = adm.get("/admin/records").get_data(as_text=True); assert patch in rec and "asha@example.com" in rec
    assert adm.get(f"/result/{uid}").status_code == 200
    act = adm.get("/admin/activity").get_data(as_text=True)
    for k in ("register", "login_blocked", "user_approve", "user_reject", "prediction", "login_failed"):
        assert k in act, k

    # disabling an approved user kills the active session
    post(adm, f"/admin/users/{mid}/disable", {}, page="/admin/users")
    assert c2.get("/dashboard").status_code == 302

    # profile + password change
    assert post(c, "/profile", {"action": "password", "current": "Cloud1234", "new": "NewCloud99", "confirm": "NewCloud99"}).status_code == 302
    # logout then old password fails, new works
    post(c, "/logout", {}, page="/dashboard")
    assert c.get("/dashboard").status_code == 302
    assert post(c, "/login", {"email": "asha@example.com", "password": "Cloud1234"}).status_code == 401
    assert post(c, "/login", {"email": "asha@example.com", "password": "NewCloud99"}).status_code == 302
    # user deletes own record
    assert post(c, f"/history/{uid2}/delete", {}, page="/history").status_code == 302
    assert db().execute("SELECT COUNT(*) FROM predictions WHERE uid=?", (uid2,)).fetchone()[0] == 0
    # admin pages render with data
    for p in ("/admin", "/admin/users", "/admin/users?status=PENDING", f"/admin/records?user={u['id']}", f"/admin/activity?user={u['id']}&action=login"):
        assert adm.get(p).status_code == 200, p
    for p in ("/dashboard", "/predict", "/history", "/profile"):
        assert c.get(p).status_code == 200, p
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    test_full_flow()
