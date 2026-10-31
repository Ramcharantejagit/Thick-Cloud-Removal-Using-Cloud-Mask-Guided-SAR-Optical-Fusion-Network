"""CloudClear — web application for SAR-optical thick-cloud removal (PSF-CR project)."""
import functools
import os
import re
import secrets
import shutil
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import inference

BASE = os.path.dirname(os.path.abspath(__file__))
INSTANCE = os.environ.get("CLOUDCLEAR_INSTANCE", os.path.join(BASE, "instance"))
DB_PATH = os.path.join(INSTANCE, "app.db")
RESULTS = os.path.join(INSTANCE, "results")
UPLOADS = os.path.join(INSTANCE, "uploads")
ADMIN_USERNAME, ADMIN_DEFAULT_PASSWORD = "admin", "Admin@123"
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
STATUSES = ("PENDING", "APPROVED", "REJECTED", "DISABLED")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    username TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED', 'DISABLED')),
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    reviewed_by INTEGER,
    last_login TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('sample', 'upload')),
    input_name TEXT NOT NULL,
    cloud_cover REAL, confidence REAL, confidence_cloud REAL, mean_uncertainty REAL, tta INTEGER,
    psnr_input REAL, psnr_pred REAL, sam_input REAL, sam_pred REAL, mae_input REAL, mae_pred REAL,
    runtime_ms REAL, model TEXT
);
CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    actor TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    ip TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_act_user ON activity(user_id);
"""


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def init_db():
    os.makedirs(INSTANCE, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(UPLOADS, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    if not con.execute("SELECT 1 FROM users WHERE username = ? AND role = 'admin'", (ADMIN_USERNAME,)).fetchone():
        con.execute("INSERT INTO users (name, username, password_hash, role, status, created_at) VALUES (?, ?, ?, 'admin', 'APPROVED', ?)",
                    ("Administrator", ADMIN_USERNAME, generate_password_hash(ADMIN_DEFAULT_PASSWORD), now()))
        con.execute("INSERT INTO activity (actor, action, detail, created_at) VALUES ('system', 'admin_created', 'Default admin account created', ?)", (now(),))
    con.commit()
    con.close()


def log(action, detail="", user=None):
    user = user or g.get("user")
    get_db().execute("INSERT INTO activity (user_id, actor, action, detail, ip, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                     (user["id"] if user else None, (user["username"] or user["email"]) if user else "anonymous",
                      action, detail[:500], request.remote_addr, now()))
    get_db().commit()


def _secret_key():
    if os.environ.get("CLOUDCLEAR_SECRET_KEY"):
        return os.environ["CLOUDCLEAR_SECRET_KEY"]
    os.makedirs(INSTANCE, exist_ok=True)
    p = os.path.join(INSTANCE, "secret_key")
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write(secrets.token_hex(32))
        os.chmod(p, 0o600)
    return open(p).read().strip()


# ---------------------------------------------------------------- app
app = Flask(__name__)
app.config.update(
    SECRET_KEY=_secret_key(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("CLOUDCLEAR_SECURE_COOKIE", "0") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
    MAX_CONTENT_LENGTH=60 * 1024 * 1024,
)
init_db()


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                                               "script-src 'self'; font-src 'self'; frame-ancestors 'none'")
    if g.get("user"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------- CSRF + session user
def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def load_user_and_check_csrf():
    g.user = None
    uid = session.get("uid")
    if uid:
        u = get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        # session is only valid while the account still exists, keeps its role and is approved
        if u and u["role"] == session.get("role") and u["status"] == "APPROVED":
            g.user = u
        else:
            session.clear()
    if request.method == "POST":
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not token or not secrets.compare_digest(token, session.get("csrf", "")):
            abort(400, description="Invalid or missing CSRF token. Reload the page and try again.")


def login_required(role):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            if g.user is None:
                if request.path.startswith("/api/"):
                    return jsonify(error="authentication required"), 401
                flash("Please log in to continue.", "warning")
                return redirect(url_for("admin_login" if role == "admin" else "login", next=request.path))
            if g.user["role"] != role:
                abort(403)
            return fn(*a, **kw)
        return wrapper
    return deco


def _start_session(u):
    session.clear()                       # prevent session fixation
    session.permanent = True
    session["uid"], session["role"] = u["id"], u["role"]
    csrf_token()
    get_db().execute("UPDATE users SET last_login = ? WHERE id = ?", (now(), u["id"]))
    get_db().commit()


def _safe_next(default):
    nxt = request.args.get("next", "")
    return nxt if nxt.startswith("/") and not nxt.startswith("//") else default


# ---------------------------------------------------------------- public pages
@app.route("/")
def home():
    return render_template("home.html", bench=inference.benchmark(), cfg=inference.MODEL_CFG)


@app.route("/about")
def about():
    return render_template("about.html", bench=inference.benchmark(), cfg=inference.MODEL_CFG)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw, pw2 = request.form.get("password", ""), request.form.get("confirm", "")
        errors = []
        if not 2 <= len(name) <= 80:
            errors.append("Name must be 2–80 characters.")
        if not EMAIL_RE.match(email) or len(email) > 120:
            errors.append("Enter a valid email address.")
        if len(pw) < 8 or not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
            errors.append("Password must be at least 8 characters and contain letters and digits.")
        if pw != pw2:
            errors.append("Passwords do not match.")
        if not errors and get_db().execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            errors.append("An account with this email already exists.")
        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("register.html", name=name, email=email), 400
        # role and status are fixed server-side: public registration can never create an admin
        cur = get_db().execute("INSERT INTO users (name, email, password_hash, role, status, created_at) VALUES (?, ?, ?, 'user', 'PENDING', ?)",
                               (name, email, generate_password_hash(pw), now()))
        get_db().commit()
        u = get_db().execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        log("register", f"New registration {email} (PENDING)", user=u)
        return render_template("pending.html", email=email, just_registered=True)
    return render_template("register.html")


def _auth(identifier_col, identifier, password, role):
    u = get_db().execute(f"SELECT * FROM users WHERE {identifier_col} = ? AND role = ?", (identifier, role)).fetchone()
    if u is None or not check_password_hash(u["password_hash"], password):
        return None
    return u


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user and g.user["role"] == "user":
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        u = _auth("email", email, request.form.get("password", ""), "user")
        if u is None:
            get_db().execute("INSERT INTO activity (actor, action, detail, ip, created_at) VALUES (?, 'login_failed', 'user login', ?, ?)",
                             (email[:120], request.remote_addr, now()))
            get_db().commit()
            flash("Invalid email or password.", "danger")
            return render_template("login.html", email=email), 401
        if u["status"] == "PENDING":
            log("login_blocked", "Account pending approval", user=u)
            return render_template("pending.html", email=email), 403
        if u["status"] in ("REJECTED", "DISABLED"):
            log("login_blocked", f"Account {u['status'].lower()}", user=u)
            flash("Your account has been rejected." if u["status"] == "REJECTED" else "Your account has been disabled by the administrator.", "danger")
            return render_template("login.html", email=email), 403
        _start_session(u)
        log("login", "User login", user=u)
        return redirect(_safe_next(url_for("dashboard")))
    return render_template("login.html")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if g.user and g.user["role"] == "admin":
        return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        u = _auth("username", username, request.form.get("password", ""), "admin")
        if u is None or u["status"] != "APPROVED":
            get_db().execute("INSERT INTO activity (actor, action, detail, ip, created_at) VALUES (?, 'login_failed', 'admin login', ?, ?)",
                             (username[:120], request.remote_addr, now()))
            get_db().commit()
            flash("Invalid admin credentials.", "danger")
            return render_template("admin_login.html", username=username), 401
        _start_session(u)
        log("login", "Admin login", user=u)
        return redirect(_safe_next(url_for("admin_dashboard")))
    return render_template("admin_login.html")


@app.route("/logout", methods=["POST"])
def logout():
    if g.user:
        log("logout", "")
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ---------------------------------------------------------------- user area
def _user_stats(user_id):
    db = get_db()
    rows = db.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at", (user_id,)).fetchall()
    with_ref = [r for r in rows if r["psnr_pred"] is not None]
    avg = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {
        "n": len(rows),
        "n_ref": len(with_ref),
        "avg_cloud": avg([r["cloud_cover"] for r in rows]),
        "avg_conf": avg([r["confidence"] for r in rows if r["confidence"] is not None]),
        "avg_gain": avg([r["psnr_pred"] - r["psnr_input"] for r in with_ref]),
        "avg_psnr": avg([r["psnr_pred"] for r in with_ref]),
        "series": [{"id": r["uid"], "t": r["created_at"], "name": r["input_name"], "cloud": r["cloud_cover"],
                    "psnr_in": r["psnr_input"], "psnr_out": r["psnr_pred"], "conf": r["confidence"]} for r in rows],
    }


@app.route("/dashboard")
@login_required("user")
def dashboard():
    recent = get_db().execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 5", (g.user["id"],)).fetchall()
    return render_template("dashboard.html", stats=_user_stats(g.user["id"]), recent=recent, bench=inference.benchmark(), cfg=inference.MODEL_CFG)


@app.route("/predict", methods=["GET", "POST"])
@login_required("user")
def predict():
    samples = inference.sample_patches()
    if request.method == "GET":
        return render_template("predict.html", samples=samples)
    mode = request.form.get("mode")
    tta = request.form.get("tta") == "on"
    pid = uuid.uuid4().hex
    out_dir = os.path.join(RESULTS, pid)
    tmp = os.path.join(UPLOADS, pid)
    try:
        if mode == "sample":
            patch = request.form.get("patch", "")
            if patch not in {s["patch"] for s in samples}:
                flash("Choose a valid sample patch.", "danger")
                return render_template("predict.html", samples=samples), 400
            s1, s2, ref = inference.sample_paths(patch)
            name, source = patch, "sample"
        elif mode == "upload":
            files = {k: request.files.get(k) for k in ("s1", "s2", "ref")}
            if not files["s1"] or not files["s1"].filename or not files["s2"] or not files["s2"].filename:
                flash("Upload both the Sentinel-1 and the cloudy Sentinel-2 GeoTIFF.", "danger")
                return render_template("predict.html", samples=samples), 400
            os.makedirs(tmp, exist_ok=True)
            paths = {}
            for k, f in files.items():
                if f and f.filename:
                    fn = secure_filename(f.filename)
                    if not fn.lower().endswith((".tif", ".tiff")):
                        flash("Only GeoTIFF files (.tif / .tiff) are accepted.", "danger")
                        return render_template("predict.html", samples=samples), 400
                    paths[k] = os.path.join(tmp, f"{k}.tif")
                    f.save(paths[k])
            s1, s2, ref = paths["s1"], paths["s2"], paths.get("ref")
            name, source = secure_filename(files["s2"].filename)[:100] or "upload", "upload"
        else:
            abort(400)
        try:
            res = inference.run(s1, s2, out_dir, ref_path=ref, tta=tta)
        except inference.InputError as e:
            shutil.rmtree(out_dir, ignore_errors=True)
            flash(str(e), "danger")
            log("prediction_rejected", f"{source}: {e}")
            return render_template("predict.html", samples=samples), 400
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    db = get_db()
    db.execute("""INSERT INTO predictions (uid, user_id, created_at, source, input_name, cloud_cover, confidence, confidence_cloud,
                  mean_uncertainty, tta, psnr_input, psnr_pred, sam_input, sam_pred, mae_input, mae_pred, runtime_ms, model)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
               (pid, g.user["id"], now(), source, name, res["cloud_cover"], res["confidence"], res["confidence_cloud"],
                res["mean_uncertainty"], int(res["tta"]), res["psnr_input"], res["psnr_pred"], res["sam_input"], res["sam_pred"],
                res["mae_input"], res["mae_pred"], res["runtime_ms"], inference.MODEL_CFG["display_name"]))
    db.commit()
    log("prediction", f"{source} {name}: cloud {res['cloud_cover'] * 100:.1f}%" +
        (f", PSNR {res['psnr_input']:.2f}→{res['psnr_pred']:.2f} dB" if res["psnr_pred"] is not None else ""))
    return redirect(url_for("result", uid=pid))


def _get_prediction(uid):
    if not re.fullmatch(r"[0-9a-f]{32}", uid):
        abort(404)
    r = get_db().execute("SELECT p.*, u.name AS owner_name, u.email AS owner_email FROM predictions p JOIN users u ON u.id = p.user_id WHERE p.uid = ?",
                         (uid,)).fetchone()
    if r is None:
        abort(404)
    if g.user["role"] != "admin" and r["user_id"] != g.user["id"]:
        abort(404)                         # do not reveal other users' records
    return r


def any_role(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if g.user is None:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return fn(*a, **kw)
    return wrapper


@app.route("/result/<uid>")
@any_role
def result(uid):
    r = _get_prediction(uid)
    d = os.path.join(RESULTS, uid)
    imgs = [k for k in ("cloudy", "sar", "mask", "prediction", "uncertainty", "reference") if os.path.exists(os.path.join(d, k + ".png"))]
    return render_template("result.html", r=r, imgs=imgs)


@app.route("/result/<uid>/file/<name>")
@any_role
def result_file(uid, name):
    r = _get_prediction(uid)
    allowed = {f"{k}.png" for k in ("cloudy", "sar", "mask", "prediction", "uncertainty", "reference")} | {"cloudfree.tif"}
    if name not in allowed:
        abort(404)
    if name.endswith(".tif"):
        log("download", f"{r['input_name']} cloudfree.tif")
    return send_from_directory(os.path.join(RESULTS, uid), name, as_attachment=name.endswith(".tif"),
                               download_name=f"{r['input_name']}_cloudfree.tif" if name.endswith(".tif") else None)


@app.route("/history")
@login_required("user")
def history():
    rows = get_db().execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC", (g.user["id"],)).fetchall()
    acts = get_db().execute("SELECT * FROM activity WHERE user_id = ? ORDER BY id DESC LIMIT 100", (g.user["id"],)).fetchall()
    return render_template("history.html", rows=rows, acts=acts)


@app.route("/history/<uid>/delete", methods=["POST"])
@login_required("user")
def delete_prediction(uid):
    r = _get_prediction(uid)
    get_db().execute("DELETE FROM predictions WHERE uid = ?", (uid,))
    get_db().commit()
    shutil.rmtree(os.path.join(RESULTS, uid), ignore_errors=True)
    log("prediction_deleted", r["input_name"])
    flash("Record deleted.", "info")
    return redirect(url_for("history"))


@app.route("/api/my-stats")
@login_required("user")
def api_my_stats():
    return jsonify(_user_stats(g.user["id"]))


@app.route("/profile", methods=["GET", "POST"])
@login_required("user")
def profile():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "name":
            name = request.form.get("name", "").strip()
            if not 2 <= len(name) <= 80:
                flash("Name must be 2–80 characters.", "danger")
            else:
                get_db().execute("UPDATE users SET name = ? WHERE id = ?", (name, g.user["id"]))
                get_db().commit()
                log("profile_update", "Name changed")
                flash("Profile updated.", "success")
        elif action == "password":
            cur, new, conf = request.form.get("current", ""), request.form.get("new", ""), request.form.get("confirm", "")
            if not check_password_hash(g.user["password_hash"], cur):
                flash("Current password is incorrect.", "danger")
            elif len(new) < 8 or not re.search(r"[A-Za-z]", new) or not re.search(r"\d", new):
                flash("New password must be at least 8 characters and contain letters and digits.", "danger")
            elif new != conf:
                flash("New passwords do not match.", "danger")
            else:
                get_db().execute("UPDATE users SET password_hash = ? WHERE id = ?", (generate_password_hash(new), g.user["id"]))
                get_db().commit()
                log("password_change", "")
                flash("Password changed.", "success")
        return redirect(url_for("profile"))
    stats = _user_stats(g.user["id"])
    return render_template("profile.html", stats=stats)


# ---------------------------------------------------------------- admin area
@app.route("/admin")
@login_required("admin")
def admin_dashboard():
    db = get_db()
    counts = {s: db.execute("SELECT COUNT(*) FROM users WHERE role='user' AND status=?", (s,)).fetchone()[0] for s in STATUSES}
    npred = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    per_day = db.execute("SELECT substr(created_at,1,10) d, COUNT(*) n FROM predictions GROUP BY d ORDER BY d").fetchall()
    per_user = db.execute("""SELECT u.name, COUNT(p.id) n FROM users u JOIN predictions p ON p.user_id=u.id
                             GROUP BY u.id ORDER BY n DESC LIMIT 10""").fetchall()
    gain = db.execute("SELECT AVG(psnr_pred-psnr_input), COUNT(psnr_pred) FROM predictions WHERE psnr_pred IS NOT NULL").fetchone()
    pending = db.execute("SELECT * FROM users WHERE role='user' AND status='PENDING' ORDER BY created_at").fetchall()
    acts = db.execute("SELECT * FROM activity ORDER BY id DESC LIMIT 12").fetchall()
    return render_template("admin_dashboard.html", counts=counts, npred=npred, per_day=[dict(r) for r in per_day],
                           per_user=[dict(r) for r in per_user], gain=gain, pending=pending, acts=acts)


@app.route("/admin/users")
@login_required("admin")
def admin_users():
    status = request.args.get("status", "")
    q = "SELECT u.*, (SELECT COUNT(*) FROM predictions p WHERE p.user_id=u.id) npred FROM users u WHERE role='user'"
    args = []
    if status in STATUSES:
        q += " AND status = ?"
        args.append(status)
    users = get_db().execute(q + " ORDER BY created_at DESC", args).fetchall()
    return render_template("admin_users.html", users=users, status=status)


@app.route("/admin/users/<int:user_id>/<action>", methods=["POST"])
@login_required("admin")
def admin_user_action(user_id, action):
    db = get_db()
    u = db.execute("SELECT * FROM users WHERE id = ? AND role = 'user'", (user_id,)).fetchone()
    if u is None:
        abort(404)                          # admins cannot be modified through this endpoint
    new = {"approve": "APPROVED", "reject": "REJECTED", "disable": "DISABLED", "enable": "APPROVED"}.get(action)
    if new:
        db.execute("UPDATE users SET status = ?, reviewed_at = ?, reviewed_by = ? WHERE id = ?", (new, now(), g.user["id"], user_id))
        db.commit()
        log(f"user_{action}", f"{u['email']}: {u['status']} → {new}")
        flash(f"{u['email']} is now {new}.", "success")
    elif action == "delete":
        for (pid,) in db.execute("SELECT uid FROM predictions WHERE user_id = ?", (user_id,)).fetchall():
            shutil.rmtree(os.path.join(RESULTS, pid), ignore_errors=True)
        db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        db.commit()
        log("user_delete", u["email"])
        flash(f"{u['email']} deleted.", "info")
    else:
        abort(400)
    return redirect(request.form.get("back") if (request.form.get("back") or "").startswith("/admin") else url_for("admin_users"))


@app.route("/admin/records")
@login_required("admin")
def admin_records():
    user_id = request.args.get("user", type=int)
    q = "SELECT p.*, u.name owner_name, u.email owner_email FROM predictions p JOIN users u ON u.id = p.user_id"
    args = []
    if user_id:
        q += " WHERE p.user_id = ?"
        args.append(user_id)
    rows = get_db().execute(q + " ORDER BY p.id DESC LIMIT 500", args).fetchall()
    users = get_db().execute("SELECT id, name, email FROM users WHERE role='user' ORDER BY name").fetchall()
    return render_template("admin_records.html", rows=rows, users=users, user_id=user_id)


@app.route("/admin/activity")
@login_required("admin")
def admin_activity():
    user_id = request.args.get("user", type=int)
    action = request.args.get("action", "")
    q, args = "SELECT * FROM activity WHERE 1=1", []
    if user_id:
        q += " AND user_id = ?"
        args.append(user_id)
    if action and re.fullmatch(r"[a-z_]{1,40}", action):
        q += " AND action = ?"
        args.append(action)
    acts = get_db().execute(q + " ORDER BY id DESC LIMIT 500", args).fetchall()
    actions = [r[0] for r in get_db().execute("SELECT DISTINCT action FROM activity ORDER BY action").fetchall()]
    users = get_db().execute("SELECT id, name, email FROM users WHERE role='user' ORDER BY name").fetchall()
    return render_template("admin_activity.html", acts=acts, actions=actions, users=users, user_id=user_id, action=action)


# ---------------------------------------------------------------- errors
@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(413)
def http_error(e):
    code = getattr(e, "code", 500)
    msg = {403: "You do not have permission to view this page.", 404: "Page not found.",
           413: "Upload too large (max 60 MB in total)."}.get(code, getattr(e, "description", "Bad request."))
    if request.path.startswith("/api/"):
        return jsonify(error=msg), code
    return render_template("error.html", code=code, msg=msg), code


@app.template_filter("pct")
def pct(v):
    return "–" if v is None else f"{v * 100:.1f}%"


@app.template_filter("num")
def num(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
