# CloudClear web application

A Flask web app for the PSF-CR cloud-removal project. It runs the saved best model from `../models/best_model/` on Sentinel-1 + cloudy Sentinel-2 GeoTIFFs. It uses the same preprocessing as `code/02_Best_Model_Prediction.ipynb`, read from `preprocessing.json`.

## Features
* **Public pages**: Home and About, both showing the real test results from `models/model_evaluation_results/`; User Login; Admin Login; Register.
* **Registration**: name, email and password. New accounts are `PENDING` and cannot log in until an admin approves them; a pending message is shown. `REJECTED` and `DISABLED` users cannot log in. Registration always creates role `user`.
* **Admin** (default account `admin` / `Admin@123`, created automatically):
  * Approve, reject, disable, enable or delete users.
  * Dashboard with user and run statistics.
  * View all cloud-removal records and the full activity log, with filters.
* **User dashboard**:
  * Personal metrics: number of runs, average cloud cover, confidence and PSNR gain.
  * Chart.js charts: your PSNR before and after each run, the cloud cover of your inputs, and the model benchmark.
  * Profile page (change name or password) and logout.
* **Remove clouds**:
  * Pick one of the 254 held-out test patches (each has a cloud-free reference, so PSNR, SAM and MAE are computed), or upload your own 256×256 GeoTIFFs: S1 with 2 bands (VV, VH in dB) and S2 L1C with 13 bands, plus an optional reference.
  * The output is:
    * a georeferenced cloud-free GeoTIFF;
    * PNG previews;
    * the cloud mask;
    * an uncertainty map and a confidence score from 8-view test-time augmentation (TTA).
* **History**:
  * Users see only their own runs and their own activity, and can delete their own runs.
  * Admins see everything.

## Security
* Passwords are hashed with Werkzeug scrypt.
* Sessions use a signed cookie: HttpOnly, SameSite=Lax, 2-hour lifetime, and cleared on login to prevent session fixation.
* Every request reloads the user from the database. Disabling or rejecting a user, or deleting their account, ends their session immediately.
* Pages and APIs enforce role and approval checks on the server.
* Every POST requires a CSRF token.
* All SQL is parameterised.
* Input validation:
  * name, email format and password policy;
  * uploads must be `.tif`, at most 60 MB, with the right band count, size, co-registration and value ranges;
  * a sample patch is accepted only if it is on the whitelist.
* Result files are served only from a fixed set of filenames. A user who requests another user's record gets 404.
* Security headers are set, including a strict Content-Security-Policy with no inline scripts and X-Frame-Options DENY.
* Bootstrap, Bootstrap Icons and Chart.js are bundled locally, so the app works offline.

## Setup and run
Run these from `PROJECT_MAIN/webapp/`. The app needs `../models/best_model/`. It also needs `../data/raw/sen12mscr_style/` and `../data/processed/test_index.csv` for the sample patches; without them, upload still works.
```bash
pip install -r requirements.txt
python init_db.py        # creates instance/app.db and the default admin (also done automatically on start)
python app.py            # http://127.0.0.1:5000
```
Optional environment variables:
* `CLOUDCLEAR_SECRET_KEY`: the session key. If unset, a random key is generated once and stored in `instance/secret_key`.
* `CLOUDCLEAR_SECURE_COOKIE=1`: send cookies only over HTTPS.
* `HOST` and `PORT`: where the server listens.
* `CLOUDCLEAR_INSTANCE`: the folder for the database and results.

For production, run behind a WSGI server, for example `gunicorn -w 1 -b 0.0.0.0:8000 app:app`. Use one worker per CPU-heavy process.

**Admin login:** username `admin`, password `Admin@123`. Change it in production; there is no admin password-change page.

## Tests
```bash
cd PROJECT_MAIN && python -m pytest -q webapp/tests
```
The test covers this flow in order:
1. Public pages load.
2. The admin password is stored hashed.
3. Anonymous users are blocked from user pages, admin pages and the API.
4. Requests without a CSRF token are rejected, and invalid registration input is rejected.
5. Registration creates a `PENDING` account. A forged `role=admin` field is ignored, and duplicate emails are refused.
6. A pending user cannot log in.
7. The admin logs in, approves one user and rejects another. The rejected user cannot log in.
8. The approved user logs in and reaches the dashboard, and cannot open admin pages.
9. A real sample run must improve PSNR. A real upload run works, and bad uploads (wrong bands, wrong file type, path traversal) are rejected.
10. History shows only the user's own records. Another user gets 404 on them.
11. The admin can see all records and the activity log.
12. Disabling a user ends their active session.
13. Changing the password works, and logout works.
