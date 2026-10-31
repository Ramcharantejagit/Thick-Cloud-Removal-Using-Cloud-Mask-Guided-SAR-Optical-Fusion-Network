"""Create the SQLite database and the default admin account (admin / Admin@123). Safe to run repeatedly."""
import app

if __name__ == "__main__":
    app.init_db()
    print("Database ready:", app.DB_PATH)
