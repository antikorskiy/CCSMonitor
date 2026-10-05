# app.py
import sys
import tkinter as tk

from config import DB_PATH
from database import Database
from async_engine import DataEngine
from ui import CCSMonitorApp


def main():
    db = Database(DB_PATH)
    engine = DataEngine(db)
    engine.start()

    root = tk.Tk()
    app = CCSMonitorApp(root, db, engine)
    try:
        root.mainloop()
    finally:
        try:
            app._shutdown_tray()
        except Exception:
            pass
        engine.stop()
        db.close()


if __name__ == "__main__":
    main()