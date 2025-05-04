# core.py
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials


def generate_user_hash(rank_m1, size_m2):
    key = f"{rank_m1}-{size_m2}"
    return hashlib.sha256(key.encode()).hexdigest()


def convert_rank_to_note_m1(rank_m1):
    return 20.0 * (1.0 - (rank_m1 - 1) / 1798.0)


def convert_rank_to_note_m2(rank_m2, size):
    return 20.0 * (1.0 - (rank_m2 - 1) / (size - 1))


def get_google_sheet(sheet_id, json_keyfile_dict):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json_keyfile_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(sheet_id).sheet1


def collect_to_google_sheet(sheet, user_data):
    nom_las, rank_m1, rank_m2, size_m2, note_m1, note_m2, rang_souhaite = user_data

    user_hash = generate_user_hash(rank_m1, size_m2)
    rows = sheet.get_all_values()

    if not rows:
        header = [
            "Nom LAS", "Rang M1", "Rang M2", "Taille M2", "Note M1", "Note M2",
            "Hash", "Rang souhaite", "Timestamp"
        ]
        sheet.append_row(header)
    else:
        hashes = [row[6] for row in rows[1:] if len(row) > 6]
        if user_hash in hashes:
            return False, "DUPLICATE"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([
        nom_las, rank_m1, rank_m2, size_m2,
        note_m1, note_m2, user_hash, rang_souhaite, timestamp
    ])
    return True, "SUCCESS"


def get_dataframe_from_sheet(sheet):
    data = sheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    df.columns = df.columns.str.strip().str.lower()

    def to_float(x):
        try:
            return float(str(x).replace(",", "."))
        except:
            return np.nan

    df["note m1"] = df["note m1"].apply(to_float)
    df["note m2"] = df["note m2"].apply(to_float)
    df = df.dropna(subset=["note m1", "note m2"]).reset_index(drop=True)
    return df


def calculate_rho(df):
    if len(df) < 3:
        return None
    return np.corrcoef(df["note m1"], df["note m2"])[0, 1]
