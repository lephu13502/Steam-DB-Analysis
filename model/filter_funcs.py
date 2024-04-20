import pandas as pd


def filter_restricted_age(rec_table: pd, threshold: float = 16):
    return rec_table[rec_table["required_age"] < threshold]

def filter_games_only(rec_table: pd):
    return rec_table[rec_table["type"] == "game"]

def filter_positive_apps(rec_table: pd, threshold: float = 6):
    return rec_table[rec_table["review_score"] >= threshold]

def filter_affordable_apps(rec_table: pd, threshold: float = 25):
    return rec_table[rec_table["final_price_usd"] <= threshold]   # $25 = 636.000đ

def filter_free_apps(rec_table: pd):
    return rec_table[rec_table["final_price_usd"] == 0]

def filter_on_sale_apps(rec_table: pd):
    return rec_table[(rec_table["discount_percent"] > 0) & (rec_table["final_price_usd"] == 0)]

def filter_light_storage_games(rec_table: pd, threshold: float = 5, including_null: bool = True):
    if including_null:
        return rec_table[(rec_table["storage_gb"] <= threshold) | rec_table["storage_gb"].isna()]
    else:
        return rec_table[rec_table["storage_gb"] <= threshold]

def filter_lightweight_games(rec_table: pd, threshold: float = 4, including_null: bool = True):
    if including_null:
        return rec_table[(rec_table["memory_gb"] <= threshold) | rec_table["memory_gb"].isna()]
    else:
        return rec_table[rec_table["memory_gb"] <= threshold]