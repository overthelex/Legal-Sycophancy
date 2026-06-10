#!/usr/bin/env python3
"""
Clean and normalize respondent names in the ECtHR verdict-free dataset.

Fixes:
  1. UNKNOWN respondents - re-extract from case_name ("v. COUNTRY" / "c. COUNTRY")
  2. Broken parsing ("V. FRANCE", "AND OTHERS V. RUSSIA") - re-extract from case_name
  3. Duplicate country names - normalize to canonical form

Usage:
  # Preview changes (dry run)
  python scripts/clean_respondent_names.py --preview

  # Apply changes and save
  python scripts/clean_respondent_names.py --apply --output data/processed/echr_verdict_free_clean.json

  # Push cleaned dataset to HuggingFace
  python scripts/clean_respondent_names.py --apply --push overthelex/echr-verdict-free
"""

import argparse
import json
import re
from collections import Counter

COUNTRY_NORMALIZATION = {
    "TÜRKİYE": "TURKEY",
    "TÜRKIYE": "TURKEY",
    "TURQUIE": "TURKEY",
    "REPUBLIC OF MOLDOVA": "MOLDOVA",
    "THE REPUBLIC OF MOLDOVA": "MOLDOVA",
    '"THE FORMER YUGOSLAV REPUBLIC OF MACEDONIA"': "NORTH MACEDONIA",
    "THE FORMER YUGOSLAV REPUBLIC OF MACEDONIA": "NORTH MACEDONIA",
    '"THE FORMER YOUGOSLAV REPUBLIC OF MACEDONIA"': "NORTH MACEDONIA",
    "THE FORMER YOUGOSLAV REPUBLIC OF MACEDONIA": "NORTH MACEDONIA",
    "FORMER YUGOSLAV REPUBLIC OF MACEDONIA": "NORTH MACEDONIA",
    "THE NETHERLANDS": "NETHERLANDS",
    "THE UNITED KINGDOM": "UNITED KINGDOM",
    "THE CZECH REPUBLIC": "CZECH REPUBLIC",
    "ITALIE": "ITALY",
    "FINLANDE": "FINLAND",
}


def strip_gc_suffix(name: str) -> str:
    return re.sub(r"\s*\[GC\]\s*$", "", name, flags=re.IGNORECASE).strip()


def extract_respondent(case_name: str) -> str | None:
    m = re.search(r"\bv\.\s+(.+?)(?:\s*$|\s*\()", case_name, re.IGNORECASE)
    if not m:
        m = re.search(r"\bc\.\s+(.+?)(?:\s*$|\s*\()", case_name, re.IGNORECASE)
    if m:
        country = m.group(1).strip().rstrip(".")
        country = country.strip('"').strip("'").strip()
        country = strip_gc_suffix(country)
        return country.upper()
    return None


KNOWN_COUNTRIES = {
    "RUSSIA", "UKRAINE", "TURKEY", "ROMANIA", "HUNGARY",
    "POLAND", "AZERBAIJAN", "BULGARIA", "CROATIA", "ITALY",
    "MOLDOVA", "SERBIA", "ARMENIA", "LITHUANIA", "SLOVAKIA",
    "SLOVENIA", "GREECE", "GEORGIA", "FRANCE", "UNITED KINGDOM",
    "GERMANY", "ALBANIA", "MALTA", "BOSNIA AND HERZEGOVINA",
    "AUSTRIA", "LATVIA", "SWITZERLAND", "SPAIN", "PORTUGAL",
    "BELGIUM", "SWEDEN", "CZECH REPUBLIC", "MONTENEGRO",
    "FINLAND", "NETHERLANDS", "CYPRUS", "ESTONIA",
    "NORTH MACEDONIA", "NORWAY", "DENMARK", "ICELAND",
    "IRELAND", "SAN MARINO", "LUXEMBOURG", "LIECHTENSTEIN",
    "ANDORRA",
}


def normalize_country(name: str) -> str:
    name = name.strip().strip('"').strip("'").strip()
    name = strip_gc_suffix(name)
    if name in COUNTRY_NORMALIZATION:
        return COUNTRY_NORMALIZATION[name]
    if name.startswith("THE "):
        stripped = name[4:]
        if stripped in COUNTRY_NORMALIZATION:
            return COUNTRY_NORMALIZATION[stripped]
        if stripped in KNOWN_COUNTRIES:
            return stripped
    if name.startswith("V. ") or name.startswith("AND "):
        for country in sorted(KNOWN_COUNTRIES, key=len, reverse=True):
            if name.endswith(country):
                return country
        for orig, norm in COUNTRY_NORMALIZATION.items():
            if name.endswith(orig):
                return norm
    return name


def needs_reextraction(respondent: str) -> bool:
    if respondent == "UNKNOWN":
        return True
    if respondent.startswith("V. "):
        return True
    if respondent.startswith("AND "):
        return True
    if "V." in respondent and respondent not in ("AZERBAIJAN", "SLOVENIA"):
        parts = respondent.split()
        if any(p == "V." for p in parts) and not all(
            p in ("V.", "AND") or len(p) > 3 for p in parts
        ):
            return True
    return False


def is_multi_respondent(respondent: str) -> bool:
    if " AND " in respondent:
        parts = [p.strip() for p in respondent.split(" AND ")]
        return all(len(p) > 3 and not p.startswith("V.") and p.isalpha() or " " in p for p in parts)
    return False


def clean_dataset(records: list[dict], preview: bool = True) -> list[dict]:
    changes = []
    reextracted = 0
    normalized = 0
    multi_respondent_split = 0
    unchanged = 0

    for rec in records:
        old = rec["respondent"]
        new = old

        if needs_reextraction(old):
            extracted = extract_respondent(rec["case_name"])
            if extracted:
                new = extracted
                reextracted += 1
            else:
                new = old

        new = normalize_country(new)

        if " AND " in new and not is_multi_respondent(new):
            parts = new.split(" AND ")
            last = parts[-1].strip()
            for country in COUNTRY_NORMALIZATION:
                if last == country or last == normalize_country(country):
                    new = normalize_country(last)
                    break
            else:
                known_countries = {
                    "RUSSIA", "UKRAINE", "TURKEY", "ROMANIA", "HUNGARY",
                    "POLAND", "AZERBAIJAN", "BULGARIA", "CROATIA", "ITALY",
                    "MOLDOVA", "SERBIA", "ARMENIA", "LITHUANIA", "SLOVAKIA",
                    "SLOVENIA", "GREECE", "GEORGIA", "FRANCE", "UNITED KINGDOM",
                    "GERMANY", "ALBANIA", "MALTA", "BOSNIA AND HERZEGOVINA",
                    "AUSTRIA", "LATVIA", "SWITZERLAND", "SPAIN", "PORTUGAL",
                    "BELGIUM", "SWEDEN", "CZECH REPUBLIC", "MONTENEGRO",
                    "FINLAND", "NETHERLANDS", "CYPRUS", "ESTONIA",
                    "NORTH MACEDONIA", "NORWAY", "DENMARK", "ICELAND",
                    "IRELAND", "SAN MARINO", "LUXEMBOURG", "LIECHTENSTEIN",
                    "ANDORRA",
                }
                if last in known_countries:
                    new = last

        new = normalize_country(new)

        if old != new:
            if old not in (rec.get("_reported", set())):
                changes.append((rec["item_id"], rec["case_name"][:80], old, new))
            if not preview:
                rec["respondent"] = new
            if old == "UNKNOWN":
                pass
            else:
                normalized += 1
        else:
            unchanged += 1

    if preview:
        print(f"\n{'='*100}")
        print(f"PREVIEW: {len(changes)} changes out of {len(records)} records")
        print(f"  Re-extracted from case_name: {reextracted}")
        print(f"  Normalized country name: {normalized}")
        print(f"  Unchanged: {unchanged}")
        print(f"{'='*100}\n")

        by_type = {"UNKNOWN->": [], "V./AND->": [], "normalize": []}
        for item_id, case_name, old, new in changes:
            if old == "UNKNOWN":
                by_type["UNKNOWN->"].append((item_id, case_name, old, new))
            elif old.startswith("V. ") or old.startswith("AND "):
                by_type["V./AND->"].append((item_id, case_name, old, new))
            else:
                by_type["normalize"].append((item_id, case_name, old, new))

        for category, items in by_type.items():
            if not items:
                continue
            unique_mappings = Counter((old, new) for _, _, old, new in items)
            print(f"\n--- {category} ({len(items)} records, {len(unique_mappings)} unique mappings) ---")
            for (old, new), count in unique_mappings.most_common():
                print(f"  {old:50s} -> {new:30s} ({count} records)")

        print(f"\n--- Resulting respondent distribution (top 30) ---")
        result_counts = Counter()
        for rec in records:
            old = rec["respondent"]
            for item_id, _, o, n in changes:
                if item_id == rec["item_id"] and o == old:
                    result_counts[n] += 1
                    break
            else:
                result_counts[normalize_country(old)] += 1

        for country, count in result_counts.most_common(30):
            print(f"  {count:>6}  {country}")
        print(f"  ... {len(result_counts)} total unique respondent values")

    return records


def main():
    parser = argparse.ArgumentParser(description="Clean respondent names in ECtHR dataset")
    parser.add_argument("--input", help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--preview", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    parser.add_argument("--push", help="Push to HuggingFace dataset (e.g. overthelex/echr-verdict-free)")
    args = parser.parse_args()

    if args.input:
        print(f"Loading from {args.input}...")
        with open(args.input) as f:
            records = json.load(f)
    else:
        print("Loading from HuggingFace overthelex/echr-verdict-free...")
        from datasets import load_dataset
        ds = load_dataset("overthelex/echr-verdict-free", split="train")
        records = [dict(r) for r in ds]

    print(f"Loaded {len(records)} records")

    if args.preview or not args.apply:
        clean_dataset(records, preview=True)
        return

    if args.apply:
        clean_dataset(records, preview=False)

        result_counts = Counter(r["respondent"] for r in records)
        print(f"\nAfter cleaning: {len(result_counts)} unique respondent values")
        for country, count in result_counts.most_common(30):
            print(f"  {count:>6}  {country}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"\nSaved to {args.output}")

        if args.push:
            from datasets import Dataset
            ds = Dataset.from_list(records)
            ds.push_to_hub(args.push)
            print(f"\nPushed to {args.push}")


if __name__ == "__main__":
    main()
