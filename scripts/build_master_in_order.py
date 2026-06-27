import argparse
import csv
import gzip
import re
import time
from pathlib import Path
from urllib.request import urlretrieve

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, inchi
from tqdm import tqdm


CID_SMILES_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"

ALLOWED_ELEMENTS = {"C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"}

PROPS = [
    "MolecularFormula",
    "MolecularWeight",
    "ConnectivitySMILES",
    "IsomericSMILES",
    "InChI",
    "InChIKey",
    "IUPACName",
    "ExactMass",
    "Charge",
    "HeavyAtomCount",
    "IsotopeAtomCount",
    "CovalentUnitCount",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
]


def parse_elements(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula or ""))


def download_cid_smiles(cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        print(f"Using cached file: {cache_path}")
        return

    print("Downloading PubChem CID-SMILES.gz ...")
    print("This may take several minutes.")
    urlretrieve(CID_SMILES_URL, cache_path)
    print(f"Downloaded to: {cache_path}")


def rdkit_check(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    elements = parse_elements(formula)

    if "C" not in elements:
        return None

    if not elements.issubset(ALLOWED_ELEMENTS):
        return None

    mw = Descriptors.MolWt(mol)

    return {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "rdkit_inchikey": inchi.MolToInchiKey(mol),
        "formula": formula,
        "rdkit_molecular_weight": round(mw, 6),
        "elements": ";".join(sorted(elements)),
    }


def collect_candidate_cids(
    cid_smiles_path: Path,
    target_candidates: int,
    min_mw: float,
    max_mw: float,
):
    candidates = []
    seen_inchikeys = set()

    print("Scanning CID-SMILES file locally...")

    with gzip.open(cid_smiles_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Local filtering"):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue

            cid_text, smiles = parts

            try:
                cid = int(cid_text)
            except ValueError:
                continue

            checked = rdkit_check(smiles)
            if checked is None:
                continue

            mw = checked["rdkit_molecular_weight"]
            if not (min_mw <= mw <= max_mw):
                continue

            key = checked["rdkit_inchikey"]
            if key in seen_inchikeys:
                continue

            seen_inchikeys.add(key)

            candidates.append(
                {
                    "cid": cid,
                    "rdkit_canonical_smiles": checked["canonical_smiles"],
                    "rdkit_inchikey": checked["rdkit_inchikey"],
                    "rdkit_formula": checked["formula"],
                    "rdkit_molecular_weight": checked["rdkit_molecular_weight"],
                    "elements": checked["elements"],
                }
            )

            if len(candidates) >= target_candidates:
                break

    print(f"Candidate CIDs collected: {len(candidates)}")
    return candidates


def fetch_pubchem_properties(cids: list[int]):
    if not cids:
        return []

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{','.join(map(str, cids))}/property/{','.join(PROPS)}/JSON"
    )

    r = requests.get(url, timeout=60)

    if r.status_code != 200:
        return []

    return r.json().get("PropertyTable", {}).get("Properties", [])


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "cid",
        "iupac_name",
        "pubchem_inchikey",
        "rdkit_inchikey",
        "canonical_smiles",
        "isomeric_smiles",
        "formula",
        "pubchem_molecular_weight",
        "rdkit_molecular_weight",
        "exact_mass",
        "charge",
        "heavy_atom_count",
        "isotope_atom_count",
        "covalent_unit_count",
        "elements",
        "xlogp",
        "tpsa",
        "h_bond_donor_count",
        "h_bond_acceptor_count",
        "rotatable_bond_count",
        "source",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build large-scale compound_master.csv from PubChem CID-SMILES."
    )
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--candidate-factor", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--min-mw", type=float, default=20)
    parser.add_argument("--max-mw", type=float, default=800)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--cache", type=str, default="cache/CID-SMILES.gz")
    parser.add_argument("--out", type=str, default="master/compound_master.csv")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    out_path = Path(args.out)

    download_cid_smiles(cache_path)

    candidate_target = args.target * args.candidate_factor

    candidates = collect_candidate_cids(
        cache_path,
        target_candidates=candidate_target,
        min_mw=args.min_mw,
        max_mw=args.max_mw,
    )

    candidate_map = {x["cid"]: x for x in candidates}
    cids = list(candidate_map.keys())

    rows = []
    seen_keys = set()

    for i in tqdm(range(0, len(cids), args.batch_size), desc="Fetching PubChem properties"):
        if len(rows) >= args.target:
            break

        batch = cids[i:i + args.batch_size]
        items = fetch_pubchem_properties(batch)

        for item in items:
            if len(rows) >= args.target:
                break

            cid = item.get("CID")
            if cid not in candidate_map:
                continue

            local = candidate_map[cid]

            key = local["rdkit_inchikey"]
            if key in seen_keys:
                continue

            seen_keys.add(key)

            rows.append(
                {
                    "cid": cid,
                    "iupac_name": item.get("IUPACName", ""),
                    "pubchem_inchikey": item.get("InChIKey", ""),
                    "rdkit_inchikey": local["rdkit_inchikey"],
                    "canonical_smiles": local["rdkit_canonical_smiles"],
                    "isomeric_smiles": item.get("IsomericSMILES", ""),
                    "formula": item.get("MolecularFormula", local["rdkit_formula"]),
                    "pubchem_molecular_weight": item.get("MolecularWeight", ""),
                    "rdkit_molecular_weight": local["rdkit_molecular_weight"],
                    "exact_mass": item.get("ExactMass", ""),
                    "charge": item.get("Charge", ""),
                    "heavy_atom_count": item.get("HeavyAtomCount", ""),
                    "isotope_atom_count": item.get("IsotopeAtomCount", ""),
                    "covalent_unit_count": item.get("CovalentUnitCount", ""),
                    "elements": local["elements"],
                    "xlogp": item.get("XLogP", ""),
                    "tpsa": item.get("TPSA", ""),
                    "h_bond_donor_count": item.get("HBondDonorCount", ""),
                    "h_bond_acceptor_count": item.get("HBondAcceptorCount", ""),
                    "rotatable_bond_count": item.get("RotatableBondCount", ""),
                    "source": "PubChem",
                }
            )

        print(f"Accepted records: {len(rows)}")
        time.sleep(args.sleep)

    write_csv(out_path, rows)

    print("Done.")
    print(f"Total records: {len(rows)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
