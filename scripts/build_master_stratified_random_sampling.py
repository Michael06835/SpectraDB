import argparse
import csv
import gzip
import random
import re
import time
import pickle
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, inchi, rdMolDescriptors
from tqdm import tqdm


RDLogger.DisableLog("rdApp.warning")

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


def element_class(elements: set[str]) -> str:
    if elements & {"F", "Cl", "Br", "I"}:
        return "halogen"
    if "P" in elements:
        return "P_containing"
    if "S" in elements:
        return "S_containing"
    if "N" in elements:
        return "N_containing"
    if elements.issubset({"C", "H", "O"}):
        return "CHO"
    return "other"


def mw_bin(mw: float) -> str:
    if mw < 150:
        return "MW_020_150"
    if mw < 300:
        return "MW_150_300"
    if mw < 500:
        return "MW_300_500"
    return "MW_500_800"


def stratum_key(mw: float, elements: set[str]) -> str:
    return f"{mw_bin(mw)}__{element_class(elements)}"


def download_cid_smiles(cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        print(f"Using cached file: {cache_path}")
        return

    print("Downloading PubChem CID-SMILES.gz ...")
    urlretrieve(CID_SMILES_URL, cache_path)
    print(f"Downloaded to: {cache_path}")


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")

    with tmp_path.open("wb") as f:
        pickle.dump(state, f)

    tmp_path.replace(path)


def load_checkpoint(path: Path):
    if not path.exists():
        return None

    with path.open("rb") as f:
        return pickle.load(f)


def rdkit_filter(smiles: str, min_mw: float, max_mw: float):
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        Chem.SanitizeMol(mol)

        for atom in mol.GetAtoms():
            if atom.GetIsotope() != 0:
                return None

        formula = rdMolDescriptors.CalcMolFormula(mol)
        elements = parse_elements(formula)

        if "C" not in elements:
            return None

        if not elements.issubset(ALLOWED_ELEMENTS):
            return None

        mw = Descriptors.MolWt(mol)

        if not (min_mw <= mw <= max_mw):
            return None

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        inchikey = inchi.MolToInchiKey(mol)

        if not inchikey:
            return None

        return {
            "canonical_smiles": canonical_smiles,
            "rdkit_inchikey": inchikey,
            "rdkit_formula": formula,
            "rdkit_molecular_weight": round(mw, 6),
            "elements": ";".join(sorted(elements)),
            "stratum": stratum_key(mw, elements),
        }

    except Exception:
        return None


def stratified_reservoir_sample(
    cid_smiles_path: Path,
    target: int,
    candidate_factor: int,
    min_mw: float,
    max_mw: float,
    seed: int,
    checkpoint_path: Path,
    checkpoint_every: int,
    resume: bool,
):
    reservoir_total = target * candidate_factor
    per_stratum_limit = max(1000, reservoir_total // 12)

    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
    else:
        checkpoint = None

    if checkpoint is not None:
        print(f"Resuming from checkpoint: {checkpoint_path}")

        reservoirs = checkpoint["reservoirs"]
        valid_counts = checkpoint["valid_counts"]
        seen_inchikeys = checkpoint["seen_inchikeys"]
        gzip_pos = checkpoint["gzip_pos"]
        scanned_lines = checkpoint["scanned_lines"]
        random.setstate(checkpoint["random_state"])

    else:
        print("Starting new stratified sampling run.")
        random.seed(seed)

        reservoirs = defaultdict(list)
        valid_counts = defaultdict(int)
        seen_inchikeys = set()
        gzip_pos = 0
        scanned_lines = 0

    print("Stratified reservoir sampling from CID-SMILES.gz ...")
    print(f"Target: {target}")
    print(f"Candidate factor: {candidate_factor}")
    print(f"Total candidate reservoir: {reservoir_total}")
    print(f"Per-stratum soft limit: {per_stratum_limit}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint every: {checkpoint_every} lines")

    try:
        with gzip.open(cid_smiles_path, "rb") as f:
            if gzip_pos:
                f.seek(gzip_pos)

            pbar = tqdm(desc="Stratified sampling", initial=scanned_lines)

            while True:
                gzip_pos = f.tell()
                line = f.readline()

                if not line:
                    break

                scanned_lines += 1
                pbar.update(1)

                try:
                    line = line.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue

                cid_text, smiles = parts

                try:
                    cid = int(cid_text)
                except ValueError:
                    continue

                checked = rdkit_filter(smiles, min_mw, max_mw)
                if checked is None:
                    continue

                key = checked["rdkit_inchikey"]

                if key in seen_inchikeys:
                    continue

                seen_inchikeys.add(key)

                s_key = checked["stratum"]
                valid_counts[s_key] += 1

                item = {
                    "cid": cid,
                    "rdkit_canonical_smiles": checked["canonical_smiles"],
                    "rdkit_inchikey": checked["rdkit_inchikey"],
                    "rdkit_formula": checked["rdkit_formula"],
                    "rdkit_molecular_weight": checked["rdkit_molecular_weight"],
                    "elements": checked["elements"],
                    "stratum": s_key,
                }

                bucket = reservoirs[s_key]
                n = valid_counts[s_key]

                if len(bucket) < per_stratum_limit:
                    bucket.append(item)
                else:
                    j = random.randint(1, n)
                    if j <= per_stratum_limit:
                        bucket[j - 1] = item

                if scanned_lines % checkpoint_every == 0:
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "reservoirs": reservoirs,
                            "valid_counts": valid_counts,
                            "seen_inchikeys": seen_inchikeys,
                            "gzip_pos": f.tell(),
                            "scanned_lines": scanned_lines,
                            "random_state": random.getstate(),
                        },
                    )
                    print(f"\nCheckpoint saved at line {scanned_lines}")

            pbar.close()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Saving checkpoint...")
        save_checkpoint(
            checkpoint_path,
            {
                "reservoirs": reservoirs,
                "valid_counts": valid_counts,
                "seen_inchikeys": seen_inchikeys,
                "gzip_pos": gzip_pos,
                "scanned_lines": scanned_lines,
                "random_state": random.getstate(),
            },
        )
        print(f"Checkpoint saved to: {checkpoint_path}")
        raise

    all_candidates = []
    for bucket in reservoirs.values():
        all_candidates.extend(bucket)

    random.shuffle(all_candidates)

    print("\nStratum summary:")
    for k in sorted(reservoirs):
        print(f"{k}: valid={valid_counts[k]}, kept={len(reservoirs[k])}")

    if len(all_candidates) < target:
        print(f"\nWarning: only {len(all_candidates)} candidates available, less than target {target}")

    selected = all_candidates[:reservoir_total]
    print(f"\nCandidates sampled: {len(selected)}")

    return selected


def fetch_pubchem_properties(cids: list[int]):
    if not cids:
        return []

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{','.join(map(str, cids))}/property/{','.join(PROPS)}/JSON"
    )

    try:
        r = requests.get(url, timeout=90)
        if r.status_code != 200:
            return []
        return r.json().get("PropertyTable", {}).get("Properties", [])
    except Exception:
        return []


def write_csv(path: Path, rows: list[dict]) -> None:
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
        "stratum",
        "xlogp",
        "tpsa",
        "h_bond_donor_count",
        "h_bond_acceptor_count",
        "rotatable_bond_count",
        "source",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build stratified-random compound_master.csv from PubChem CID-SMILES."
    )

    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--candidate-factor", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--min-mw", type=float, default=20)
    parser.add_argument("--max-mw", type=float, default=800)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=str, default="cache/CID-SMILES.gz")
    parser.add_argument("--out", type=str, default="master/compound_master_100000_random_raw.csv")
    parser.add_argument("--checkpoint",type=str,default="cache/stratified_sampling_checkpoint.pkl")
    parser.add_argument("--checkpoint-every",type=int,default=1000000)
    parser.add_argument("--resume",action="store_true")

    args = parser.parse_args()

    cache_path = Path(args.cache)
    out_path = Path(args.out)

    download_cid_smiles(cache_path)

    candidates = stratified_reservoir_sample(
        cid_smiles_path=cache_path,
        target=args.target,
        candidate_factor=args.candidate_factor,
        min_mw=args.min_mw,
        max_mw=args.max_mw,
        seed=args.seed,
        checkpoint_path=Path(args.checkpoint),
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
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
                    "stratum": local["stratum"],
                    "xlogp": item.get("XLogP", ""),
                    "tpsa": item.get("TPSA", ""),
                    "h_bond_donor_count": item.get("HBondDonorCount", ""),
                    "h_bond_acceptor_count": item.get("HBondAcceptorCount", ""),
                    "rotatable_bond_count": item.get("RotatableBondCount", ""),
                    "source": "PubChem",
                }
            )

        if len(rows) > 0 and (len(rows) % 1000 == 0 or len(rows) >= args.target):
            print(f"Accepted records: {len(rows)}")

        time.sleep(args.sleep)

    write_csv(out_path, rows)

    print("Done.")
    print(f"Total records: {len(rows)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()