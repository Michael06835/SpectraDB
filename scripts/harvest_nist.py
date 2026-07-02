import csv
import re
import time
import argparse
import json
from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MASTER_PATH = Path("master/compound_master_100000_random_clean.csv")
OUT_DIR = Path("raw/nist")
META_PATH = Path("raw/nist/nist_metadata.csv")
CHECKPOINT_PATH = Path("cache/nist_checkpoint.json")

NIST_BASE = "https://webbook.nist.gov"
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")

def create_session():
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": "SpectraDB/0.1 academic research contact: local"
        }
    )

    return session


SESSION = create_session()


def get_pubchem_cas(cid: int) -> str:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
    try:
        r = SESSION.get(url, timeout=20)
        if r.status_code != 200:
            return ""
        data = r.json()
        syns = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
        for s in syns:
            if CAS_RE.match(s):
                return s
    except Exception:
        return ""
    return ""


def get_nist_page(cas: str) -> str:
    url = f"{NIST_BASE}/cgi/cbook.cgi?ID={cas}&Units=SI"
    try:
        r = SESSION.get(url, timeout=30)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return ""


def find_jcamp_links(html: str):
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a"):
        text = a.get_text(" ", strip=True)
        href = a.get("href", "")

        if not href:
            continue

        full = urljoin(NIST_BASE, href)
        s = (text + " " + href).lower()

        if "ir spectrum" in s or "type=ir" in s:
            links.append(("ir", full))
            continue

        if (
            "uv spectrum" in s
            or "uv/vis" in s
            or "uv-visible" in s
            or "visible spectrum" in s
            or "type=uv" in s
        ):
            links.append(("uvvis", full))
            continue
        
        if "raman" in s or "type=raman" in s:
            links.append(("raman", full))
            continue

    return links


def download_file(url: str, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        r = SESSION.get(url, timeout=60)
        if r.status_code != 200:
            return False
        text = r.text
        if len(text) < 50:
            return False
        path.write_text(text, encoding="utf-8", errors="ignore")
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ir").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "uvvis").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raman").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MASTER_PATH)
    
    start_index = 0

    if args.resume and CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
            state = json.load(f)
        start_index = state.get("current_index", 0)
        print(
            f"Resume from index {start_index}, "
            f"last CID={state.get('last_cid')}, "
            f"time={state.get('timestamp')}"
        )

    df = df.iloc[start_index:]

    if args.limit is not None:
        df = df.head(args.limit)

    meta_fields = [
        "cid",
        "inchikey",
        "cas",
        "modality",
        "source",
        "url",
        "local_path",
        "status",
    ]

    if not META_PATH.exists():
        with META_PATH.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=meta_fields).writeheader()

    def save_progress(index, cid, inchikey):
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "current_index": index + 1,
                    "last_cid": cid,
                    "last_inchikey": inchikey,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    
    stats = {
        "total": 0,
        "cas_found": 0,
        "nist_page_found": 0,
        "spectra_links_found": 0,
        "ir_success": 0,
        "uvvis_success": 0,
        "raman_success": 0,
        "download_failed": 0,
    }
    
    for global_idx, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="Harvesting NIST"),
        start=start_index,
    ):
        cid = int(row["cid"])
        inchikey = row["rdkit_inchikey"]
        stats["total"] += 1

        cas = get_pubchem_cas(cid)
        if not cas:
            save_progress(global_idx, cid, inchikey)
            continue
        stats["cas_found"] += 1

        html = get_nist_page(cas)
        if not html:
            save_progress(global_idx, cid, inchikey)
            continue
        stats["nist_page_found"] += 1

        links = find_jcamp_links(html)
        stats["spectra_links_found"] += len(links)

        for idx, (modality, url) in enumerate(links):
            if modality not in {"ir", "uvvis", "raman"}:
                continue

            filename = f"{cid}_{inchikey}_{idx}.jdx"
            local_path = OUT_DIR / modality / filename
            if local_path.exists():
                continue

            ok = download_file(url, local_path)
            if ok:
                if modality == "ir":
                    stats["ir_success"] += 1
                elif modality == "uvvis":
                    stats["uvvis_success"] += 1
                elif modality == "raman":
                    stats["raman_success"] += 1
            else:
                stats["download_failed"] += 1

            with META_PATH.open("a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=meta_fields)
                writer.writerow(
                    {
                        "cid": cid,
                        "inchikey": inchikey,
                        "cas": cas,
                        "modality": modality,
                        "source": "NIST",
                        "url": url,
                        "local_path": str(local_path),
                        "status": "success" if ok else "failed",
                    }
                )

        time.sleep(0.5)
        
        save_progress(global_idx, cid, inchikey)
    
    print("\nNIST Harvest Summary")
    print(f"Total compounds checked: {stats['total']}")
    print(f"CAS found: {stats['cas_found']}")
    print(f"NIST pages found: {stats['nist_page_found']}")
    print(f"Spectral links found: {stats['spectra_links_found']}")
    print(f"IR downloaded: {stats['ir_success']}")
    print(f"UV-Vis downloaded: {stats['uvvis_success']}")
    print(f"Raman downloaded: {stats['raman_success']}")
    print(f"Download failed: {stats['download_failed']}")

if __name__ == "__main__":
    main()