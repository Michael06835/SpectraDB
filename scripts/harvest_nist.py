import argparse
import csv
import json
import os
import re
import time
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MASTER_PATH = Path("master/compound_master_100000_in_order_clean.csv")
OUT_DIR = Path("raw/nist")
META_PATH = OUT_DIR / "nist_metadata.csv"
CHECKPOINT_PATH = Path("cache/nist_checkpoint.json")
CAS_CACHE_PATH = Path("cache/cid_cas.csv")
FAILED_LOG_PATH = Path("logs/nist_failed.log")

NIST_BASE = "https://webbook.nist.gov"
NIST_CBOOK = f"{NIST_BASE}/cgi/cbook.cgi"
PUBCHEM_SYNONYMS = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
NIST_ID_RE = re.compile(r"\bC\d+\b")
JCAMP_URL_RE = re.compile(r"(?P<url>/cgi/cbook\.cgi\?JCAMP=[^'\"<>\s)]+)", re.IGNORECASE)
JCAMP_MARKERS = (b"##JCAMP-DX=", b"##TITLE=", b"##DATA TYPE=")
HTML_MARKERS = (
    b"<html",
    b"<!doctype html",
    b"<body",
    b"nist chemistry webbook",
    b"webbook.nist.gov",
)

MODALITY_TYPES = {
    "ir": "IR",
    "raman": "Raman",
    "uvvis": "UVVis",
}

TYPE_MODALITIES = {value.lower(): key for key, value in MODALITY_TYPES.items()}
REQUEST_TIMEOUT = (3, 8)
CID_TIME_LIMIT = 20
NETWORK_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
)
PUBCHEM_COOLDOWN_SECONDS = 20 * 60
PUBCHEM_COOLDOWN_LOCK = threading.Lock()
PUBCHEM_COOLDOWN_UNTIL = 0.0

def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def create_session():
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "SpectraDB/0.1 academic research"})
    return session


THREAD_LOCAL = threading.local()
CAS_CACHE_LOCK = threading.Lock()
FAILED_LOG_LOCK = threading.Lock()
META_LOCK = threading.Lock()
CHECKPOINT_LOCK = threading.Lock()


def get_session():
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = create_session()
        THREAD_LOCAL.session = session
    return session


def reset_thread_session():
    session = getattr(THREAD_LOCAL, "session", None)
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
    THREAD_LOCAL.session = create_session()


def is_pubchem_cooldown_error(message):
    text = (message or "").lower()
    return (
        "pubchem.ncbi.nlm.nih.gov" in text
        and (
            "too many 503" in text
            or "max retries exceeded" in text
            or "retryerror" in text
            or "proxyerror" in text
            or "connectionpool" in text
            or "503" in text
            or "429" in text
        )
    )


def pubchem_cooldown(message):
    global PUBCHEM_COOLDOWN_UNTIL

    with PUBCHEM_COOLDOWN_LOCK:
        now = time.monotonic()
        if PUBCHEM_COOLDOWN_UNTIL <= now:
            PUBCHEM_COOLDOWN_UNTIL = now + PUBCHEM_COOLDOWN_SECONDS
            wait_seconds = PUBCHEM_COOLDOWN_SECONDS
            print(
                f"\n[PubChem cooldown] Sleeping {wait_seconds // 60} min. "
                f"reason={str(message)[:200]}",
                flush=True,
            )
        else:
            wait_seconds = max(1, PUBCHEM_COOLDOWN_UNTIL - now)
            print(
                f"\n[PubChem cooldown] Existing cooldown. "
                f"Sleeping {wait_seconds / 60:.1f} min.",
                flush=True,
            )

    time.sleep(wait_seconds)
    reset_thread_session()


def request_timeout(deadline=None):
    if deadline is None:
        return REQUEST_TIMEOUT

    remaining = deadline - time.monotonic()
    if remaining <= 2:
        raise requests.exceptions.Timeout("cid_time_limit_exceeded")

    connect_timeout = max(1, min(REQUEST_TIMEOUT[0], remaining * 0.25))
    read_timeout = max(1, min(REQUEST_TIMEOUT[1], remaining - connect_timeout))
    return connect_timeout, read_timeout


def cid_time_left(deadline):
    return max(0.0, deadline - time.monotonic())


def print_progress(index, total, cid, cas, status, reason, elapsed):
    total_text = str(total) if total is not None else "?"
    print(
        f"[{index}/{total_text}] CID={cid} CAS={cas or '-'} "
        f"status={status} reason={reason or '-'} elapsed={elapsed:.1f}s",
        flush=True,
    )


def print_summary(stats):
    print(
        "Summary: "
        f"total={stats['total']} cas_found={stats['cas_found']} "
        f"success={stats['success']} failed={stats['failed']} skipped={stats['skipped']}",
        flush=True,
    )


def log_failure(cid, cas, reason, url="", modality="", message=""):
    with FAILED_LOG_LOCK:
        FAILED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FAILED_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(
                f"{now_text()}\tcid={cid}\tcas={cas}\tmodality={modality}\t"
                f"url={url}\treason={reason}\tmessage={message}\n"
            )


def log_cache_failure(reason):
    try:
        log_failure("", "", f"cache_write_failed:{reason}")
    except Exception:
        pass


def replace_with_retry(tmp_path, target_path, attempts=5, delay=0.75):
    last_error = None
    for _ in range(attempts):
        try:
            os.replace(tmp_path, target_path)
            return True
        except OSError as exc:
            last_error = exc
            time.sleep(delay)
    log_cache_failure(f"{type(last_error).__name__}:{last_error}")
    return False


def load_cas_cache():
    if not CAS_CACHE_PATH.exists():
        return {}

    cache = {}
    with CAS_CACHE_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                cache[int(row["cid"])] = row
            except (KeyError, ValueError):
                continue
    return cache


def save_cas_cache(cache):
    with CAS_CACHE_LOCK:
        try:
            CAS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log_cache_failure(f"mkdir:{type(exc).__name__}:{exc}")
            return False

        fields = ["cid", "cas", "source", "status", "updated_at", "message"]
        tmp_path = CAS_CACHE_PATH.with_name(f"{CAS_CACHE_PATH.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp")

        try:
            with tmp_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for cid in sorted(cache):
                    row = dict(cache[cid])
                    for field in fields:
                        row.setdefault(field, "")
                    writer.writerow(row)
        except OSError as exc:
            log_cache_failure(f"write_tmp:{type(exc).__name__}:{exc}")
            return False

        ok = replace_with_retry(tmp_path, CAS_CACHE_PATH)
        if not ok:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return ok


def get_pubchem_cas(cid, cas_cache, deadline=None):
    with CAS_CACHE_LOCK:
        cached = cas_cache.get(cid)
        if cached:
            return cached.get("cas", "")

    cas = ""
    status = "not_found"
    message = ""
    url = PUBCHEM_SYNONYMS.format(cid=cid)

    for attempt in range(2):
        if deadline is not None and deadline - time.monotonic() <= 2:
            deadline = time.monotonic() + CID_TIME_LIMIT
        
        try:
            response = get_session().get(url, timeout=request_timeout(deadline))

            if response.status_code == 404:
                status = "not_found"
                message = "pubchem_synonyms_404"
                break

            if response.status_code in (429, 500, 502, 503, 504):
                status = "request_failed"
                message = f"http_{response.status_code}"

                if response.status_code in (429, 503) and attempt == 0:
                    pubchem_cooldown(f"PubChem HTTP {response.status_code}")
                    deadline = time.monotonic() + CID_TIME_LIMIT
                    continue

                break

            if response.status_code != 200:
                status = "request_failed"
                message = f"http_{response.status_code}"
                break

            data = response.json()
            synonyms = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])

            for synonym in synonyms:
                if CAS_RE.match(str(synonym)):
                    cas = str(synonym)
                    status = "success"
                    break

            if not cas:
                status = "not_found"
                message = "no_cas_in_synonyms"

            break

        except NETWORK_EXCEPTIONS as exc:
            status = "request_failed"
            message = f"{type(exc).__name__}:{exc}"

            if is_pubchem_cooldown_error(message) and attempt == 0:
                pubchem_cooldown(message)
                deadline = time.monotonic() + CID_TIME_LIMIT
                continue

            break

        except Exception as exc:
            status = "request_failed"
            message = f"{type(exc).__name__}:{exc}"

            if is_pubchem_cooldown_error(message) and attempt == 0:
                pubchem_cooldown(message)
                deadline = time.monotonic() + CID_TIME_LIMIT
                continue

            break

    with CAS_CACHE_LOCK:
        existing = cas_cache.get(cid)
        if existing:
            return existing.get("cas", "")

        cas_cache[cid] = {
            "cid": cid,
            "cas": cas,
            "source": "PubChem",
            "status": status,
            "updated_at": now_text(),
            "message": message,
        }

    save_cas_cache(cas_cache)
    return cas

def save_checkpoint(index, cid):
    with CHECKPOINT_LOCK:
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_PATH.write_text(
            json.dumps(
                {
                    "current_index": index + 1,
                    "last_cid": cid,
                    "timestamp": now_text(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def load_start_index(args):
    if args.resume and CHECKPOINT_PATH.exists():
        try:
            data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            return int(data.get("current_index", args.start))
        except (ValueError, json.JSONDecodeError):
            return args.start
    return args.start


def fetch_nist_page(cas, deadline=None):
    params = {"ID": cas, "Units": "SI", "Mask": "480"}
    url = f"{NIST_CBOOK}?{urlencode(params)}"
    try:
        response = get_session().get(url, timeout=request_timeout(deadline))
    except NETWORK_EXCEPTIONS as exc:
        return "", url, f"nist_request_{type(exc).__name__}"
    if response.status_code != 200:
        return "", url, f"nist_http_{response.status_code}"
    text = response.text or ""
    lower = text.lower()
    if "name not found" in lower or "registry number not found" in lower or "not found" in lower:
        return text, response.url, "nist_not_found"
    return text, response.url, ""


def extract_nist_id(html, cas):
    soup = BeautifulSoup(html or "", "html.parser")
    candidates = []

    for a in soup.find_all("a"):
        href = a.get("href", "")
        parsed = urlparse(urljoin(NIST_BASE, href))
        query = parse_qs(parsed.query)
        for value in query.get("ID", []):
            if NIST_ID_RE.fullmatch(value):
                candidates.append(value)

    for match in NIST_ID_RE.findall(html or ""):
        candidates.append(match)

    if candidates:
        return candidates[0]
    if CAS_RE.match(cas):
        return f"C{cas.replace('-', '')}"
    return ""


def modality_from_type(value):
    return TYPE_MODALITIES.get((value or "").lower(), "")


def normalize_jcamp_url(url):
    absolute = urljoin(NIST_BASE, url.replace("&amp;", "&"))
    parsed = urlparse(absolute)
    query = parse_qs(parsed.query)
    if "JCAMP" not in query or "Type" not in query:
        return "", ""
    modality = modality_from_type(query.get("Type", [""])[0])
    if not modality:
        return "", ""
    return absolute, modality


def page_has_modality(html, modality):
    lower = (html or "").lower()
    if modality == "ir":
        return "ir spectrum" in lower or "type=ir" in lower or "#ir-spec" in lower
    if modality == "uvvis":
        return "uv/visible spectrum" in lower or "type=uvvis" in lower or "#uv-vis-spec" in lower
    if modality == "raman":
        return "raman" in lower or "type=raman" in lower
    return False


def build_jcamp_url(nist_id, modality, index):
    params = {"JCAMP": nist_id, "Index": index, "Type": MODALITY_TYPES[modality]}
    return f"{NIST_CBOOK}?{urlencode(params)}"


def extract_jcamp_links(html, cas):
    nist_id = extract_nist_id(html, cas)
    links = []

    soup = BeautifulSoup(html or "", "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href", "")
        url, modality = normalize_jcamp_url(href)
        if url:
            links.append((modality, url))

    for match in JCAMP_URL_RE.finditer(html or ""):
        url, modality = normalize_jcamp_url(match.group("url"))
        if url:
            links.append((modality, url))

    deduped = []
    seen = set()
    for modality, url in links:
        key = (modality, url)
        if key not in seen:
            seen.add(key)
            deduped.append((modality, url))
    return deduped


def is_valid_jcamp(content):
    if not content:
        return False, "empty_response"
    head = content[:2048].lower()
    if any(marker in head for marker in HTML_MARKERS):
        return False, "html_response"
    if not any(marker in content[:8192] for marker in JCAMP_MARKERS):
        return False, "missing_jcamp_marker"
    return True, ""


def file_has_valid_jcamp(path):
    try:
        content = path.read_bytes()
    except OSError:
        return False
    ok, _ = is_valid_jcamp(content)
    return ok


def download_jcamp(url, deadline=None):
    try:
        response = get_session().get(url, timeout=request_timeout(deadline))
    except NETWORK_EXCEPTIONS as exc:
        return False, b"", f"download_{type(exc).__name__}"
    except Exception as exc:
        return False, b"", f"download_exception:{type(exc).__name__}"

    if response.status_code != 200:
        return False, response.content or b"", f"download_http_{response.status_code}"

    ok, reason = is_valid_jcamp(response.content)
    if not ok:
        return False, response.content or b"", reason
    return True, response.content, ""


def output_path(cid, cas, modality, item_index):
    safe_cas = cas.replace("/", "_")
    return OUT_DIR / modality / f"{cid}_{safe_cas}_{item_index}.jdx"


def write_metadata(row):
    with META_LOCK:
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        exists = META_PATH.exists()
        fields = ["cid", "cas", "modality", "file_path", "source_url", "downloaded_at"]
        with META_PATH.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def load_master(args, start_index):
    nrows = None
    if args.limit is not None:
        nrows = start_index + args.limit
    df = pd.read_csv(MASTER_PATH, nrows=nrows)
    required = {"cid", "rdkit_inchikey", "canonical_smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"master missing required columns: {', '.join(sorted(missing))}")
    df = df.iloc[start_index:]
    if args.limit is not None:
        df = df.head(args.limit)
    return df


def print_success_previews(paths):
    for path in paths:
        print(f"\n{path}")
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[:5]:
                print(line)
        except OSError as exc:
            print(f"<preview failed: {exc}>")



def process_row(global_index, row, cas_cache, sleep_seconds):
    cid = int(row["cid"])
    cid_start = time.monotonic()
    cid_deadline = cid_start + CID_TIME_LIMIT
    cas = ""
    result = {
        "global_index": global_index,
        "cid": cid,
        "cas": "",
        "status": "unknown",
        "reason": "",
        "elapsed": 0.0,
        "cas_found": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "success_paths": [],
    }

    try:
        cas = get_pubchem_cas(cid, cas_cache, cid_deadline)
        result["cas"] = cas
        if not cas:
            reason = cas_cache.get(cid, {}).get("status", "cas_missing")
            message = cas_cache.get(cid, {}).get("message", "")
            log_failure(cid, "", reason, message=message)
            result.update({"status": "skipped", "reason": reason, "skipped": 1})
            return result

        result["cas_found"] = 1
        html, page_url, page_reason = fetch_nist_page(cas, cid_deadline)
        if page_reason:
            log_failure(cid, cas, page_reason, page_url)
            result.update({"status": "skipped", "reason": page_reason, "skipped": 1})
            return result

        links = extract_jcamp_links(html, cas)
        if not links:
            log_failure(cid, cas, "no_jcamp_links", page_url)
            result.update({"status": "skipped", "reason": "no_jcamp_links", "skipped": 1})
            return result

        item_counts = {modality: 0 for modality in MODALITY_TYPES}
        cid_success = 0
        cid_existing = 0
        cid_failed = 0
        last_reasons = []

        for modality, source_url in links:
            if cid_time_left(cid_deadline) <= 0:
                cid_failed += 1
                last_reasons.append("cid_time_limit_exceeded")
                log_failure(cid, cas, "cid_time_limit_exceeded", page_url, modality)
                break

            item_index = item_counts[modality]
            item_counts[modality] += 1
            path = output_path(cid, cas, modality, item_index)

            if path.exists() and file_has_valid_jcamp(path):
                cid_existing += 1
                result["success_paths"].append(path)
                continue

            ok, content, download_reason = download_jcamp(source_url, cid_deadline)
            if not ok:
                cid_failed += 1
                last_reasons.append(f"{modality}:{download_reason}")
                log_failure(cid, cas, download_reason, source_url, modality)
                continue

            path.write_bytes(content)
            cid_success += 1
            result["success"] += 1
            result["success_paths"].append(path)
            write_metadata(
                {
                    "cid": cid,
                    "cas": cas,
                    "modality": modality,
                    "file_path": str(path),
                    "source_url": source_url,
                    "downloaded_at": now_text(),
                }
            )

        if cid_success == 0:
            if cid_existing:
                result.update({"status": "skipped", "reason": f"{cid_existing}_existing_valid_jcamp", "skipped": 1})
            elif cid_failed:
                result.update({"status": "failed", "reason": "all_jcamp_downloads_failed", "failed": 1})
            else:
                result.update({"status": "skipped", "reason": "no_new_downloads", "skipped": 1})
            if last_reasons:
                log_failure(cid, cas, "all_jcamp_downloads_failed", page_url, message=";".join(last_reasons))
        else:
            result.update({"status": "success", "reason": f"{cid_success}_spectra"})

        return result

    except requests.exceptions.Timeout as exc:
        reason = f"cid_timeout:{type(exc).__name__}"
        log_failure(cid, cas, reason, message=str(exc))
        result.update({"status": "failed", "reason": reason, "failed": 1, "cas": cas})
        return result
    except requests.exceptions.ConnectionError as exc:
        reason = f"cid_connection_error:{type(exc).__name__}"
        log_failure(cid, cas, reason, message=str(exc))
        result.update({"status": "failed", "reason": reason, "failed": 1, "cas": cas})
        return result
    except Exception as exc:
        reason = f"unexpected_{type(exc).__name__}"
        log_failure(cid, cas, reason, message=str(exc))
        result.update({"status": "failed", "reason": reason, "failed": 1, "cas": cas})
        return result
    finally:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        result["elapsed"] = time.monotonic() - cid_start


def advance_checkpoint(completed_indices, next_checkpoint_index, cid_by_index):
    while next_checkpoint_index in completed_indices:
        cid = cid_by_index.get(next_checkpoint_index, "")
        save_checkpoint(next_checkpoint_index, cid)
        next_checkpoint_index += 1
    return next_checkpoint_index


def main():
    global CHECKPOINT_PATH, CAS_CACHE_PATH, FAILED_LOG_PATH

    parser = argparse.ArgumentParser(description="Harvest real JCAMP-DX spectra from NIST WebBook.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--cas-cache", type=str, default=str(CAS_CACHE_PATH))
    parser.add_argument("--failed-log", type=str, default=str(FAILED_LOG_PATH))
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent worker threads. Use 1 for serial mode.")
    parser.add_argument("--max-pending", type=int, default=200, help="Maximum pending futures in threaded mode.")
    args = parser.parse_args()

    CHECKPOINT_PATH = Path(args.checkpoint)
    CAS_CACHE_PATH = Path(args.cas_cache)
    FAILED_LOG_PATH = Path(args.failed_log)

    for modality in ("ir", "raman", "uvvis"):
        (OUT_DIR / modality).mkdir(parents=True, exist_ok=True)

    start_index = load_start_index(args)
    df = load_master(args, start_index)
    records = [(global_index, row) for global_index, (_, row) in enumerate(df.iterrows(), start=start_index)]
    total = len(records)
    cid_by_index = {idx: int(row["cid"]) for idx, row in records}
    cas_cache = load_cas_cache()

    stats = {
        "total": 0,
        "cas_found": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
    }
    success_paths = []
    completed_indices = set()
    next_checkpoint_index = start_index
    run_start = time.monotonic()

    try:
        if args.workers <= 1:
            for global_index, row in records:
                result = process_row(global_index, row, cas_cache, args.sleep)
                completed_indices.add(global_index)
                next_checkpoint_index = advance_checkpoint(completed_indices, next_checkpoint_index, cid_by_index)

                stats["total"] += 1
                stats["cas_found"] += result["cas_found"]
                stats["success"] += result["success"]
                stats["failed"] += result["failed"]
                stats["skipped"] += result["skipped"]
                success_paths.extend(result["success_paths"])

                print_progress(stats["total"], total, result["cid"], result["cas"], result["status"], result["reason"], result["elapsed"])
                if stats["total"] % 10 == 0:
                    print_summary(stats)

        else:
            pending = {}
            record_iter = iter(records)

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                while True:
                    while len(pending) < args.max_pending:
                        try:
                            global_index, row = next(record_iter)
                        except StopIteration:
                            break
                        future = executor.submit(process_row, global_index, row, cas_cache, args.sleep)
                        pending[future] = global_index

                    if not pending:
                        break

                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        global_index = pending.pop(future)
                        result = future.result()

                        completed_indices.add(global_index)
                        next_checkpoint_index = advance_checkpoint(completed_indices, next_checkpoint_index, cid_by_index)

                        stats["total"] += 1
                        stats["cas_found"] += result["cas_found"]
                        stats["success"] += result["success"]
                        stats["failed"] += result["failed"]
                        stats["skipped"] += result["skipped"]
                        success_paths.extend(result["success_paths"])

                        print_progress(
                            stats["total"],
                            total,
                            result["cid"],
                            result["cas"],
                            result["status"],
                            result["reason"],
                            result["elapsed"],
                        )

                        if stats["total"] % 10 == 0:
                            print_summary(stats)

                        if stats["total"] % 100 == 0:
                            elapsed = max(1e-6, time.monotonic() - run_start)
                            items_per_min = stats["total"] / elapsed * 60
                            remaining = max(0, total - stats["total"])
                            eta_min = remaining / items_per_min if items_per_min > 0 else 0
                            print(
                                f"Speed: processed={stats['total']}/{total} "
                                f"items/min={items_per_min:.1f} ETA={eta_min/60:.1f}h",
                                flush=True,
                            )

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Saving checkpoint and CAS cache.")
        next_checkpoint_index = advance_checkpoint(completed_indices, next_checkpoint_index, cid_by_index)
        save_cas_cache(cas_cache)
        raise

    save_cas_cache(cas_cache)
    next_checkpoint_index = advance_checkpoint(completed_indices, next_checkpoint_index, cid_by_index)

    print("\nNIST Harvest Summary")
    for key in ("total", "cas_found", "success", "failed", "skipped"):
        print(f"{key}: {stats[key]}")
    print_success_previews(success_paths[:20])


if __name__ == "__main__":
    main()
