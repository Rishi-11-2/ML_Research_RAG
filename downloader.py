#!/usr/bin/env python3
"""
download_papers.py

Download PDFs from a list of URLs and store them in a target directory.

Usage:
    python download_papers.py --input urls.txt --dest ./papers --workers 8

Input file format (urls.txt): one URL per line (ignore blank lines and lines starting with #).
"""

import argparse
import os
import re
import hashlib
import tempfile
import shutil
import time
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Optional: nice progress bars if tqdm is installed
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Try to import a PDF metadata reader; if unavailable, we'll skip title extraction.
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def safe_filename(name: str) -> str:
    """Sanitize filename to remove unsafe characters."""
    name = unquote(name)
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    if len(name) == 0:
        name = "file"
    return name


def filename_from_cd(cd: str) -> str:
    """
    Parse filename from content-disposition header.
    Example header: 'attachment; filename="paper.pdf"'
    """
    if not cd:
        return None
    m = re.search(r'filename\*=([^;]+)', cd, flags=re.IGNORECASE)
    if m:
        # RFC 5987: filename*=utf-8''fname.pdf
        val = m.group(1).strip()
        if "''" in val:
            val = val.split("''", 1)[1]
        return safe_filename(val.strip(' "'))
    m = re.search(r'filename=([^;]+)', cd, flags=re.IGNORECASE)
    if m:
        return safe_filename(m.group(1).strip(' "'))
    return None


def name_from_url(url: str) -> str:
    p = urlparse(url)
    candidate = os.path.basename(p.path) or p.netloc
    if candidate == "":
        candidate = hashlib.sha1(url.encode()).hexdigest()
    return safe_filename(candidate)


def ensure_pdf_extension(name: str) -> str:
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def _extract_pdf_title(path: str):
    """
    Try to read the PDF metadata title using PyPDF2 (if available).
    Returns a sanitized filename (without path) or None.
    """
    if PdfReader is None:
        return None
    try:
        reader = PdfReader(path)
        # PyPDF2 metadata is available at reader.metadata in newer versions
        meta = getattr(reader, "metadata", None)
        title = None
        if meta:
            # meta may be an object with attributes or a dict-like mapping
            # Try common accessors
            title = getattr(meta, "title", None)
            if not title:
                # meta might be a dictionary with keys like '/Title'
                try:
                    title = meta.get("/Title") if isinstance(meta, dict) else None
                except Exception:
                    title = None
        # As a fallback, some PDFs expose documentInfo
        if not title and hasattr(reader, "documentInfo"):
            try:
                dinfo = reader.documentInfo
                if dinfo:
                    title = getattr(dinfo, "title", None)
                    if not title:
                        try:
                            title = dinfo.get("/Title") if isinstance(dinfo, dict) else None
                        except Exception:
                            title = None
            except Exception:
                title = None
        if title:
            title = str(title).strip()
            if title:
                return safe_filename(title)
    except Exception:
        # any error reading PDF metadata -> ignore and fallback
        return None
    return None


def download_pdf(url: str, dest_dir: str, session: requests.Session, timeout: int = 20, retries: int = 3, backoff: float = 1.0):
    """
    Download a single URL and save it into dest_dir.
    Returns (url, filepath, success, message)
    """
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            resp = session.get(url, stream=True, timeout=timeout)
        except requests.RequestException as e:
            msg = f"Request failed (attempt {attempt}/{retries}): {e}"
            if attempt < retries:
                time.sleep(backoff * attempt)
                continue
            return (url, None, False, msg)

        status = resp.status_code
        if status >= 400:
            msg = f"HTTP {status}"
            if attempt < retries:
                time.sleep(backoff * attempt)
                continue
            return (url, None, False, msg)

        content_type = resp.headers.get("Content-Type", "")
        # Some servers use "application/octet-stream" or no content type -> still try to save but warn
        if "pdf" not in content_type.lower() and "application/octet-stream" not in content_type.lower():
            # Not necessarily fatal: some pdfs come back without content-type; we will still try but mark
            # If certain you only want to download when content-type includes pdf, fail here.
            # For now, just warn and continue to attempt to save.
            pass

        # Determine filename
        cd = resp.headers.get("Content-Disposition")
        fname = filename_from_cd(cd) or name_from_url(url)
        fname = ensure_pdf_extension(fname)

        dest_path = os.path.join(dest_dir, fname)
        # if file already exists, create a unique name
        base, ext = os.path.splitext(dest_path)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = f"{base}__{counter}{ext}"
            counter += 1

        # stream to temp file then move
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=dest_dir) as tmpf:
                chunk_size = 64 * 1024
                total_written = 0
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmpf.write(chunk)
                        total_written += len(chunk)
                tmpf.flush()
                tmpname = tmpf.name

            # very simple sanity check: ensure file isn't ridiculously small
            tmp_size = os.path.getsize(tmpname)
            if tmp_size < 1024:  # <1KB
                # maybe not a pdf; try again or fail
                os.remove(tmpname)
                msg = f"Downloaded file too small ({tmp_size} bytes)."
                if attempt < retries:
                    time.sleep(backoff * attempt)
                    continue
                return (url, None, False, msg)

            # Try to extract title from PDF metadata and use it as filename if present
            title_name = _extract_pdf_title(tmpname)
            if title_name:
                # create filename from title
                fname_from_title = ensure_pdf_extension(title_name)
                dest_path = os.path.join(dest_dir, fname_from_title)
                base, ext = os.path.splitext(dest_path)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = f"{base}__{counter}{ext}"
                    counter += 1

            # final move
            shutil.move(tmpname, dest_path)
            return (url, dest_path, True, "OK")
        except Exception as e:
            # cleanup and retry if allowed
            try:
                if 'tmpname' in locals() and os.path.exists(tmpname):
                    os.remove(tmpname)
            except Exception:
                pass
            msg = f"Failed saving file: {e}"
            if attempt < retries:
                time.sleep(backoff * attempt)
                continue
            return (url, None, False, msg)

    return (url, None, False, "Exceeded retries")


def load_urls(input_path: str):
    urls = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def main(args):
    os.makedirs(args.dest, exist_ok=True)
    urls = load_urls(args.input)
    if len(urls) == 0:
        print("No URLs found in", args.input)
        return

    session = requests.Session()
    session.headers.update({
        "User-Agent": args.user_agent
    })

    results = []
    if args.workers and args.workers > 1:
        # parallel
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(download_pdf, url, args.dest, session, args.timeout, args.retries, args.backoff): url for url in urls}
            if tqdm and args.show_progress:
                pbar = tqdm(total=len(futures), desc="Downloading", unit="file")
            else:
                pbar = None

            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()
    else:
        # sequential
        iterable = urls
        if tqdm and args.show_progress:
            iterable = tqdm(urls, desc="Downloading", unit="file")
        for url in iterable:
            res = download_pdf(url, args.dest, session, args.timeout, args.retries, args.backoff)
            results.append(res)

    # summary
    ok = [r for r in results if r[2]]
    bad = [r for r in results if not r[2]]

    print(f"\nFinished. Success: {len(ok)}. Failed: {len(bad)}.")
    if len(ok) > 0:
        print("\nSaved files:")
        for _, path, _, _ in ok:
            print("  ", path)
    if len(bad) > 0:
        print("\nFailures:")
        for url, _, _, msg in bad:
            print("  ", url, "->", msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PDF files from URLs into a directory.")
    parser.add_argument("--input", "-i", required=True, help="Path to text file containing PDF URLs (one per line).")
    parser.add_argument("--dest", "-d", default="./papers", help="Destination directory (default ./papers).")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Parallel downloads (default 4). Set 1 to disable concurrency.")
    parser.add_argument("--timeout", type=int, default=30, help="Per-request timeout seconds (default 10).")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries on failure (default 5).")
    parser.add_argument("--backoff", type=float, default=1.0, help="Backoff multiplier between retries (default 1.0).")
    parser.add_argument("--user-agent", default="pdf-downloader/1.0 (+https://example.com)", help="User-Agent header.")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", help="Disable tqdm progress bar if installed.")
    args = parser.parse_args()
    main(args)
