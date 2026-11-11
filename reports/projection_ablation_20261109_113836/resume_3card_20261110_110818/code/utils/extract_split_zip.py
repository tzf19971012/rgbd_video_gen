#!/usr/bin/env python3
"""
Extract a multi-disk split zip (z01, z02, ..., zNN, .zip) without 7z or
the standard zipfile module, which does not support multi-volume zips.

Algorithm:
  1. Open all disk files as a virtual concatenated stream.
  2. Find and parse the EOCD / EOCD64 record in the last disk.
  3. Parse every central directory entry, resolving disk-relative offsets
     to absolute positions in the concatenated stream.
  4. Read and decompress each file (STORED or DEFLATED), writing to out_dir.

Usage:
  python utils/extract_split_zip.py \
      --dir /path/to/BlendedMVS \
      --stem BlendedMVS \
      --out_dir /path/to/BlendedMVS
"""

import argparse
import glob
import os
import struct
import zlib

# ---------------------------------------------------------------------------
# ZIP format constants
# ---------------------------------------------------------------------------
LOCAL_SIG           = b"PK\x03\x04"
CD_SIG              = b"PK\x01\x02"
EOCD_SIG            = b"PK\x05\x06"
EOCD64_LOCATOR_SIG  = b"PK\x06\x07"
EOCD64_SIG          = b"PK\x06\x06"

def _u16(b, o): return struct.unpack_from("<H", b, o)[0]
def _u32(b, o): return struct.unpack_from("<I", b, o)[0]
def _u64(b, o): return struct.unpack_from("<Q", b, o)[0]


# ---------------------------------------------------------------------------
# Multi-disk reader
# ---------------------------------------------------------------------------

class MultiDiskReader:
    def __init__(self, disk_paths):
        self._paths = disk_paths
        self._sizes = [os.path.getsize(p) for p in disk_paths]
        self._starts = []
        pos = 0
        for s in self._sizes:
            self._starts.append(pos)
            pos += s
        self._total = pos

    def read_disk(self, disk_idx, disk_offset, length):
        """Read bytes from disk_idx starting at disk_offset (may span disks)."""
        result = bytearray()
        remaining = length
        d, off = disk_idx, disk_offset
        while remaining > 0 and d < len(self._paths):
            available = self._sizes[d] - off
            chunk_len = min(remaining, available)
            with open(self._paths[d], "rb") as f:
                f.seek(off)
                chunk = f.read(chunk_len)
            result.extend(chunk)
            remaining -= len(chunk)
            d += 1
            off = 0
        return bytes(result)

    def abs_offset(self, disk_idx, disk_offset):
        return self._starts[disk_idx] + disk_offset

    def read_abs(self, abs_offset, length):
        disk = self._find_disk(abs_offset)
        disk_offset = abs_offset - self._starts[disk]
        return self.read_disk(disk, disk_offset, length)

    def _find_disk(self, abs_pos):
        lo, hi = 0, len(self._starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._starts[mid] <= abs_pos:
                lo = mid
            else:
                hi = mid - 1
        return lo


# ---------------------------------------------------------------------------
# EOCD / CD parsing
# ---------------------------------------------------------------------------

def parse_eocd(reader):
    last_path = reader._paths[-1]
    last_size = reader._sizes[-1]

    search = min(last_size, 65536 + 22)
    with open(last_path, "rb") as f:
        f.seek(last_size - search)
        tail = f.read(search)

    idx = tail.rfind(EOCD_SIG)
    if idx == -1:
        raise ValueError("EOCD not found in last disk")

    e = tail[idx:]
    cd_start_disk  = _u16(e, 6)
    num_entries    = _u16(e, 10)
    cd_size        = _u32(e, 12)
    cd_disk_offset = _u32(e, 16)

    # ZIP64 locator?
    locator_idx = tail.rfind(EOCD64_LOCATOR_SIG)
    if 0 <= locator_idx < idx:
        loc = tail[locator_idx:]
        eocd64_disk   = _u32(loc, 4)
        eocd64_offset = _u64(loc, 8)
        abs64 = reader.abs_offset(eocd64_disk, eocd64_offset)
        eocd64 = reader.read_abs(abs64, 56)
        if eocd64[:4] == EOCD64_SIG:
            num_entries    = _u64(eocd64, 32)
            cd_size        = _u64(eocd64, 40)
            cd_disk_offset = _u64(eocd64, 48)
            cd_start_disk  = _u32(eocd64, 20)

    return cd_start_disk, cd_disk_offset, cd_size, num_entries


def parse_central_directory(reader, cd_disk, cd_disk_offset, cd_size):
    cd_data = reader.read_disk(cd_disk, cd_disk_offset, cd_size)
    entries = []
    pos = 0
    while pos + 46 <= len(cd_data):
        if cd_data[pos:pos+4] != CD_SIG:
            break

        compression  = _u16(cd_data, pos + 10)
        comp_size    = _u32(cd_data, pos + 20)
        uncomp_size  = _u32(cd_data, pos + 24)
        fname_len    = _u16(cd_data, pos + 28)
        extra_len    = _u16(cd_data, pos + 30)
        comment_len  = _u16(cd_data, pos + 32)
        file_disk    = _u16(cd_data, pos + 34)
        local_offset = _u32(cd_data, pos + 42)
        flags        = _u16(cd_data, pos +  8)

        extra = cd_data[pos+46+fname_len : pos+46+fname_len+extra_len]
        if (comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF
                or local_offset == 0xFFFFFFFF or file_disk == 0xFFFF):
            ep = 0
            while ep + 4 <= len(extra):
                tag = _u16(extra, ep); esz = _u16(extra, ep + 2)
                if tag == 1:
                    vp = ep + 4
                    if uncomp_size  == 0xFFFFFFFF and vp + 8 <= ep+4+esz: uncomp_size  = _u64(extra, vp); vp += 8
                    if comp_size    == 0xFFFFFFFF and vp + 8 <= ep+4+esz: comp_size    = _u64(extra, vp); vp += 8
                    if local_offset == 0xFFFFFFFF and vp + 8 <= ep+4+esz: local_offset = _u64(extra, vp); vp += 8
                    if file_disk    == 0xFFFF     and vp + 4 <= ep+4+esz: file_disk    = _u32(extra, vp)
                    break
                ep += 4 + esz

        fname_bytes = cd_data[pos+46 : pos+46+fname_len]
        try:
            fname = fname_bytes.decode("utf-8")
        except UnicodeDecodeError:
            fname = fname_bytes.decode("latin-1")

        entries.append(dict(name=fname, disk=file_disk, local_offset=local_offset,
                            comp_size=comp_size, uncomp_size=uncomp_size,
                            compression=compression, flags=flags))
        pos += 46 + fname_len + extra_len + comment_len

    return entries


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _data_start(reader, disk, local_offset):
    hdr = reader.read_disk(disk, local_offset, 30)
    if hdr[:4] != LOCAL_SIG:
        raise ValueError(f"Bad local header at disk={disk} off={local_offset:#x}")
    return local_offset + 30 + _u16(hdr, 26) + _u16(hdr, 28)


def extract(disk_dir, stem, out_dir, force=False, verbose=True):
    pattern = os.path.join(disk_dir, f"{stem}.z[0-9][0-9]")
    parts   = sorted(glob.glob(pattern))
    last    = os.path.join(disk_dir, f"{stem}.zip")
    if not parts:
        raise FileNotFoundError(f"No parts matching {pattern}")
    if not os.path.isfile(last):
        raise FileNotFoundError(f"Missing: {last}")

    all_disks = parts + [last]
    print(f"Disks ({len(all_disks)}):")
    for d in all_disks:
        print(f"  {os.path.basename(d)}  {os.path.getsize(d):>14,} bytes")

    reader = MultiDiskReader(all_disks)
    cd_disk, cd_off, cd_size, approx_n = parse_eocd(reader)
    print(f"\nCD: disk={cd_disk}, disk_offset={cd_off:#x}, size={cd_size:#x}, ~{approx_n} entries")

    entries = parse_central_directory(reader, cd_disk, cd_off, cd_size)
    n = len(entries)
    print(f"Parsed {n} entries. Extracting to: {out_dir}\n")
    os.makedirs(out_dir, exist_ok=True)

    done = skipped = errors = 0
    for i, e in enumerate(entries):
        if verbose and i % 2000 == 0:
            print(f"  [{i}/{n}] {e['name']}")

        name = e["name"]
        out_path = os.path.join(out_dir, name)

        if name.endswith("/"):
            os.makedirs(out_path, exist_ok=True)
            continue

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        if (not force and os.path.isfile(out_path)
                and os.path.getsize(out_path) == e["uncomp_size"]):
            skipped += 1
            continue

        try:
            data_off   = _data_start(reader, e["disk"], e["local_offset"])
            compressed = reader.read_disk(e["disk"], data_off, e["comp_size"])
            if e["compression"] == 0:
                raw = compressed
            elif e["compression"] == 8:
                raw = zlib.decompress(compressed, -15)
            else:
                print(f"  [skip] compression={e['compression']}: {name}")
                continue
            with open(out_path, "wb") as f:
                f.write(raw)
            done += 1
        except Exception as exc:
            print(f"  [error] {name}: {exc}")
            errors += 1

    print(f"\nExtraction complete: {done} extracted, {skipped} skipped, {errors} errors.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir",     required=True)
    p.add_argument("--stem",    default="BlendedMVS")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--force",   action="store_true")
    args = p.parse_args()
    extract(args.dir, args.stem, args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
