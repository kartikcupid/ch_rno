#!/usr/bin/env python3
"""Report used / available storage on an HPC.

For each interesting directory ($HOME, $SCRATCH, cwd, ...) prints the filesystem
size, used, available and inode counts, then what you are *allowed* to use: your
user / group / fileset quota via whichever tool matches the filesystem (GPFS,
Lustre, BeeGFS, NFS).  The filesystem totals are shared by everyone; the quota is
the limit that actually stops your writes.

Usage:
    python3 hpc_storage.py                 # standard locations
    python3 hpc_storage.py /path/a /path/b # plus extra paths
    python3 hpc_storage.py --du ~/runs     # also `du` those paths (slow on shared fs)

Written for Python >= 3.6 (older HPC login-node pythons).
"""
import getpass
import grp
import os
import shutil
import subprocess
import sys

# Env vars that commonly point at the big filesystems on HPC systems.
ENV_VARS = ["HOME", "SCRATCH", "WORK", "PROJECT", "PROJECTDIR", "LUSTRE", "TMPDIR"]

# Quota tools are often installed off the default PATH (GPFS: /usr/lpp/mmfs/bin).
TOOL_DIRS = ["/usr/lpp/mmfs/bin", "/opt/beegfs/sbin"]


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024 or unit == "PB":
            return "%d B" % n if unit == "B" else "%.1f %s" % (n, unit)
        n /= 1024.0


def run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=timeout)
        return p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return "<%s failed: %s>" % (cmd[0], e)


def find_tool(name):
    """`name` on PATH, else in the usual off-PATH install dirs, else None."""
    found = shutil.which(name)
    if found:
        return found
    for d in TOOL_DIRS:
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def quota_like_tools():
    """Every command named *quota* (catches site wrappers such as myquota)."""
    names = set()
    for d in os.environ.get("PATH", "").split(os.pathsep) + TOOL_DIRS:
        try:
            names.update(n for n in os.listdir(d) if "quota" in n.lower())
        except OSError:
            pass
    return sorted(names)


def mount_info(path):
    """(filesystem type, device) of `path` from /proc/mounts (longest matching
    mount)."""
    best, info = "", ("?", "")
    try:
        with open("/proc/mounts") as f:
            for line in f:
                dev, mnt, typ = line.split()[:3]
                mnt = mnt.replace("\\040", " ")
                inside = path == mnt or path.startswith(mnt.rstrip("/") + "/")
                if inside and len(mnt) > len(best):
                    best, info = mnt, (typ, dev)
    except OSError:
        pass
    return info


def targets(extra):
    """(labels, path) per distinct filesystem, so a shared mount is reported once."""
    cands = [("cwd", os.getcwd())]
    cands += [("$" + v, os.environ[v]) for v in ENV_VARS if os.environ.get(v)]
    cands += [("/tmp", "/tmp")] + [(p, p) for p in extra]
    by_dev = {}
    for label, p in cands:
        p = os.path.realpath(p)
        if os.path.isdir(p):
            labels, _ = by_dev.setdefault(os.stat(p).st_dev, ([], p))
            labels.append(label)
    return list(by_dev.values())


def report_fs(labels, path):
    s = os.statvfs(path)
    total = s.f_blocks * s.f_frsize
    used = (s.f_blocks - s.f_bfree) * s.f_frsize
    avail = s.f_bavail * s.f_frsize          # what a non-root user can still write
    pct = 100.0 * used / total if total else 0.0
    print("%s  ->  %s  [%s]" % (" ".join(labels), path, mount_info(path)[0]))
    print("  size %s | used %s (%.0f%%) | available %s"
          % (human(total), human(used), pct, human(avail)))
    if s.f_files:
        print("  inodes: %s used of %s" % (format(s.f_files - s.f_ffree, ","),
                                           format(s.f_files, ",")))


def group_names(path):
    """Your primary group and the group that owns `path` (group quotas are per
    group)."""
    names = []
    for gid in (os.getgid(), os.stat(path).st_gid):
        try:
            name = grp.getgrgid(gid).gr_name
        except KeyError:
            continue
        if name not in names:
            names.append(name)
    return names


def gpfs_fileset(path):
    """Fileset containing `path`; GPFS home/scratch quotas are often set per fileset."""
    mmlsattr = find_tool("mmlsattr")
    if mmlsattr:
        for line in run([mmlsattr, "-L", path]).splitlines():
            if line.strip().startswith("fileset name:"):
                return line.split(":", 1)[1].strip()
    return None


def report_quotas(paths):
    """What you are allowed to use.  Raw tool output: formats differ per site."""
    user = getpass.getuser()
    shown, missing = set(), []

    def show(cmd):
        if tuple(cmd) not in shown:
            shown.add(tuple(cmd))
            print("\n$ " + " ".join(cmd))
            print(run(cmd))

    print("\n=== Quota: what you are allowed to use (user %s) ===" % user)
    for p in paths:
        kind, dev = mount_info(p)
        if kind in ("gpfs", "mmfs"):
            mmlsquota = find_tool("mmlsquota")
            if not mmlsquota:
                missing.append((p, kind, "mmlsquota"))
                continue
            dev = dev.split("/")[-1]

            def mmq(who, name, target):
                return [mmlsquota, who, name, "--block-size", "auto", target]

            show(mmq("-u", user, dev))
            for g in group_names(p):
                show(mmq("-g", g, dev))
            fileset = gpfs_fileset(p)
            if fileset:
                show(mmq("-j", fileset, dev))
        elif kind == "lustre":
            lfs = find_tool("lfs")
            if lfs:
                show([lfs, "quota", "-h", "-u", user, p])
            else:
                missing.append((p, kind, "lfs"))
        elif kind == "beegfs":
            ctl = find_tool("beegfs-ctl")
            if ctl:
                show([ctl, "--getquota", "--uid", user])
            else:
                missing.append((p, kind, "beegfs-ctl"))
        elif kind.startswith("nfs"):
            quota = find_tool("quota")
            if quota:
                show([quota, "-s"])
            else:
                missing.append((p, kind, "quota"))

    for p, kind, tool in missing:
        print("\nNo `%s` found for %s (%s): cannot read the quota there."
              % (tool, p, kind))
    tools = quota_like_tools()
    print("\nQuota-related commands visible here: %s" % (", ".join(tools) or "none"))
    print("If none of the above shows a limit, ask your HPC admins or check "
          "the site docs.")


def main():
    args = sys.argv[1:]
    do_du = "--du" in args
    extra = [a for a in args if a != "--du"]
    found = targets(extra)
    for labels, path in found:
        report_fs(labels, path)
    report_quotas([p for _, p in found])
    if do_du:
        for p in extra:
            print("\n$ du -sh %s" % p)
            print(run(["du", "-sh", p], timeout=600))


if __name__ == "__main__":
    main()
