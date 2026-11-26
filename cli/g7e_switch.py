#!/usr/bin/env python3
"""Move the render box to a g7e (96 GB GPU): find capacity by really launching, then apply.

    python3 cli/g7e_switch.py [--dry-run] [--force]      (or: make g7e)

1. Preflight: AWS credentials; the worker's torch can run on a g7e (>= 2.7); no job queued or running
   (--force skips those two). It also says whether the model mirror holds the weights (if not, first boot
   downloads them, ~45 min).
2. Capacity: launches a probe instance and terminates it at once - spot g7e.4xlarge in each zone, then spot
   g7e.8xlarge, then on-demand g7e.4xlarge, then on-demand g7e.8xlarge. The first that reaches "running"
   wins. (AWS has no yes/no for spot capacity; a probe costs cents.)
3. Writes instance_type, on_demand and availability_zone to terraform/terraform.tfvars and runs
   `make build`, `terraform plan`, `terraform apply`. THE APPLY REPLACES THE RENDER BOX AND ITS DISK; it
   does not ask again. If the apply fails, the next candidate is tried.

--dry-run stops after step 2 and changes nothing. See docs/specs/2026-11-25-render-on-g7e.md.
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (size, market), in the order they are tried.
CANDIDATES = [("g7e.4xlarge", "spot"), ("g7e.8xlarge", "spot"), ("g7e.4xlarge", "on-demand"), ("g7e.8xlarge", "on-demand")]
AMI_NAME = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"  # terraform/ec2.tf
PROBE_TAGS = "Tags=[{Key=Name,Value=ai-video-probe},{Key=Project,Value=ai-video-trial}]"
PROBE_WAIT = 120  # seconds for a probe to reach "running"
MIN_TORCH = (2, 7)  # the first torch with kernels for the g7e's GPU (built for CUDA 12.8)
KEYS = ("instance_type", "on_demand", "availability_zone")


class Stop(Exception):
    """Preflight said no, or nothing could be launched."""


def shell(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout, p.stderr


def shell_live(cmd, say):
    """Run a command showing its output as it comes (Terraform's apply can take many minutes: it retries a
    launch that is refused for capacity). Returns (code, output, "")."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        if line.strip() and "Refreshing state" not in line and "Reading..." not in line and "Read complete" not in line:
            say("  " + line)
    return proc.wait(), "\n".join(lines), ""


# ---------------------------------------------------------------- terraform.tfvars

def _value(raw):
    raw = raw.strip()
    return {"true": True, "false": False}.get(raw, raw.strip('"'))


def read_tfvars(path):
    """The simple `key = value` lines of a tfvars file (strings and booleans; lists are left alone)."""
    out = {}
    for line in open(path):
        m = re.match(r'^([a-z_]+)\s*=\s*("[^"]*"|true|false)\s*(#.*)?$', line.strip())
        if m:
            out[m[1]] = _value(m[2])
    return out


def read_defaults(path):
    """variable name -> its string default in variables.tf (variables with another kind of default are left out)."""
    out = {}
    blocks = re.split(r'^variable "([a-z_0-9]+)" \{', open(path).read(), flags=re.M)
    for name, body in zip(blocks[1::2], blocks[2::2]):
        m = re.search(r'^\s*default\s*=\s*"([^"]*)"\s*$', body, re.M)
        if m:
            out[name] = m[1]
    return out


def write_tfvars(path, changes):
    """Set `key = value` lines, keeping every other line; a key that isn't there is added at the top."""
    lines = open(path).read().splitlines(keepends=True)
    for key, value in changes.items():
        text = "true" if value is True else "false" if value is False else f'"{value}"'
        for i, line in enumerate(lines):
            m = re.match(rf"^({key}\s*=\s*)(\"[^\"]*\"|true|false)(.*)$", line.rstrip("\n"))
            if m:
                lines[i] = f"{m[1]}{text}{m[3]}\n"
                break
        else:
            lines.insert(0, f"{key} = {text}\n")
    with open(path, "w") as f:
        f.writelines(lines)


def torch_pin(requirements):
    """(major, minor) of the `torch==` pin in a requirements file, or None."""
    m = re.search(r"^torch==(\d+)\.(\d+)", requirements, re.M)
    return (int(m[1]), int(m[2])) if m else None


# ---------------------------------------------------------------- the switch

class Switch:
    def __init__(self, run, root, say, wait, live=None):
        self.run, self.root, self.say, self.wait = run, root, say, wait
        self.live = live  # runs a command showing its output as it comes; None: like `run`
        self.tfvars = os.path.join(root, "terraform", "terraform.tfvars")
        self.vars = {**read_defaults(os.path.join(root, "terraform", "variables.tf")), **read_tfvars(self.tfvars)}
        self.region = self.vars.get("region", "us-east-1")
        self.probes = []  # instance ids launched and not yet terminated

    def aws(self, *args, check=True):
        code, out, err = self.run(["aws", *args, "--region", self.region, "--output", "text"])
        if code and check:
            raise Stop(f"aws {' '.join(args[:2])} failed: {err.strip()[:300]}")
        return code, out.strip(), err

    # -- 1. preflight
    def preflight(self, force):
        _, account, _ = self.aws("sts", "get-caller-identity", "--query", "Account")
        problems = []
        pin = torch_pin(open(os.path.join(self.root, "worker", "requirements.txt")).read())
        if pin is None or pin < MIN_TORCH:
            problems.append(f"the worker's torch pin ({'.'.join(map(str, pin)) if pin else 'none'}) can't run on a "
                            f"g7e GPU: it needs torch >= {MIN_TORCH[0]}.{MIN_TORCH[1]} (worker/requirements.txt)")
        project = self.vars.get("project", "ai-video")
        _, url, _ = self.aws("sqs", "get-queue-url", "--queue-name", f"{project}-jobs", "--query", "QueueUrl")
        _, counts, _ = self.aws("sqs", "get-queue-attributes", "--queue-url", url, "--attribute-names",
                                "ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible", "--query",
                                "Attributes.[ApproximateNumberOfMessages,ApproximateNumberOfMessagesNotVisible]")
        queued, running = (int(x) for x in counts.split())
        if queued or running:
            problems.append(f"a job is {'running' if running else 'queued'} ({queued} queued, {running} running): "
                            "replacing the box now would lose it")
        bucket = f"{project}-{account}-{self.region}"
        _, keys, _ = self.aws("s3api", "list-objects-v2", "--bucket", bucket, "--prefix", "models/", "--query",
                              "Contents[].Key")
        have = {k[len("models/"):-len("/.mirror-manifest.json")] for k in keys.split() if k.endswith("/.mirror-manifest.json")}
        need = [f"bf16/{self.vars[v].split('/')[-1]}" for v in ("video_model", "i2v_model") if v in self.vars]
        need += [f"hub/models--{self.vars[v].replace('/', '--')}" for v in ("vlm_model", "edit_model") if v in self.vars]
        missing = [u for u in need + ["mmaudio"] if u not in have]
        if missing:  # not a reason to stop: the first move can't have a mirror (the old box never ran this code)
            self.say("note: the model mirror doesn't hold " + ", ".join(missing) + ", so the new disk downloads those "
                     "from Hugging Face and converts Wan again at first boot (~45 min); the new box then fills "
                     "the mirror for next time")
        if problems and not force:
            raise Stop("Not switching:\n- " + "\n- ".join(problems) + "\n(--force goes on anyway.)")
        for p in problems:
            self.say(f"warning (--force): {p}")

    # -- 2. capacity
    def lookups(self):
        _, self.ami, _ = self.aws("ec2", "describe-images", "--owners", "amazon", "--filters",
                                  f"Name=name,Values={AMI_NAME}", "Name=architecture,Values=x86_64",
                                  "--query", "sort_by(Images,&CreationDate)[-1].ImageId")
        _, vpc, _ = self.aws("ec2", "describe-vpcs", "--filters", f"Name=tag:Name,Values={self.vars['vpc_name']}",
                             "--query", "Vpcs[0].VpcId")
        _, subnets, _ = self.aws("ec2", "describe-subnets", "--filters", f"Name=vpc-id,Values={vpc}",
                                 f"Name=tag:Visibility,Values={self.vars.get('subnet_visibility_tag', 'private')}",
                                 "--query", "Subnets[].[SubnetId,AvailabilityZone]")
        self.subnets = {line.split()[1]: line.split()[0] for line in subnets.splitlines() if line.strip()}
        # The project's own security group (no inbound): in a shared VPC there is no default one to launch into.
        _, self.group, _ = self.aws("ec2", "describe-security-groups", "--filters", f"Name=vpc-id,Values={vpc}",
                                    f"Name=group-name,Values={self.vars.get('project', 'ai-video')}-worker",
                                    "--query", "SecurityGroups[0].GroupId")
        if not self.group.startswith("sg-"):
            raise Stop("the project's security group (<project>-worker) was not found: has `make apply` run once?")

    def zones(self, itype):
        _, out, _ = self.aws("ec2", "describe-instance-type-offerings", "--location-type", "availability-zone",
                             "--filters", f"Name=instance-type,Values={itype}", "--query",
                             "InstanceTypeOfferings[].Location")
        return sorted(z for z in out.split() if z in self.subnets)

    def probe(self, itype, zone, market):
        """Launch one instance of that kind and terminate it. True when it reached "running"."""
        cmd = ["ec2", "run-instances", "--image-id", self.ami, "--instance-type", itype, "--subnet-id",
               self.subnets[zone], "--security-group-ids", self.group, "--count", "1",
               "--instance-initiated-shutdown-behavior", "terminate",
               "--tag-specifications", f"ResourceType=instance,{PROBE_TAGS}", f"ResourceType=volume,{PROBE_TAGS}",
               "--query", "Instances[0].InstanceId"]
        if market == "spot":
            cmd += ["--instance-market-options",
                    "MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}"]
        code, iid, err = self.aws(*cmd, check=False)
        if code or not iid.startswith("i-"):
            reason = (re.search(r"\((\w+)\)", err) or [None, "failed"])[1]
            self.say(f"  {market} {itype} in {zone}: no ({reason})")
            if "Capacity" not in reason and "SpotMaxPrice" not in reason and "InstanceLimit" not in reason \
                    and "MaxSpotInstanceCount" not in reason:
                # not "no capacity": something is wrong with the request itself, and it would be for every probe
                raise Stop(f"the probe launch was refused for another reason than capacity: {err.strip()[-300:]}")
            return False
        self.probes.append(iid)
        try:
            state, waited = "", 0
            while waited <= PROBE_WAIT:
                _, state, _ = self.aws("ec2", "describe-instances", "--instance-ids", iid, "--query",
                                       "Reservations[0].Instances[0].State.Name", check=False)
                if state != "pending":
                    break
                self.wait(5)
                waited += 5
            ok = state == "running"
            self.say(f"  {market} {itype} in {zone}: {'yes' if ok else 'no (' + (state or 'unknown') + ')'}")
            return ok
        finally:
            self.cleanup()

    def cleanup(self):
        """Terminate every probe still alive (also on Ctrl-C or an error)."""
        while self.probes:
            iid = self.probes[-1]
            code, _, err = self.aws("ec2", "terminate-instances", "--instance-ids", iid, check=False)
            if code:
                self.say(f"COULD NOT TERMINATE probe {iid}: {err.strip()[:200]} - terminate it by hand")
            self.probes.pop()

    def candidates(self):
        """(size, zone, market) that launched, in order of preference (found one at a time)."""
        for itype, market in CANDIDATES:
            for zone in self.zones(itype):
                if self.probe(itype, zone, market):
                    yield itype, zone, market

    # -- 3. apply
    def apply(self, itype, zone, market):
        write_tfvars(self.tfvars, {"instance_type": itype, "on_demand": market == "on-demand", "availability_zone": zone})
        for cmd in (["make", "build"],
                    ["terraform", "-chdir=terraform", "init", "-input=false"],
                    ["terraform", "-chdir=terraform", "plan", "-input=false", "-out", "tfplan"]):
            code, out, err = self.run(cmd)
            if code:
                self.say(f"{' '.join(cmd[:3])} failed:\n{(err or out).strip()[-1500:]}")
                return False
            for line in out.splitlines():
                if line.startswith("Plan:") or "must be replaced" in line or "will be destroyed" in line:
                    self.say("  " + line.strip())
        self.say("applying (this replaces the render box and its disk; the old box goes first, and Terraform "
                 "retries the new launch for up to ~10 minutes if AWS refuses it for capacity)")
        cmd = ["terraform", "-chdir=terraform", "apply", "-input=false", "tfplan"]
        code, out, err = self.live(cmd, self.say) if self.live else self.run(cmd)
        if code:
            self.say(f"terraform apply failed:\n{(err or out).strip()[-1500:]}")
        return not code


def main(argv, run=shell, root=ROOT, say=print, wait=time.sleep):
    ap = argparse.ArgumentParser(description="Move the render box to a g7e: probe capacity, then apply.")
    ap.add_argument("--dry-run", action="store_true", help="probe and report; change nothing")
    ap.add_argument("--force", action="store_true", help="go on although preflight found a problem")
    a = ap.parse_args(argv)
    sw = Switch(run, root, say, wait, live=shell_live if run is shell else None)
    before = open(sw.tfvars).read()
    tried_apply = False
    try:
        sw.preflight(a.force)
        sw.lookups()
        say("looking for g7e capacity (each check launches an instance and terminates it):")
        for itype, zone, market in sw.candidates():
            say(f"chosen: {market} {itype} in {zone}")
            if a.dry_run:
                say("--dry-run: nothing changed")
                return 0
            tried_apply = True
            if sw.apply(itype, zone, market):
                say(f"done: the render box is now a {market} {itype} in {zone}. First boot restores the weights; "
                    "watch it with the commands in RUNBOOK 1.4. Jobs sent meanwhile wait in the queue.")
                return 0
            say("trying the next option")
        raise Stop("No g7e could be launched right now (spot or on-demand, 4xlarge or 8xlarge)."
                   if not tried_apply else "Every option that launched as a probe failed to apply.")
    except Stop as e:
        say(str(e))
        if open(sw.tfvars).read() != before:
            with open(sw.tfvars, "w") as f:
                f.write(before)
            say("terraform.tfvars is back as it was.")
        if tried_apply:
            say("A failed apply may already have removed the old render box: run `make plan && make apply` to "
                "bring it back with the settings in terraform.tfvars.")
        return 1
    finally:
        sw.cleanup()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
