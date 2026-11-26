"""cli/g7e_switch.py: find g7e capacity by really launching a probe (spot 8xlarge, spot 4xlarge, else
on-demand 4xlarge), write the choice to terraform.tfvars and apply (docs/specs/2026-11-25-render-on-g7e.md,
criteria 5-12). Run against a fake `aws` / `terraform` / `make`."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cli"))
import g7e_switch as sw  # noqa: E402

TFVARS = '''region            = "us-east-1"
availability_zone = "us-east-1a"
instance_type     = "g6e.4xlarge"
on_demand         = false
train_availability_zone = "us-east-1a"

authorized_ssh_public_keys = [
  "ssh-ed25519 AAAA me",
]
'''


class Cloud:
    """What the fake commands answer. capacity: {(type, zone, "spot"|"on-demand")} that can launch."""

    def __init__(self, capacity=(), queue=(0, 0), mirror=True, apply_fails=(), torch="2.9.0"):
        self.capacity, self.queue, self.mirror, self.apply_fails = set(capacity), queue, mirror, list(apply_fails)
        self.launched, self.terminated, self.calls, self.applied = [], [], [], []
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "terraform"))
        os.makedirs(os.path.join(self.root, "worker"))
        self.tfvars = os.path.join(self.root, "terraform", "terraform.tfvars")
        open(self.tfvars, "w").write(TFVARS)
        open(os.path.join(self.root, "terraform", "variables.tf"), "w").write(
            'variable "project" {\n  default = "ai-video"\n}\nvariable "vpc_name" {\n  type = string\n  default = "the-vpc"\n}\n'
            'variable "subnet_visibility_tag" {\n  default = "private"\n}\n'
            'variable "vlm_model" {\n  default = "Qwen/Qwen2.5-VL-7B-Instruct"\n}\n'
            'variable "edit_model" {\n  default = "Qwen/Qwen-Image-Edit-2511"\n}\n'
            'variable "video_model" {\n  default = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"\n}\n'
            'variable "i2v_model" {\n  default = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"\n}\n')
        open(os.path.join(self.root, "worker", "requirements.txt"), "w").write(f"torch=={torch}\ndiffusers==0.37.1\n")

    def run(self, cmd):
        self.calls.append(cmd)
        a = " ".join(cmd)
        if cmd[0] == "make":
            return 0, "", ""
        if cmd[0] == "terraform":
            if "init" in cmd:
                return 0, "", ""
            if "plan" in cmd:
                return 0, "Plan: 1 to add, 0 to change, 1 to destroy.\n", ""
            if "apply" in cmd:
                now = sw.read_tfvars(self.tfvars)
                choice = (now["instance_type"], now["availability_zone"], now["on_demand"])
                if self.apply_fails and self.apply_fails[0] == choice[:2]:
                    self.apply_fails.pop(0)
                    return 1, "", "Error: creating EC2 Instance: InsufficientInstanceCapacity"
                self.applied.append(choice)
                return 0, "Apply complete!", ""
        if "sts get-caller-identity" in a:
            return 0, "123456789012\n", ""
        if "sqs get-queue-url" in a:
            assert "ai-video-jobs" in a
            return 0, "https://sqs/q\n", ""
        if "sqs get-queue-attributes" in a:
            return 0, f"{self.queue[0]}\t{self.queue[1]}\n", ""
        if "s3api list-objects-v2" in a:
            assert "ai-video-123456789012-us-east-1" in a
            keys = ["models/bf16/Wan2.2-T2V-A14B-Diffusers/.mirror-manifest.json",
                    "models/bf16/Wan2.2-I2V-A14B-Diffusers/.mirror-manifest.json",
                    "models/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/.mirror-manifest.json",
                    "models/hub/models--Qwen--Qwen-Image-Edit-2511/.mirror-manifest.json",
                    "models/mmaudio/.mirror-manifest.json", "models/mmaudio/weights/w.pth"] if self.mirror else []
            return 0, "\n".join(keys) + "\n", ""
        if "describe-images" in a:
            return 0, "ami-1\n", ""
        if "describe-vpcs" in a:
            assert "the-vpc" in a
            return 0, "vpc-1\n", ""
        if "describe-subnets" in a:
            return 0, "subnet-a\tus-east-1a\nsubnet-b\tus-east-1b\nsubnet-d\tus-east-1d\nsubnet-c\tus-east-1c\n", ""
        if "describe-instance-type-offerings" in a:
            return 0, "us-east-1a\nus-east-1b\nus-east-1d\n", ""
        if "run-instances" in a:
            itype = cmd[cmd.index("--instance-type") + 1]
            zone = {"subnet-a": "us-east-1a", "subnet-b": "us-east-1b", "subnet-d": "us-east-1d"}[cmd[cmd.index("--subnet-id") + 1]]
            market = "spot" if "--instance-market-options" in cmd else "on-demand"
            assert "ai-video-probe" in a and "terminate" in a and "--iam-instance-profile" not in cmd
            if (itype, zone, market) not in self.capacity:
                return 254, "", "An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation"
            iid = f"i-{len(self.launched)}"
            self.launched.append((iid, itype, zone, market))
            return 0, iid + "\n", ""
        if "describe-instances" in a:
            return 0, "running\n", ""
        if "terminate-instances" in a:
            self.terminated += [x for x in cmd if x.startswith("i-")]
            return 0, "", ""
        raise AssertionError(f"unexpected command: {a}")


def go(cloud, *args):
    out = []
    code = sw.main(list(args), run=cloud.run, root=cloud.root, say=out.append, wait=lambda s: None)
    return code, "\n".join(out)


def tfvars(cloud):
    return open(cloud.tfvars).read()


# ---- a job queued or running: stops before any probe, saying why (criterion 5)
for q in ((1, 0), (0, 1)):
    c = Cloud(capacity={("g7e.8xlarge", "us-east-1a", "spot")}, queue=q)
    code, out = go(c)
    assert code != 0 and "job" in out and not c.launched and tfvars(c) == TFVARS, out
# ... the same for a worker that can't run on a g7e yet, and for a mirror without the weights
c = Cloud(capacity={("g7e.8xlarge", "us-east-1a", "spot")}, torch="2.5.1")
code, out = go(c)
assert code != 0 and "torch" in out and not c.launched, out
# a model mirror without the weights doesn't stop it (the first move can't have one: the old box never ran
# this code), but it says what that means for the first boot
c = Cloud(capacity={("g7e.8xlarge", "us-east-1a", "spot")}, mirror=False)
code, out = go(c)
assert code == 0 and "mirror" in out and "Hugging Face" in out and c.applied == [("g7e.8xlarge", "us-east-1a", False)], out
# --force goes on although a job is queued
c = Cloud(capacity={("g7e.8xlarge", "us-east-1a", "spot")}, queue=(1, 0))
code, out = go(c, "--force")
assert code == 0 and c.applied == [("g7e.8xlarge", "us-east-1a", False)], out

# the order: spot 4xlarge, spot 8xlarge, on-demand 4xlarge, on-demand 8xlarge
assert sw.CANDIDATES == [("g7e.4xlarge", "spot"), ("g7e.8xlarge", "spot"), ("g7e.4xlarge", "on-demand"),
                         ("g7e.8xlarge", "on-demand")]

# ---- spot 4xlarge launches: chosen, nothing else probed, applied; only three lines change
c = Cloud(capacity={("g7e.4xlarge", "us-east-1b", "spot"), ("g7e.8xlarge", "us-east-1a", "spot"),
                    ("g7e.4xlarge", "us-east-1a", "on-demand")})
code, out = go(c)
assert code == 0 and c.applied == [("g7e.4xlarge", "us-east-1b", False)], out
assert [x[1:] for x in c.launched] == [("g7e.4xlarge", "us-east-1b", "spot")], c.launched
assert c.terminated == ["i-0"], "the probe is terminated"
want = TFVARS.replace('availability_zone = "us-east-1a"', 'availability_zone = "us-east-1b"', 1).replace(
    'instance_type     = "g6e.4xlarge"', 'instance_type     = "g7e.4xlarge"')
assert tfvars(c) == want, tfvars(c)
assert 'train_availability_zone = "us-east-1a"' in tfvars(c)
assert "g7e.4xlarge" in out and "us-east-1b" in out and "spot" in out
# the order of the commands: build, init, plan to a file, apply that file
tf = [x for x in c.calls if x[0] in ("make", "terraform")]
assert tf[0][:2] == ["make", "build"] and "init" in tf[1] and "plan" in tf[2] and "apply" in tf[3] and tf[3][-1] == "tfplan", tf

# ---- no spot 4xlarge anywhere, a spot 8xlarge launches
c = Cloud(capacity={("g7e.8xlarge", "us-east-1d", "spot"), ("g7e.4xlarge", "us-east-1a", "on-demand")})
code, out = go(c)
assert code == 0 and c.applied == [("g7e.8xlarge", "us-east-1d", False)], out

# ---- no spot at all: on-demand 4xlarge in the first zone that launches (criterion 8)
c = Cloud(capacity={("g7e.4xlarge", "us-east-1b", "on-demand"), ("g7e.4xlarge", "us-east-1d", "on-demand")})
code, out = go(c)
assert code == 0 and c.applied == [("g7e.4xlarge", "us-east-1b", True)], out
assert 'on_demand         = true' in tfvars(c) and "on-demand" in out
assert c.terminated == ["i-0"]
# no spot and no on-demand 4xlarge: on-demand 8xlarge is the last option
c = Cloud(capacity={("g7e.8xlarge", "us-east-1d", "on-demand")})
code, out = go(c)
assert code == 0 and c.applied == [("g7e.8xlarge", "us-east-1d", True)], out
# a tfvars file without the on_demand line gets one
c = Cloud(capacity={("g7e.4xlarge", "us-east-1b", "on-demand")})
open(c.tfvars, "w").write(TFVARS.replace("on_demand         = false\n", ""))
code, out = go(c)
assert code == 0 and sw.read_tfvars(c.tfvars)["on_demand"] is True and "authorized_ssh_public_keys" in tfvars(c)

# ---- nothing launches anywhere: non-zero, tfvars untouched, nothing applied (criterion 8)
c = Cloud()
code, out = go(c)
assert code != 0 and tfvars(c) == TFVARS and not c.applied and not [x for x in c.calls if x[0] == "terraform"], out

# ---- --dry-run: says what it would choose, changes nothing (criterion 12)
c = Cloud(capacity={("g7e.4xlarge", "us-east-1a", "spot")})
code, out = go(c, "--dry-run")
assert code == 0 and "g7e.4xlarge" in out and tfvars(c) == TFVARS and not c.applied, out
assert c.terminated == ["i-0"] and not [x for x in c.calls if x[0] in ("terraform", "make")]

# ---- the apply fails for capacity: the next candidate is tried (criterion 11)
c = Cloud(capacity={("g7e.8xlarge", "us-east-1a", "spot"), ("g7e.4xlarge", "us-east-1a", "on-demand")},
          apply_fails=[("g7e.8xlarge", "us-east-1a")])
code, out = go(c)
assert code == 0 and c.applied == [("g7e.4xlarge", "us-east-1a", True)], (out, c.applied)
# ... and when none is left: tfvars put back, non-zero, and it says the render box may be gone
c = Cloud(capacity={("g7e.4xlarge", "us-east-1a", "on-demand")}, apply_fails=[("g7e.4xlarge", "us-east-1a")])
code, out = go(c)
assert code != 0 and tfvars(c) == TFVARS and "make plan" in out, out

# ---- a probe is terminated whatever happens after it launched (criterion 9)
c = Cloud(capacity={("g7e.8xlarge", "us-east-1a", "spot")})
real = c.run


def breaks_after_launch(cmd):
    if "describe-instances" in " ".join(cmd):
        raise KeyboardInterrupt()
    return real(cmd)


c.run = breaks_after_launch
try:
    go(c)
    raise AssertionError("the interrupt was swallowed")
except KeyboardInterrupt:
    pass
assert c.terminated == ["i-0"] and tfvars(c) == TFVARS
# a probe that never reaches running doesn't count, and is terminated
c = Cloud(capacity={("g7e.8xlarge", "us-east-1a", "spot"), ("g7e.4xlarge", "us-east-1a", "on-demand")})
real2 = c.run
c.run = lambda cmd: (0, "pending\n", "") if "describe-instances" in " ".join(cmd) else real2(cmd)
code, out = go(c)
assert code != 0 and set(c.terminated) == {"i-0", "i-1"} and not c.applied, (out, c.terminated)

# ---- tfvars helpers
assert sw.read_tfvars(c.tfvars) == {"region": "us-east-1", "availability_zone": "us-east-1a", "instance_type": "g6e.4xlarge",
                                    "on_demand": False, "train_availability_zone": "us-east-1a"}
vf = os.path.join(tempfile.mkdtemp(), "variables.tf")
open(vf, "w").write('variable "on_demand" {\n  description = "a variable without a string default"\n  type = bool\n  default = false\n}\n\n'
                    'variable "video_model" {\n  description = <<-EOT\n    Text here.\n  EOT\n  type    = string\n  default = "Wan-AI/T2V"\n}\n'
                    'variable "n" {\n  default = 500\n}\nvariable "i2v_model" {\n  default     = "Wan-AI/I2V"\n}\n')
assert sw.read_defaults(vf) == {"video_model": "Wan-AI/T2V", "i2v_model": "Wan-AI/I2V"}, sw.read_defaults(vf)
assert sw.torch_pin("torch==2.9.0+cu128\n") == (2, 9) and sw.torch_pin("# torch\ntorch==2.5.1\n") == (2, 5)
assert sw.torch_pin("diffusers==1\n") is None
print(json.dumps("g7e switch tests passed"))
