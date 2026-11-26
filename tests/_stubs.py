"""Shared setup for the unit tests: stub out GPU/cloud libraries and set the env the modules
read at import. Each test file runs in its own process (`make test`)."""
import json
import os
import sys
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAVY = ["torch", "torch.nn", "torch.nn.functional", "PIL", "PIL.Image", "PIL.ImageOps", "imageio", "numpy",
         "boto3", "boto3.dynamodb", "boto3.dynamodb.conditions", "botocore", "botocore.config",
         "botocore.exceptions", "cryptography", "cryptography.exceptions", "cryptography.hazmat",
         "cryptography.hazmat.primitives", "cryptography.hazmat.primitives.asymmetric",
         "cryptography.hazmat.primitives.asymmetric.ed25519"]


def install(*paths, keep=()):
    for m in HEAVY:
        if m not in keep:
            sys.modules[m] = mock.MagicMock()
    os.environ.update(
        VIDEO_MODEL="v", VLM_MODEL="q", EDIT_MODEL="Qwen/Qwen-Image-Edit-2511", RESIDENT_MODELS="1",
        BUCKET="b", QUEUE_URL="q", JOBS_TABLE="j", AWS_REGION="us-east-1", INSTANCE_ID="i", NONCES_TABLE="n",
        AUTHORIZED_KEYS="", TRAINER_INSTANCE_ID="", TRAIN_QUEUE_URL="tq", MAX_UPTIME_HOURS="8",
        TRAIN_MAX_UPTIME_HOURS="24")
    for p in paths:
        sys.path.insert(0, os.path.join(ROOT, p))


class FakeS3:
    """In-memory S3 with the calls the API uses; objects hold bytes."""

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def copy_object(self, Bucket, Key, CopySource):
        assert CopySource["Key"] in self.objects, f"copy of missing {CopySource['Key']}"
        self.objects[Key] = self.objects[CopySource["Key"]]

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)  # tests set handler.ClientError = KeyError
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, Bucket, Key, Range=None):
        import io
        data = self.objects[Key]
        if Range:
            a, b = (int(x) for x in Range.split("=")[1].split("-"))
            data = data[a:b + 1]
        return {"Body": io.BytesIO(data)}

    def generate_presigned_url(self, op, Params, ExpiresIn):
        part = f"?part={Params['PartNumber']}&upload={Params['UploadId']}" if op == "upload_part" else ""
        return "https://signed/" + Params["Key"] + part

    multipart = {}

    def create_multipart_upload(self, Bucket, Key, ContentType=None):
        upload_id = f"up{len(self.multipart) + 1}"
        self.multipart[upload_id] = {"key": Key, "state": "open"}
        return {"UploadId": upload_id}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        mp = self.multipart[UploadId]
        assert mp["key"] == Key and mp["state"] == "open"
        mp.update(state="done", parts=MultipartUpload["Parts"])
        self.objects[Key] = b"multipart:" + str(len(MultipartUpload["Parts"])).encode()
        return {}

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.multipart[UploadId]["state"] = "aborted"
        return {}

    def list_objects_v2(self, Bucket, Prefix, Delimiter="/"):
        subs = sorted({Prefix + k[len(Prefix):].split("/")[0] + "/" for k in self.objects
                       if k.startswith(Prefix) and "/" in k[len(Prefix):]})
        return {"CommonPrefixes": [{"Prefix": p} for p in subs]}

    def get_paginator(self, name):
        return mock.MagicMock(paginate=lambda Bucket, Prefix: [{"Contents": [
            {"Key": k, "Size": len(v)} for k, v in sorted(self.objects.items()) if k.startswith(Prefix)]}])

    refuse = ()  # key prefixes delete_objects answers with an error for (as S3 does without permission)

    def delete_objects(self, Bucket, Delete):
        errors = [o for o in Delete["Objects"] if self.refuse and o["Key"].startswith(tuple(self.refuse))]
        for o in Delete["Objects"]:
            if o not in errors:
                self.objects.pop(o["Key"], None)
        return {"Errors": errors} if errors else {}

    def json(self, key):
        return json.loads(self.objects[key]) if key in self.objects else None

    def put_json(self, key, value):
        self.objects[key] = json.dumps(value).encode()


def expect_error(fn, *args, contains="", exc=ValueError):
    try:
        fn(*args)
    except exc as e:
        assert contains in str(e), (args, str(e))
        return
    raise AssertionError(f"no {exc.__name__} for {args}")
