"""Runs the AI-look critic (Qwen3-VL-32B) in the Krea venv, which has a transformers with Qwen3-VL (the worker's
venv is older). One process = one model load, then every step of a request in order.

    python vlm_judge.py request.json
    request: {"model_dir", "result", "steps": [{"images": [paths], "text", "max_new_tokens", "sample", "seed"}]}
             A step's text may contain {step0}, {step1}, ...: the answers of earlier steps.
    result:  {"answers": [str, ...], "load_seconds", "seconds"} or {"error"}
docs/specs/2026-11-28-realism-retry.md (critic upgrade)
"""
import json
import re
import sys
import time


def fill(text, answers):
    """{stepN} -> the answer of step N (an unknown index stays as written)."""
    return re.sub(r"\{step(\d+)\}", lambda m: answers[int(m.group(1))] if int(m.group(1)) < len(answers) else m.group(0),
                  text)


def main(request_path):
    with open(request_path) as f:
        req = json.load(f)
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        t0 = time.time()
        model = Qwen3VLForConditionalGeneration.from_pretrained(req["model_dir"], dtype=torch.bfloat16,
                                                                device_map="cuda")
        processor = AutoProcessor.from_pretrained(req["model_dir"])
        load = round(time.time() - t0, 1)
        answers, t1 = [], time.time()
        for step in req["steps"]:
            content = [{"type": "image", "image": p} for p in step.get("images") or []]
            content.append({"type": "text", "text": fill(step["text"], answers)})
            inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                   add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            kw = {"max_new_tokens": int(step.get("max_new_tokens") or 1024)}
            if step.get("sample"):
                torch.manual_seed(int(step.get("seed") or 0))
                kw.update(do_sample=True, temperature=0.8, top_p=0.9)
            else:
                kw.update(do_sample=False)
            out = model.generate(**inputs, **kw)
            out = out[:, inputs["input_ids"].shape[1]:]
            answers.append(processor.batch_decode(out, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0])
            print(json.dumps({"step": len(answers)}), flush=True)
        result = {"answers": answers, "load_seconds": load, "seconds": round(time.time() - t1, 1)}
    except Exception as e:  # noqa: BLE001 - the worker reads the reason from the result file
        result = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
    with open(req["result"], "w") as f:
        json.dump(result, f)
    sys.exit(0 if "answers" in result else 1)


if __name__ == "__main__":
    main(sys.argv[1])
