# Examples

| file | what it is |
|---|---|
| `request.json` | request body shape (base64 pcap truncated for readability) |
| `response.json` | **real** response from the 306M checkpoint on `crafted_attack_trace.pcap` |
| `client.py` | working client; reads your HF token from `huggingface-cli login` |

## Local (no endpoint needed)

```bash
huggingface-cli download Anvyon/protogrok-jax-306m --local-dir ckpt
python handler.py ckpt capture.pcap
```

## Against a deployed endpoint

```bash
python examples/client.py https://YOUR-ENDPOINT.endpoints.huggingface.cloud capture.pcap
```

## curl

```bash
base64 -i capture.pcap | tr -d '\n' > /tmp/b64
printf '{"inputs":"%s","parameters":{"threshold":0.5}}' "$(cat /tmp/b64)" > /tmp/req.json
curl -X POST "$ENDPOINT_URL" \
     -H "Authorization: Bearer $HF_TOKEN" \
     -H "Content-Type: application/json" \
     --data @/tmp/req.json
```

## Parameters

| name | default | meaning |
|---|---|---|
| `threshold` | 0.5 | cutoff for the `anomaly` boolean |
| `diagnose` | true | include findings / root cause / remediation |
| `max_flows` | 200 | cap on returned flow reports |

A cold endpoint (scale-to-zero) returns **HTTP 503** on the first call for
~30-60s while it boots. Retry once.
