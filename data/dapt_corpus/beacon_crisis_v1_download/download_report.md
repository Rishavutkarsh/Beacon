# Beacon Crisis v1 Download Attempt

I prepared a bounded source set focused on official and reputable disaster/crisis guidance:
FEMA/Ready, NDMA/NIDM India, WHO, UNICEF, IFRC, and Sphere.

The download pass did not complete because outbound network fetches were denied by sandbox review.

## What was blocked

- PowerShell + Python bounded downloader/extractor pipeline
- `Invoke-WebRequest https://www.example.com` connectivity check

## What remains

- No raw PDFs, ZIPs, or extracted text were written.
- No train/dev DAPT JSONL rows were produced.
- No token growth toward the ~2M target could be measured from fresh downloads.

## Files written in this directory

- `source_list.jsonl`
- `blocked_document_cards.jsonl`
- `manifest.json`
- `download_report.md`

## Curated source families

- FEMA / Ready course materials and preparedness guides
- NDMA and MHA disaster-management guidance for India
- NIDM awareness and safety materials, including Hindi-facing pages
- WHO WASH and outbreak-response handbooks
- UNICEF Pacific WASH emergencies handbook
- IFRC disaster risk reduction messaging guide
- Sphere Handbook

## Rejection class

- `network_fetch_blocked`
