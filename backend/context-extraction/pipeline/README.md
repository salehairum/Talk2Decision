# Running Files
## Loader.py
**loader.py** — reads your Slack JSON, skips channel_join and other noise subtypes, resolves real names from user_profile, converts timestamps to ISO format. Run it as python loader.py your_export.json my-decision.
```bash
python context-extraction/pipeline/loader.py context-extraction/data/chat.json chat
```
output:
```bash
[loader] Loaded 2 messages from 'context-extraction/data/chat.json'
[loader] First message:
{
  "message_id": "125e1387-dffb-4090-bfb4-3c3a6b6c6caa",
  "thread_id": "chat",
  "role": "user",
  "author_id": "U0AMZS4QFPS",
  "author_name": "Saleha Irum",
  "content": "acha these channels were created by default",
  "timestamp": "2026-03-17T16:47:42",
  "ts_raw": 1773766062.735799
}
[loader] Last message:
{
  "message_id": "9f2481ff-ad10-4bb3-a431-90edd79fead9",
  "thread_id": "chat",
  "role": "user",
  "author_id": "U0AMZS4QFPS",
  "author_name": "Saleha Irum",
  "content": "adding ik do msg just to see how the export works",
  "timestamp": "2026-03-17T16:47:51",
  "ts_raw": 1773766071.301369
}
```

## Preprocess.py

**preprocess.py** — strips <@mentions>, URLs, Slack formatting markers, emojis, drops boilerplate one-word replies like "ok" / "thanks". Adds a content_clean field to each message while preserving the original content for debugging.
```bash
PS C:\Users\Noor\Desktop\Talk2Decision> python context-extraction/pipeline/preprocess.py context-extraction/data/chat.json chat
```
output 
```bashx
[preprocess] 2 in → 2 kept, 0 boilerplate dropped

Sample cleaned messages (2):
{
  "author_name": "Saleha Irum",
  "timestamp": "2026-03-17T16:47:42",
  "content": "acha these channels were created by default",
  "content_clean": "acha these channels were created by default"
}
{
  "author_name": "Saleha Irum",
  "timestamp": "2026-03-17T16:47:51",
  "content": "adding ik do msg just to see how the export works",
  "content_clean": "adding ik do msg just to see how the export works"
}
```

## filter_for_search.py
```
python context-extraction/pipeline/filter_for_search.py context-extraction/data/chat.json
```

## Main.py
python context-extraction/pipeline/main.py context-extraction/data/chat.json