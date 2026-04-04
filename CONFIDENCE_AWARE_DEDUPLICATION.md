# ✅ Confidence-Aware Decision Deduplication

## Overview

The decision tracking system now intelligently manages decisions across multiple files with confidence-aware deduplication. This ensures high-quality decisions are not overwritten by lower-confidence extractions.

## Key Features

### 1. **Confidence Protection**

Low-confidence extractions will **NOT** replace existing high-confidence decisions.

**Confidence Hierarchy**: `High` > `Medium` > `Low`

```
Scenario: Same query extracted multiple times with different confidence levels

Day 1 (File 1): Query "What is the decision?" → High confidence
  ✓ Decision created with HIGH confidence: "The team goes with blue theme"

Day 2 (File 2): Same query → Medium confidence  
  ✓ Decision updated (Medium > Low, but < High)
  ✓ Decision text may change if quality is comparable
  
Day 3 (File 3): Same query → Low confidence
  ✗ Decision NOT updated (Low < High)
  ✓ Source file STILL tracked
  ✓ Status moved to "In-Progress"
```

### 2. **Multi-Topic Support**

The system correctly handles tracking multiple **different** decisions simultaneously.

```
User asks multiple topics:

1. "What is the decision?" 
   → Decision #1 created (query-specific)
   → Only matches identical/similar queries

2. "What are the action items?"
   → Decision #2 created (different topic)
   → Tracked independently from Decision #1

3. "What was decided?"
   → May match Decision #1 (similar but not exact)
   → Or create Decision #3 (if matching logic differs)

Result: Each unique topic = separate decision entry
```

### 3. **Source File Tracking**

All files contributing to a decision are tracked in `source_files` array.

```json
{
  "decision_id": 1,
  "query": "What is the decision?",
  "decision": "The team goes with blue theme",
  "confidence": "High",
  "source_files": ["2026-03-23.json", "2026-03-26.json", "2026-04-03.json"],
  "status": "In-Progress"
}
```

**Behavior**:
- New source file always added to array
- Added even if decision text not updated (low confidence)
- Maintains chronological order of updates

### 4. **Selective Evidence & Action Items Update**

Evidence and action items are only updated when confidence improves.

```
Decision #1 (High confidence)
├─ Evidence: ["msg1", "msg2", "msg3"]  (from Day 1)
└─ Actions: ["Task A", "Task B"]

Day 2 Update (Medium confidence)
├─ New Evidence: ["msg4", "msg5"]  
│  ✓ Added to existing evidence
├─ New Actions: ["Task C"]
│  ✓ Added to existing actions
└─ Result: Merged set ["msg1", "msg2", "msg3", "msg4", "msg5"]

Day 3 Update (Low confidence - NO UPDATE)
├─ New Evidence: ["msg6"] 
│  ✗ Not added (confidence too low)
├─ New Actions: ["Task D"]
│  ✗ Not added (confidence too low)  
└─ Result: Evidence/Actions unchanged
```

### 5. **Decision History Tracking**

Complete audit trail of all changes with timestamps.

```
Decision History for #1:

Entry 1: SOURCE_FILE
  From: "2026-03-23"
  To:   "2026-03-26"
  Time: 2026-04-04 10:38:31

Entry 2: DECISION
  From: "light theme with blue accents"
  To:   "The team goes with blue theme"
  Time: 2026-04-04 10:38:31

Entry 3: CONFIDENCE
  From: "Medium"
  To:   "High"
  Time: 2026-04-04 10:39:15
```

## Implementation Details

### Decision Matching Logic

```python
# Query normalization for consistent matching
normalized_query = query.strip().lower()

# Lookup existing decision by normalized query
SELECT id FROM decisions 
WHERE LOWER(TRIM(query)) = LOWER(:normalized_query)

# If found, check confidence
if new_confidence >= existing_confidence:
    # Update decision text, evidence, and items
else:
    # Keep existing decision, just track source file
```

### Confidence Ranking

```python
confidence_rank = {
    "High": 3,
    "Medium": 2, 
    "Low": 1
}

should_update = new_rank >= old_rank
```

### Behavior Matrix

| Existing | New | Update Decision? | Update Evidence? | Track Source File? |
|----------|-----|------------------|------------------|-------------------|
| High | High | ✓ Yes | ✓ Yes | ✓ Yes |
| High | Medium | ✓ Yes | ✓ Yes | ✓ Yes |
| High | Low | ✗ No | ✗ No | ✓ Yes |
| Medium | High | ✓ Yes | ✓ Yes | ✓ Yes |
| Medium | Medium | ✓ Yes | ✓ Yes | ✓ Yes |
| Medium | Low | ✗ No | ✗ No | ✓ Yes |
| Low | High | ✓ Yes | ✓ Yes | ✓ Yes |
| Low | Low | ✓ Yes | ✓ Yes | ✓ Yes |

## Testing

### Test 1: Confidence Protection
```bash
python test_confidence_aware.py
```

Verifies:
- ✅ Same query matches across files
- ✅ Decision text protected from low-confidence overwrites
- ✅ Source files tracked even when text not updated
- ✅ Status properly managed

### Test 2: Multiple Decision Tracking
```bash
python test_diagnostics.py
```

Verifies:
- ✅ Different queries create separate decisions
- ✅ Each decision tracked independently
- ✅ History correctly recorded per decision
- ✅ Confidence levels independently managed

## Example Workflow

```
USER UPLOADS 3 SLACK EXPORTS (Different Days)

Day 1: 2026-03-23.json
├─ Query: "What is the decision?"
├─ LLM Extraction: confidence=High, decision="Blue theme"
├─ Database: ✓ CREATE Decision #1
└─ source_files: ["2026-03-23"]

Day 2: 2026-03-26.json  
├─ Query: "What is the decision?"
├─ LLM Extraction: confidence=Medium, decision="Blue with accents"
├─ Database: ✓ UPDATE Decision #1 (Medium >= Low baseline)
└─ source_files: ["2026-03-23", "2026-03-26"]

Day 3: 2026-04-03.json
├─ Query: "What is the decision?"
├─ LLM Extraction: confidence=Low, decision="Not found"
├─ Database: ✗ NO UPDATE (Low < High, keep blue theme)
│             ✓ ADD source file  
└─ source_files: ["2026-03-23", "2026-03-26", "2026-04-03"]

FINAL RESULT:
- Decision #1: "Blue theme" (highest confidence version wins!)
- Source Files: All 3 days tracked
- History: Shows all updates and improvements
- Evidence: From the best extractions
- Action Items: Consolidated from high-confidence runs
```

## Benefits

1. **Quality Assurance**: High-confidence decisions are protected
2. **Complete Tracking**: Source files always tracked, regardless of confidence
3. **Audit Trail**: Full history of all changes and reasoning
4. **Multi-Topic Support**: Different topics tracked independently
5. **Smart Merging**: Information only integrated when quality improves
6. **Status Management**: Clear progression from Open → In-Progress → Resolved

## Configuration

No configuration needed. The system uses:
- Confidence levels from LLM extractions (High/Medium/Low)
- Query text for topic matching (case-insensitive, normalized)
- File IDs for source tracking
- Timestamps for audit trail

All logic is built into the `/query` endpoint in `backend/api/app.py`.
