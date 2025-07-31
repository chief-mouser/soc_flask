import pandas as pd
import asyncio
import httpx
import json
import math
from typing import List

API_URL = "http://localhost:105/api/classify"
BATCH_SIZE = 45  
TIMEOUT = 50.0 

### ALWAYS CHECK BEFORE RUNNING ###
with open("../PROMPT.txt", "r") as f:
    SYS_PROMPT = f.read()

df = pd.read_csv("./INPUTFILE.csv")

JOB_INPUT = [row['id'] for _, row in df.iterrows()]

true_map = {
    row["id"]: {"office_soc": row["office_soc"], 
                "init_ans": row["job_text"]}
    for _, row in df.iterrows()
}

INIT_Q = "What was your main job over the last seven days? Please enter your job title and a description of your firm or industry if applicable."
### ALWAYS CHECK BEFORE RUNNING ###

async def classify(client: httpx.AsyncClient, job_id: str):
    metadata = true_map.get(job_id)
    job_text = metadata["init_ans"]

    payload = {
        "sys_prompt": SYS_PROMPT,
        "init_q": INIT_Q,
        "init_ans": job_text,
        "k": 30,
        "index": "soc4d",
        "model": "o4-mini-2025-04-16"
    }

    try:
        response = await client.post(API_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        print(f"{job_text} → {result.get('soc_code', 'N/A')}")
        return {
            "id": job_id,
            "job_title": job_text,
            "soc_code": result.get("soc_code"),
            "office_soc": metadata.get("office_soc"),
            "response": result
        }
    except Exception as e:
        print(f"❌ {job_text} failed: {e}")
        return {"id": job_id, "job_title": job_text, "error": str(e)}

async def process_batch(batch: List[str]):
    async with httpx.AsyncClient() as client:
        tasks = [classify(client, job_id) for job_id in batch]
        return await asyncio.gather(*tasks)

async def main():
    all_results = []
    total_batches = math.ceil(len(JOB_INPUT) / BATCH_SIZE)

    for i in range(total_batches):
        batch = JOB_INPUT[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        print(f"\nProcessing batch {i + 1}/{total_batches} ({len(batch)} items)")
        batch_results = await process_batch(batch)
        all_results.extend(batch_results)

        await asyncio.sleep(5)

### ALWAYS CHECK BEFORE RUNNING ###
    with open("./test_results/OUTPUTFILE.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{len(all_results)} job titles processed and saved to test_results/OUTPUTFILE.json")
### ALWAYS CHECK BEFORE RUNNING ###

if __name__ == "__main__":
    asyncio.run(main())