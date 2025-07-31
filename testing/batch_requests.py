import pandas as pd
import asyncio
import httpx
import json
import math
import time
from typing import List

API_URL = "http://localhost:105/api/classify"
BATCH_SIZE = 30  
TIMEOUT = 50.0 

### ALWAYS CHECK BEFORE RUNNING ###
with open("../classify_prompt.txt", "r") as f:
    SYS_PROMPT = f.read()

df = pd.read_csv("./input_time_100.csv")

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
    start_time = time.time()
    print(f"Starting batch processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    total_batches = math.ceil(len(JOB_INPUT) / BATCH_SIZE)

    for i in range(total_batches):
        batch_start = time.time()
        batch = JOB_INPUT[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        print(f"\nProcessing batch {i + 1}/{total_batches} ({len(batch)} items)")
        batch_results = await process_batch(batch)
        all_results.extend(batch_results)
        
        batch_time = time.time() - batch_start
        print(f"Batch {i + 1} completed in {batch_time:.2f} seconds")

        await asyncio.sleep(8)

### ALWAYS CHECK BEFORE RUNNING ###
    with open("../test_results_new/publicvoice_timed_o4-mini.json", "w") as f:
        json.dump(all_results, f, indent=2)

    end_time = time.time()
    total_time = end_time - start_time
    
    # Save timing information
    timing_info = {
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        "end_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
        "total_seconds": total_time,
        "total_minutes": total_time/60,
        "total_jobs": len(all_results),
        "jobs_per_second": len(all_results)/total_time,
        "batch_size": BATCH_SIZE,
        "total_batches": total_batches
    }

    with open("../test_results_new/publicvoice_timed_o4-mini.json", "w") as f:
        json.dump(timing_info, f, indent=2)

    print(f"\n✅ Finished. {len(all_results)} job titles processed and saved to test_results_new/publicvoice_timed_o4-mini.json")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📊 Processing rate: {len(all_results)/total_time:.2f} jobs/second")
    print(f"📁 Timing details saved to test_results_new/publicvoice_timed_o4-mini.json")
### ALWAYS CHECK BEFORE RUNNING ###

if __name__ == "__main__":
    asyncio.run(main())