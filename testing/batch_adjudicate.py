import pandas as pd
import asyncio
import httpx
import json
import math
from typing import List

API_URL = "http://localhost:105/api/adjudicate"
BATCH_SIZE = 45  
TIMEOUT = 50.0 

### ALWAYS CHECK BEFORE RUNNING ###
with open("../PROMPT.txt", "r") as f:
   ADJ_PROMPT = f.read()

# Load SOC code descriptions for lookup
soc_df = pd.read_csv("SOC_DESCRIPTION.csv")
soc_lookup = {row['code']: row['title'] for _, row in soc_df.iterrows()}

df = pd.read_csv("./INPUTFILE.csv")

JOB_INPUT = [row['id'] for _, row in df.iterrows()]

true_map = {
   row["id"]: {
       **{k: v for k, v in row.items() if k.startswith("coder_")}, 
       "user_soc": row.get("user_soc"), 
       "office_soc": row.get("office_soc"),  
       "init_ans": row["job_text"]
   }
   for _, row in df.iterrows()
}

INIT_Q = "What was your main job over the last seven days? Please enter your job title and a description of your firm or industry if applicable."
### ALWAYS CHECK BEFORE RUNNING ###

async def classify(client: httpx.AsyncClient, job_id: str):
   metadata = true_map.get(job_id)
   job_text = metadata["init_ans"]

   coders = []
   coder_keys = [k for k in metadata.keys() if k.startswith("coder_") and metadata.get(k)]
   
   for coder_key in coder_keys:
       coder_soc = metadata[coder_key]
       coders.append({
           "coder_id": coder_key,
           "classification": coder_soc,
           "description": soc_lookup.get(coder_soc, f"Unknown SOC {coder_soc}")
       })
   
   if len(coders) < 2:
       print(f"⚠️ Skipping {job_text[:50]}... - insufficient coders ({len(coders)})")
       return {"job_title": job_text, "error": "Insufficient coders"}
       
   # Base payload without user_soc
   payload = {
       "adj_prompt": ADJ_PROMPT,
       "init_q": INIT_Q,
       "init_ans": job_text,
       "coders": coders,
       "include_self_classification": False,
       "model": "o4-mini-2025-04-16"
   }

   # Add user_soc fields if user_soc exists and is not null
   user_soc = metadata.get("user_soc")
   if user_soc is not None and pd.notna(user_soc):
       payload["user_soc"] = user_soc
       payload["user_soc_desc"] = soc_lookup.get(user_soc, f"Unknown SOC {user_soc}")
       payload["include_self_classification"] = True

   try:
       response = await client.post(API_URL, json=payload, timeout=TIMEOUT)
       response.raise_for_status()
       result = response.json()
       print(f"{job_text} → {result.get('adj_soc_code', 'N/A')}")
       return {
           "id": job_id,
           "job_title": job_text,
           "user_soc": metadata.get("user_soc"),
           "office_soc": metadata.get("office_soc"),
           "adj_soc_code": result.get("adj_soc_code"),
           **{k: v for k, v in metadata.items() if k.startswith("coder_")},  # All coder columns
           "num_coders": len(coders),
           "response": result
       }
   except Exception as e:
       print(f"❌ {job_text} failed: {e}")
       return {"job_title": job_text, "error": str(e)}

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

       await asyncio.sleep(10)

### ALWAYS CHECK BEFORE RUNNING ###
   with open("../test_results_new/OUTPUTFILE.json", "w") as f:
       json.dump(all_results, f, indent=2)

   print(f"\n Finished. {len(all_results)} job titles processed and saved to test_results_new/OUTPUTFILE.json")
### ALWAYS CHECK BEFORE RUNNING ###

if __name__ == "__main__":
   asyncio.run(main())