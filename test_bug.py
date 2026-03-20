import asyncio
import httpx
from models import VerifyRequest
from services.gemini import fact_check
import logging

logging.basicConfig(level=logging.DEBUG)

async def run():
    async with httpx.AsyncClient() as http_client:
        req = VerifyRequest(text="URGENT FREE LPG")
        print("Running fact_check...")
        try:
            res = await fact_check(req, http_client)
            print("Response:", res)
        except Exception as e:
            print("Exception:", e)

asyncio.run(run())
