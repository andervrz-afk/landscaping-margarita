import os, time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

SYSTEM_PROMPT = """PEGA AQUÍ EL SYSTEM PROMPT"""
MODEL = os.environ["GEMINI_MODEL"]        # nombre del modelo Flash vigente
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://TU-DOMINIO.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

sessions = {}  # session_id -> {"history": [...], "last": timestamp}
MAX_MSGS, TTL = 24, 3600

class ChatIn(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(body: ChatIn):
    msg = body.message.strip()[:500]
    if not msg:
        raise HTTPException(400, "Mensaje vacío")
    now = time.time()
    for sid in [k for k, v in sessions.items() if now - v["last"] > TTL]:
        del sessions[sid]                      # borra sesiones viejas
    s = sessions.setdefault(body.session_id, {"history": [], "last": now})
    s["last"] = now
    s["history"].append(types.Content(role="user", parts=[types.Part(text=msg)]))
    s["history"] = s["history"][-MAX_MSGS:]
    try:
        r = client.models.generate_content(
            model=MODEL,
            contents=s["history"],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=300,
                temperature=0.3,
            ),
        )
    except Exception:
        s["history"].pop()
        raise HTTPException(502, "No se pudo generar respuesta")
    reply = r.text or "No pude responder eso. Escríbenos por WhatsApp."
    s["history"].append(types.Content(role="model", parts=[types.Part(text=reply)]))
    return {"reply": reply}
