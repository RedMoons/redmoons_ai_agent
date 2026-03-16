import os, time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

app = FastAPI()

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options={'api_version': 'v1beta'}
)

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, lang: str = "en", audio: str = "false"):
    await websocket.accept()

    lang_map = {
        "ko": "Korean", "en": "English",
        "ja": "Japanese", "zh": "Chinese", "es": "Spanish",
    }
    lang_name = lang_map.get(lang, "English")
    enable_audio = audio.lower() == "true"

    BATCH_INTERVAL = 15.0
    audio_queue = asyncio.Queue()  # ✅ audio chunk queue

    async def collect_audio():
        """browser audio data append on queue"""
        try:
            while True:
                data = await websocket.receive_bytes()
                if data and len(data) > 0:
                    await audio_queue.put(data)
        except WebSocketDisconnect:
            print("end client connection")
        except Exception as e:
            if "1005" not in str(e):
                print(f"❌ collect error: {e}")

    async def process_batch():
        """every 15s queue process"""
        while True:
            await asyncio.sleep(BATCH_INTERVAL)

            # ✅ pop all chunk from queue
            chunks = []
            while not audio_queue.empty():
                chunks.append(await audio_queue.get())

            if not chunks:
                print("⏭️ skip since no audio")
                continue

            print(f"📦 start processing {len(chunks)} fo chunk")
            await websocket.send_text("[STATUS] under analysis...")

            try:
                # ✅ everytime new session to collect transcript
                transcript = await get_transcript(chunks)

                if not transcript.strip():
                    print("⏭️ no transcript, skip")
                    continue

                print(f"🎤 transcript: {transcript[:80]}...")

                # ✅ feedback from Gemini API
                feedback_text, audio_b64 = await get_feedback(transcript, lang_name, enable_audio)
                print(f"💡 Feedback: {feedback_text[:80]}...")
                if audio_b64:
                    await websocket.send_text(f"[AUDIO] {audio_b64}")
                await websocket.send_text(f"[FEEDBACK] {feedback_text}")

            except Exception as e:
                print(f"❌ process error: {e}")

    async def get_transcript(chunks: list) -> str:
        """new Live session audio → transcript convert"""
        transcript_parts = []

        try:
            async with client.aio.live.connect(
                model="models/gemini-2.5-flash-native-audio-preview-12-2025",
                config=types.LiveConnectConfig(
                    response_modalities=[types.Modality.AUDIO],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Puck"
                            )
                        )
                    ),
                    input_audio_transcription=types.AudioTranscriptionConfig(),
                )
            ) as session:

                # send audio
                for chunk in chunks:
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
                print(f"✅ {len(chunks)} num of chunk finished")

                # collect transcript (end when turn_complete)
                try:
                    async with asyncio.timeout(10):
                        async for message in session.receive():
                            if message.server_content:
                                sc = message.server_content
                                if sc.input_transcription and sc.input_transcription.text:
                                    transcript_parts.append(sc.input_transcription.text.strip())
                                if sc.turn_complete:
                                    break  # ✅ turn_complete → end session
                except asyncio.TimeoutError:
                    print(f"⏱️ timeout → {len(transcript_parts)} of num")

        except Exception as e:
            print(f"❌ transcript error: {e}")

        return " ".join(transcript_parts)

    async def get_feedback(transcript: str, lang_name: str, enable_audio: bool) -> str:
        """feedback creation by Gemini API"""
        prompt = f"""
You are a professional speech coach for workplace meetings.
Analyze this speech and provide feedback in {lang_name}:

"{transcript}"

Provide:
1. 📋 Summary
2. 💬 Communication Feedback
3. ✨ Improvement Tips
4. 👍 What was done well

Keep it short (total 30 second reading volume) and constructiveas. Do NOT add any preamble.
"""
        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )
        feedback_text = response.text

        audio_b64 = None
        if enable_audio:
            try:
                tts_response = await asyncio.to_thread(
                    lambda: client.models.generate_content(
                        model="gemini-2.5-flash-preview-tts",
                        contents=feedback_text,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name="Puck"
                                    )
                                )
                            ),
                        )
                    )
                )
                import base64
                audio_data = tts_response.candidates[0].content.parts[0].inline_data.data
                audio_b64 = base64.b64encode(audio_data).decode()
                print("✅ completed TTS creation")
            except Exception as e:
                print(f"❌ TTS error: {e}")

        return feedback_text, audio_b64

    await asyncio.gather(
        collect_audio(),
        process_batch(),
    )

    try:
        await websocket.close()
    except RuntimeError:
        pass
    print("🔌 end session")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)