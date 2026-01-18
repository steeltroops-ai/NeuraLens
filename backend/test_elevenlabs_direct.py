from elevenlabs.client import ElevenLabs
import os

# Use the key from your .env
API_KEY = "sk_51e9eeb8245869ede7df1a32feed89c942d21a57cffeab43"

print(f"Testing with API Key: {API_KEY[:5]}...{API_KEY[-5:]}")

try:
    client = ElevenLabs(api_key=API_KEY)

    print("Sending request to ElevenLabs...")
    audio_generator = client.text_to_speech.convert(
        text="The first move is what sets everything in motion.",
        voice_id="21m00Tcm4TlvDq8ikWAM", # Rachel
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    # Consume the generator to get bytes
    audio_bytes = b"".join(audio_generator)
    
    print(f"Success! Received {len(audio_bytes)} bytes of audio.")
    
    with open("test_output.mp3", "wb") as f:
        f.write(audio_bytes)
    print("Saved to test_output.mp3")

except Exception as e:
    print("\nXXX ERROR OCCURRED XXX")
    print(e)
