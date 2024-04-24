from whisper_live.client import TranscriptionClient
client = TranscriptionClient(
  "127.0.0.1",
  58142,
  lang="en",
  translate=False,
  model="small",
  use_vad=False,
)

client()