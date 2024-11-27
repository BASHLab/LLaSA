from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama2-70b",
  messages=[{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

