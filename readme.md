curl http://localhost:11434/api/generate -d '{
  "model": "qwen:7b",
  "prompt": "你好，请用一句话介绍你自己",
  "stream": false
}'