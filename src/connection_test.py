#!/usr/bin/env python3

import ollama

response = ollama.chat(model='codellama', messages=[
    {'role': 'user', 'content': 'Hello, just testing connection.'}
])

print(response['message']['content'])