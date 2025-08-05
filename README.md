# 🤖 AI Image Processing Application

## Architecture Overview

```
User's Device (Client)    Render Server (Your Backend)    Hugging Face
       📱                          🖥️                        ☁️
       |                           |                         |
       | Uploads image              |                         |
       |-------------------------->|                         |
       |                           | Downloads model         |
       |                           |<------------------------|
       |                           | Loads model in RAM      |
       |                           | Processes image         |
       | Gets results              | Sends prediction        |
       |<--------------------------|                         |
```
