


# 高德开发者平台注册
## Apply API Key
https://console.amap.com/
## MCP Intergration
https://lbs.amap.com/api/mcp-server/gettingstarted

```
{
  "mcpServers": {
    "amap-amap-sse": {
      "url": "https://mcp.amap.com/sse?key=<api-key>"
    }
  }
}
```

# MINI Max 
https://api.minimax.chat/
https://platform.minimaxi.com/

## MCP Integration
```
{
  "mcpServers": {
    "MiniMax": {
      "command": "uvx",
      "args": [
        "minimax-mcp"
      ],
      "env": {
        "MINIMAX_API_KEY": "<api-key>",
        "MINIMAX_MCP_BASE_PATH": " C:\\Users\\DY\\Desktop\\minimax_workspace",
        "MINIMAX_API_HOST": "https://api.minimax.chat"
      }
    }
  }
}
```

# Edge One Pages
Convert HTML to public website
https://edgeone.cloud.tencent.com/pages/document/173172415568367616
## MCP Integration
```
{
  "mcpServers": {
    "edgeone-pages-mcp-server": {
      "command": "npx",
      "args": ["edgeone-pages-mcp"],
      "env": {
        "EDGEONE_PAGES_API_TOKEN": "",
        "EDGEONE_PAGES_PROJECT_NAME": ""
      }
    }
  }
}
```

# Test

```
查询大连各个景点的信息，要求保证真实有效

你是一个专业导游，筛选可供游玩的景点，在百度百科上搜索相关历史典故，名称由来，对每个景点生成一个详细介绍，方便介绍给游客，加深他们的印象，生成一个markdown文件

请把该生成的MD文件转换成语音文件 保存到本地目录

https://minimax-algeng-chat-tts.oss-cn-wulanchabu.aliyuncs.com/audio%2Ftts-20250706200656-McOxhUuL.mp3?Expires=86401751803616&OSSAccessKeyId=LTAI5tGLnRTkBjLuYPjNcKQ8&Signature=fnQbfD9wsg%2BebD3onnebAmg8zAk%3D

```