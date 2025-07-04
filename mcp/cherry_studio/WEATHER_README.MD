# Cherry Stuio Integration

## 配置模型
设置 -> 模型服务 -> DeepSeek -> 配置API_KEY

## 启动MCP
Refer to 
server/
server/sse
## 添加MCP
设置 -> MCP服务器 -> 添加服务器 -> 从JSON导入
```
{
    "mcpServers": {
        "weather_sse": {
            "type": "sse",
            "url": "http://localhost:8000/sse"
        }
    }
}
```
```
{
    "mcpServers": {
        "weather_stdio": {
            "command": "uv",
            "args": [
                "--directory", 
                "G:\\\\ai_learning\\\\mcp\\\\server",
                "run",
                "weather.py"
            ]
        }
    }
}

```
