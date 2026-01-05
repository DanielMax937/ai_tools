# ai_tools

## 环境与依赖安装（使用 uv）

1. 安装 uv（如果尚未安装）：
   ```bash
   pip install uv
   ```
2. 在项目根目录安装依赖：
   ```bash
   uv sync
   ```
3. 示例：运行聊天界面：
   ```bash
   uv run python chat/chat_ui.py
   ```

## 其他说明

原有的 `python -m playwright install` 等命令仍可在依赖安装完成后按需运行。

## TODO 0307
1. 整合到iread
2. 抓取的子记录存储数据库，通过modify_time判断是否需要修改，存取的url，内容，总结，页面时间
3. 添加一个承载页，用户点击后，然后跳转，后续记录点击次数
