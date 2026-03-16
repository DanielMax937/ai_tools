# NVIDIA Chat Script

使用 OpenAI SDK 调用 NVIDIA 免费模型的交互式聊天脚本。

## 安装

```bash
npm install
```

## 使用方法

### 1. 列出可用模型

```bash
node nvidia-chat.js list
# 或
node nvidia-chat.js --list
# 或
node nvidia-chat.js -l
```

### 2. 交互式聊天

```bash
node nvidia-chat.js
```

执行后：
1. 首先会显示所有可用的模型列表
2. 选择一个模型（输入编号或完整模型 ID）
3. 进入聊天模式，输入消息与模型对话
4. 输入 `exit` 或 `quit` 退出

## 示例

```bash
# 列出模型
$ node nvidia-chat.js list

# 开始聊天
$ node nvidia-chat.js
🚀 NVIDIA AI Model Chat

🔍 Fetching available models...

Found 10 models:

1. nvidia/llama-3.1-nemotron-70b-instruct
2. meta/llama-3.1-8b-instruct
...

Select a model number (or enter model ID): 1

💬 Starting chat with model: nvidia/llama-3.1-nemotron-70b-instruct
Type "exit" or "quit" to end the conversation.

You: Hello!
Assistant: Hi! How can I help you today?

You: exit
👋 Goodbye!
```

## API Key

## 特性

- ✅ 列出所有可用的 NVIDIA 模型
- ✅ 交互式聊天界面
- ✅ 流式响应（实时显示）
- ✅ 会话历史记录
- ✅ 支持多轮对话
- ✅ 错误处理和重试

# NVIDIA AI Chat - 交互式聊天工具

一个基于 OpenAI SDK 的命令行工具，用于与 NVIDIA 免费 AI 模型进行交互式对话。

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [API 限制说明](#api-限制说明)
- [可用模型](#可用模型)
- [常见问题](#常见问题)
- [故障排除](#故障排除)

## ✨ 功能特性

- 🤖 **188+ 免费 AI 模型** - 包括 Meta Llama、DeepSeek、Gemma、Qwen 等
- 💬 **交互式聊天** - 支持多轮对话，自动维护上下文
- 🌊 **流式响应** - 实时显示模型回复，无需等待
- 📋 **模型浏览** - 快速查看所有可用模型列表
- 🎯 **简单易用** - 单一脚本，无需复杂配置
- 🔒 **免费使用** - NVIDIA Developer Program 免费访问

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Users/daniel/Desktop/git/ai_tools
npm install
```

### 2. 运行脚本

```bash
# 列出所有可用模型
node nvidia-chat.js list

# 开始交互式聊天
node nvidia-chat.js
```

## 📖 使用方法

### 列出可用模型

```bash
node nvidia-chat.js list
# 或
node nvidia-chat.js --list
# 或
node nvidia-chat.js -l
```

**输出示例：**
```
🔍 Fetching available models...

Found 188 models:

1. 01-ai/yi-large
2. deepseek-ai/deepseek-v3.2
3. meta/llama-3.3-70b-instruct
4. nvidia/llama-3.1-nemotron-70b-instruct
...
```

### 交互式聊天

```bash
node nvidia-chat.js
```

**使用流程：**

1. **选择模型** - 输入模型编号或完整 ID
   ```
   Select a model number (or enter model ID): 111
   ```

2. **开始对话** - 输入你的问题
   ```
   You: What is machine learning?
   Assistant: Machine learning is a subset of artificial intelligence...
   ```

3. **继续对话** - 保持上下文继续提问
   ```
   You: Give me an example
   Assistant: Sure! A common example is email spam filtering...
   ```

4. **退出** - 输入 `exit` 或 `quit`
   ```
   You: exit
   👋 Goodbye!
   ```

## ⚠️ API 限制说明

### 免费层级限制

NVIDIA Developer Program 提供的免费 API 访问有以下限制：

| 限制类型 | 数值 | 说明 |
|---------|------|------|
| **请求速率** | ~40 RPM | 每分钟约 40 个请求 (Requests Per Minute) |
| **并发连接** | 有限制 | 建议单线程顺序调用 |
| **使用场景** | 原型开发 | 仅限开发、测试、研究用途 |
| **GPU 数量** | 最多 16 个 | 自托管部署限制 |
| **商业使用** | ❌ 不允许 | 生产环境需要 AI Enterprise 许可证 |

### 速率限制详情

#### 40 RPM (每分钟 40 个请求)
- **平均间隔**: 每 1.5 秒一个请求
- **突发请求**: 可能触发限流
- **建议策略**: 
  - 实现请求间隔控制
  - 添加重试机制
  - 避免并发大量请求

#### 错误码

当超过速率限制时，API 会返回：

```json
{
  "error": {
    "message": "Rate limit exceeded",
    "type": "rate_limit_error",
    "code": "rate_limit_exceeded"
  }
}
```

HTTP 状态码: `429 Too Many Requests`

### 使用场景限制

#### ✅ 允许的用途

- 🧪 原型开发和测试
- 🔬 研究和实验
- 📚 学习和教育
- 🎨 个人项目开发
- 💡 概念验证 (POC)

#### ❌ 不允许的用途

- 🏢 生产环境部署
- 💰 商业服务
- 👥 面向真实用户的应用
- 📈 大规模应用
- 🔄 持续的业务操作

> **生产环境需求**: 如需用于生产环境，需要购买 [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/) 许可证（$4,500/GPU/年 或 ~$1/GPU/小时）

### 最佳实践

#### 1. 实现请求间隔

```javascript
// 在连续请求间添加延迟
async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// 每次请求后等待
await makeApiCall();
await sleep(2000); // 等待 2 秒
```

#### 2. 错误重试

```javascript
async function apiCallWithRetry(fn, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (error.code === 'rate_limit_exceeded' && i < maxRetries - 1) {
        await sleep(60000); // 等待 60 秒后重试
        continue;
      }
      throw error;
    }
  }
}
```

#### 3. 监控使用量

建议记录：
- 每分钟请求数
- 错误率
- 响应时间

## 🎯 可用模型

### 推荐模型

| 模型 | 适用场景 | 参数量 |
|------|---------|--------|
| `nvidia/llama-3.1-nemotron-70b-instruct` | 通用对话、编程 | 70B |
| `deepseek-ai/deepseek-v3.2` | 代码生成、推理 | Large |
| `meta/llama-3.3-70b-instruct` | 通用任务 | 70B |
| `qwen/qwen3.5-397b-a17b` | 高性能对话 | 397B |
| `mistralai/mistral-large-3-675b-instruct-2512` | 复杂推理 | 675B |
| `microsoft/phi-4-mini-instruct` | 快速响应 | Mini |

### 按类别分类

#### 代码生成
- `deepseek-ai/deepseek-coder-6.7b-instruct`
- `mistralai/codestral-22b-instruct-v0.1`
- `qwen/qwen3-coder-480b-a35b-instruct`

#### 推理与思考
- `deepseek-ai/deepseek-r1-distill-qwen-32b`
- `qwen/qwen3-next-80b-a3b-thinking`
- `moonshotai/kimi-k2-thinking`

#### 多语言支持
- `qwen/qwen2.5-7b-instruct` (中文优化)
- `institute-of-science-tokyo/llama-3.1-swallow-70b-instruct-v0.1` (日语)
- `yentinglin/llama-3-taiwan-70b-instruct` (繁体中文)

#### 视觉理解
- `meta/llama-3.2-90b-vision-instruct`
- `microsoft/phi-4-multimodal-instruct`

### 查看完整列表

```bash
node nvidia-chat.js list
```

## 🔧 配置说明

### API Key

```javascript
const BASE_URL = 'https://build.nvidia.com/settings/api-keys';
```

**安全建议**（可选）：
- 使用环境变量存储 API Key
- 不要将 API Key 提交到公开仓库

### Base URL

```javascript
const BASE_URL = 'https://integrate.api.nvidia.com/v1';
```

这是 NVIDIA API Catalog 的标准端点，兼容 OpenAI API 格式。

## ❓ 常见问题

### Q1: 如何获取 NVIDIA API Key？

**A:** 
1. 访问 [build.nvidia.com](https://build.nvidia.com)
2. 注册 NVIDIA Developer Program（免费）
3. 在 API Catalog 中选择任意模型
4. 点击 "Get API Key" 生成密钥

### Q2: 为什么收到 "Rate limit exceeded" 错误？

**A:** 你已超过 40 RPM 的速率限制。建议：
- 降低请求频率
- 在请求间添加延迟（至少 1.5 秒）
- 等待 1 分钟后重试

### Q3: 可以用于商业项目吗？

**A:** 免费层级**不能**用于生产环境或商业项目。如需商业使用：
- 购买 NVIDIA AI Enterprise 许可证
- 或使用自托管的 NIM 微服务

### Q4: 哪些模型最适合编程任务？

**A:** 推荐：
- `deepseek-ai/deepseek-coder-6.7b-instruct`
- `mistralai/codestral-22b-instruct-v0.1`
- `qwen/qwen3-coder-480b-a35b-instruct`

### Q5: 响应速度慢怎么办？

**A:** 
- 选择较小的模型（如 7B 或 8B 参数）
- 减少 `max_tokens` 参数
- 避免在高峰时段使用

### Q6: 如何切换到自托管模式？

**A:** 
1. 下载 NIM 容器镜像
2. 在本地 GPU 服务器上部署
3. 修改脚本的 `BASE_URL` 为本地地址
4. 需要 NVIDIA AI Enterprise 许可证用于生产

## 🐛 故障排除

### 错误：`chat not found`

**原因**: API Key 无效或已过期

**解决**:
1. 检查 API Key 是否正确
2. 在 build.nvidia.com 重新生成 API Key
3. 确认账号状态正常

### 错误：`429 Too Many Requests`

**原因**: 超过速率限制（40 RPM）

**解决**:
1. 等待 1 分钟后重试
2. 减少请求频率
3. 实现请求队列和速率限制

### 错误：`Model not found`

**原因**: 模型 ID 错误或模型已下线

**解决**:
1. 运行 `node nvidia-chat.js list` 查看最新模型列表
2. 使用正确的模型 ID
3. 确认模型名称拼写正确

### 连接超时

**原因**: 网络问题或服务器繁忙

**解决**:
1. 检查网络连接
2. 稍后重试
3. 考虑使用代理

### 响应不完整或中断

**原因**: 流式传输错误

**解决**:
1. 检查网络稳定性
2. 重新发送请求
3. 减小 `max_tokens` 值

## 📚 参考资源

- [NVIDIA API Catalog](https://build.nvidia.com)
- [NVIDIA NIM 文档](https://docs.api.nvidia.com/nim/)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [NVIDIA Developer Program](https://developer.nvidia.com/)
- [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/)

## 📄 许可证

本脚本仅供学习和开发测试使用。

使用 NVIDIA API 需遵守：
- [NVIDIA Developer Program 条款](https://developer.nvidia.com/developer-program-terms)
- [NVIDIA API 使用条款](https://build.nvidia.com/terms-of-use)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请访问 [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-nim/678)

---

**最后更新**: 2026-03-17

**版本**: 1.0.0
