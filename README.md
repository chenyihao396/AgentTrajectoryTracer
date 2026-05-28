# AgentTrajectoryTracer

`AgentTrajectoryTracer` 是一个轻量级本地 Agent Trajectory 记录器。

一次 Agent 运行结束后，轨迹会直接写入：

```text
output/<timestamp>_<trace_name>_<trace_id>/
```

适合用来在本地记录和分析：

- LLM 每轮调用的输入、输出、token usage
- tool call 的参数和执行结果
- Agent 的完整执行链路
- 原始 LLM API response
- 自定义评分和事件

## 推荐用法

如果你的代码使用 OpenAI SDK 或 DeepSeek 这类 OpenAI-compatible 接口，只需要把原来的 client 替换为 `tracer.OpenAI(...)`。

原始写法：

```python
from openai import OpenAI

client = OpenAI(...)
response = client.chat.completions.create(...)
```

替换为：

```python
from agent_trajectory_tracer import AgentTrajectoryTracer

# output为输出路径，trace_name和trace_tags类比于这是一趟<trace_name>运行，标签是<trace_tags>（trace_name和trace_tags均用于进行标记本次trajectory tracing）。
tracer = AgentTrajectoryTracer(
    output_root="output",
    trace_name="my_agent",
    trace_tags=["deepseek", "tool-calling"],
)

tracer.OpenAI(
    api_key="...",
    base_url="https://api.deepseek.com",
)

response = tracer.client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    tools=tools,
    stream=False,
)

# 程序退出时会自动尝试落盘；但推荐通过`tracer.flush()`显式调用，便于立即获取输出目录。
output_dir = tracer.flush(output=response.choices[0].message.content)
print(output_dir)
```

第一次调用 `tracer.client.chat.completions.create(...)` 时会自动创建 trace。每次 LLM 调用都会自动记录为一个 `GENERATION` observation。

## Tool 封装

推荐用 `@tracer.tool` 注册工具，然后用 `tracer.execute_tool_call(tool_call)` 执行模型返回的 tool call。

```python
import json
from agent_trajectory_tracer import AgentTrajectoryTracer

tracer = AgentTrajectoryTracer(trace_name="weather_agent")

@tracer.tool
def get_weather(location: str) -> str:
    return json.dumps({
        "location": location,
        "temperature": "72",
        "unit": "fahrenheit",
    })
```

处理模型返回的 tool calls：

```python
response_message = response.choices[0].message
messages.append(response_message.model_dump(exclude_none=True))

if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        messages.append(tracer.execute_tool_call(tool_call))

    second_response = tracer.client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=tools,
        stream=False,
    )
```

`execute_tool_call` 会自动完成：

- 解析 `tool_call.function.arguments`
- 查找已注册的 Python 工具函数
- 执行工具
- 记录 `TOOL` observation
- 保存工具 input/output
- 返回可直接追加到 `messages` 的 tool message

## Claude Agent SDK 用法

先安装可选依赖：

```bash
cd AgentTrajectoryTracer
pip install -e ".[claude]"
```

然后在运行 Claude Agent SDK 前启用 tracing：

```python
from agent_trajectory_tracer import AgentTrajectoryTracer

tracer = AgentTrajectoryTracer(
    output_root="output",
    trace_name="claude_agent_sdk_run",
    trace_tags=["claude-agent-sdk", "openinference"],
)

tracer.ClaudeAgentSDK()

# 这里保持原来的 claude-agent-sdk 调用方式。
# SDK 运行期间产生的 OpenInference spans 会被本地收集。

output_dir = tracer.flush()
print(output_dir)
```

完整示例见：

```bash
python example_claude_agent_sdk.py
```

`tracer.ClaudeAgentSDK()` 会自动完成：

- 安装一个本地内存 OpenTelemetry exporter。
- 调用 `ClaudeAgentSDKInstrumentor().instrument()`。
- 由 `trace_claude_agent_query(...)` 生成一个统一的 `GENERATION` observation，记录 prompt、answer、reasoning_content、usage 和原始 response。
- 将 `openinference.span.kind=TOOL` 映射为 `TOOL`，并把 parent 重定向到对应的 `GENERATION`。
- 将重复的 `ClaudeSDKClient.receive_response` AGENT span 折叠进 `GENERATION.metadata.openinferenceAgentSpans`，避免 observations 中出现两份相同 input/output 的主节点。
- 在 `flush()` 时把保留的 spans 转换为当前项目的本地 trajectory 输出。

推荐的 Claude Agent SDK 输出结构类似：

```text
GENERATION llm.<model>
  ├─ TOOL <tool-name>
  └─ TOOL <tool-name>
```

其中 reasoning 会写入：

```text
GENERATION.output.reasoning_content
```

## mini-SWE-agent 用法

如果你的 Agent 使用 `mini-swe-agent`，可以直接包裹现有 agent。tracer 会自动记录：

- 每次 `model.query(...)` 为一个 `GENERATION` observation。
- 每次 `env.execute(action)` 为一个 `TOOL` observation。
- `GENERATION` 产生的 bash action 会作为对应 `TOOL` 的父节点。
- mini-swe-agent 的原始 LLM response 会写入 `llm_responses.json/jsonl`。

安装可选依赖：

```bash
pip install -e ".[swe]"
```

推荐用法：

```python
from agent_trajectory_tracer import AgentTrajectoryTracer
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

tracer = AgentTrajectoryTracer(
    output_root="output",
    trace_name="mini_swe_agent_run",
    trace_tags=["mini-swe-agent"],
)

agent = DefaultAgent(
    get_model(input_model_name="..."),
    LocalEnvironment(cwd="/path/to/project"),
    ...,
)

result = tracer.trace_mini_swe_agent_run(agent, "Fix the failing test")
output_dir = tracer.flush()
print(output_dir)
```

如果你已经创建好了 agent，也可以只做原地包裹，然后保持原来的调用方式：

```python
agent = tracer.MiniSweAgent(agent)
result = agent.run("Fix the failing test")
tracer.flush(output=result)
```

无需 API key 的确定性 demo：

```bash
python example_mini_swe_agent.py
```

## 输出文件

每次运行会生成一个独立目录，例如：

```text
output/<timestamp>_<trace_name>_<trace_id>/
```

目录内包含：

- `trace.json`：整趟 Agent 任务的封面信息，只应包含初始输入和最终输出。
- `trajectory.json`：完整结构化轨迹，包含 trace、observations、scores、events。
- `observations.jsonl`：每个 observation 一行，方便批处理。
- `llm_responses.json`：所有 LLM 调用的原始 response 输出。
- `llm_responses.jsonl`：每次 LLM 调用一行。
- `scores.jsonl`：评分记录。
- `events.jsonl`：事件记录。
- `summary.json`：计数、usage、cost 汇总。
- `trajectory.md`：易读的层级摘要。
- `output/latest`：指向最近一次运行目录的符号链接。

## trajectory.json 结构

`trajectory.json` 顶层结构如下：

```json
{
  "trace": {},
  "observations": [],
  "scores": [],
  "events": []
}
```

### trace

`trace` 表示一次完整 Agent 任务：

- `input`：初始输入快照，例如最开始的 user/system messages。
- `output`：最终输出。
- `status`：运行状态。
- `tags`：标签。
- `metadata`：额外信息。

### observations

`observations` 表示执行过程中的每一步：

- `GENERATION`：一次 LLM 调用。
- `TOOL`：一次工具调用。
- `AGENT`：一个 Agent 阶段。
- `CHAIN`：链式流程。
- `RETRIEVER`：检索步骤。
- `EVENT`：普通事件。

典型 tool-calling 轨迹：

```text
trace
  ├─ GENERATION: LLM 决定调用工具
  ├─ TOOL: 执行工具
  └─ GENERATION: LLM 基于工具结果生成最终回答
```

### scores

`scores` 用于保存评价结果，例如任务是否成功、输出质量、安全性分数等。

```python
tracer.score(name="task_success", value=True, data_type="BOOLEAN")
```

### events

`events` 用于保存额外事件或异常信息。

```python
tracer.log_event("phase.start", {"phase": "retrieval"})
```
