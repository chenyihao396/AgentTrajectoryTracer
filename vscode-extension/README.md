# Agent Trajectory Tracer Viewer

VSCode extension for visualizing `AgentTrajectoryTracer` `trajectory.json` files.

## Features

- Opens a `trajectory.json` file as an interactive graph.
- Builds a recursive tree layout from `parentObservationId`, computes subtree heights, centers each parent over its children, and renders arrows with SVG paths.
- Shows root observations as a main execution flow connected by head-to-tail arrows.
- Shows child observations as right-side branches connected exactly from parent node boundary to child node boundary.
- Shows TOOL branches with two arrows: parent-to-tool for tool input and tool-to-parent for tool output.
- Normalizes malformed traces for rendering: duplicate ids are renamed in the view, missing parents become roots, and cyclic parent links are broken with an on-screen warning.
- Click any block to inspect:
  - input
  - reasoning
  - output
  - token usage
  - time usage
  - metadata

## Run In Development

Open this folder in VSCode:

```text
AgentTrajectoryTracer/vscode-extension
```

Press `F5` to launch an Extension Development Host.

In the development host, open the `AgentTrajectoryTracer` project and run:

```text
Agent Trajectory: Visualize output/latest
```

or open a `trajectory.json` file and run:

```text
Agent Trajectory: Visualize Current trajectory.json
```

Run these commands from VSCode's Command Palette:

```text
Ctrl+Shift+P
```

Do not type the command title in the integrated terminal. It is a VSCode command,
not a shell command.

If no folder is opened, the extension will ask you to select a `trajectory.json`
file manually.

## Notes

The graph layout uses `parentObservationId` for child blocks. Observations without a valid parent are shown in the main execution flow, sorted by `startTime`.

This viewer intentionally uses a layered execution-forest layout instead of a
general-purpose cyclic graph layout. Agent trajectories are parent-pointer
traces in practice, so this layout preserves the execution hierarchy, keeps
parent-to-child arrows readable, and avoids shipping a large layout engine in
the VSCode webview. If a trace contains invalid graph structure, the viewer
normalizes it into a renderable forest and reports the adjustment above the
graph.
